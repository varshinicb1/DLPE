import requests
import pandas as pd
import numpy as np
import re
import sys
import argparse
import logging
import warnings
from typing import List, Dict, Any, Optional
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
warnings.filterwarnings('ignore')

class CKANHarvester:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {'api-key': api_key} if api_key else {}
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def _make_request(self, action: str, params: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Helper to make API requests with error handling."""
        url = f"{self.base_url}/api/3/action/{action}"
        try:
            # Add strict timeout
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            if not data.get('success'):
                # Some portals return success: false with an error
                logging.error(f"API Error from {url}: {data.get('error')}")
                return None
            return data['result']
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {action}: {e}")
            return None
        except ValueError as e:
            logging.error(str(e))
            return None

    def search_packages(self, query: str, rows: int = 10) -> List[Dict[str, Any]]:
        """Wrapper for package_search."""
        params = {'q': query, 'rows': rows}
        result = self._make_request('package_search', params)
        if result:
            count = result.get('count', 0)
            results = result.get('results', [])
            logging.info(f"Found {count} packages for query '{query}'. Fetching {len(results)}.")
            return results
        return []

    def get_resource_data(self, resource_id: str, limit: int = 10000) -> pd.DataFrame:
        """Wrapper for datastore_search to fetch actual data."""
        params = {'resource_id': resource_id, 'limit': limit}
        result = self._make_request('datastore_search', params)
        if result and 'records' in result:
            return pd.DataFrame(result['records'])
        return pd.DataFrame()

def normalize_schema(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    """Standardizes DataFrame columns to [District, Year, Feature, Value]."""
    # Clean column names
    df.columns = df.columns.astype(str).str.lower().str.strip()
    
    # 1. Identify District Column
    district_col = next((c for c in df.columns if 'dist' in c), None)
    
    # 2. Identify Year Column
    year_col = next((c for c in df.columns if 'year' in c), None)
    
    if not district_col:
        logging.warning(f"Skipping {source_name}: No district column found. Columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # 3. Identify Value Columns
    # Exclude metadata columns
    exclude_cols = [district_col, year_col, '_id', 's.no.', 'sl.no.', 'state', 'state_name']
    value_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Standardize
    df = df.rename(columns={district_col: 'District'})
    if year_col:
        df = df.rename(columns={year_col: 'Year'})
    else:
        # Infer year from title or default to 2024 for testing
        year_match = re.search(r'\d{4}', source_name)
        df['Year'] = year_match.group(0) if year_match else '2022'

    # Melt to Long Format
    # Ensure District and Year are preserved
    if 'Year' not in df.columns:
        df['Year'] = '2022'

    df_melt = df.melt(id_vars=['District', 'Year'], value_vars=value_cols, 
                      var_name='Feature_Raw', value_name='Value')
    
    # Clean Feature Names
    df_melt['Feature'] = df_melt['Feature_Raw'].apply(lambda x: re.sub(r'[^a-zA-Z0-9_]', '_', str(x)).strip('_').lower())
    
    # Clean Values
    df_melt['Value'] = pd.to_numeric(df_melt['Value'], errors='coerce')
    df_melt = df_melt.dropna(subset=['Value'])
    
    return df_melt[['District', 'Year', 'Feature', 'Value']]

def train_baseline_model(df: pd.DataFrame):
    """Trains a simple XGBoost model to predict crop stress (yield/production)."""
    # Simply check for any target column
    targets = [c for c in df.columns if 'yield' in c or 'production' in c]
    if not targets:
        logging.warning("No target variable (yield/production) found in feature store.")
        logging.info(f"Available features: {df.columns.tolist()}")
        return
    
    target_col = targets[0]
    logging.info(f"🎯 Training Baseline XGBoost Model for Target: {target_col}")
    
    X = df.drop(columns=['District', 'Year', target_col])
    # Ensure all X are numeric
    X = X.select_dtypes(include=[np.number])
    y = df[target_col]
    
    if X.empty:
        logging.warning("❌ No numeric features available for training.")
        return

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    logging.info(f"✅ Model Results -> MSE: {mse:.2f}, R²: {r2:.2f}")

def main():
    parser = argparse.ArgumentParser(description='DLPE Ingestion Script')
    parser.add_argument('--api-key', required=True, help='API Key for data.gov.in')
    parser.add_argument('--base-url', default='https://karnataka.data.gov.in', help='CKAN Base URL')
    args = parser.parse_args()

    # 1. Init Harvester
    logging.info(f"Initializing Harvester for {args.base_url}...")
    harvester = CKANHarvester(args.base_url, args.api_key)

    # 2. Search
    search_terms = ["district", "production", "rainfall", "crop"]
    all_resources = []
    seen_packages = set()

    for term in search_terms:
        packages = harvester.search_packages(term, rows=5)
        for pkg in packages:
            pkg_id = pkg['id']
            if pkg_id in seen_packages: continue
            seen_packages.add(pkg_id)
            
            # Filter for relevance
            title_lower = pkg['title'].lower()
            if 'district' not in title_lower and 'state' not in title_lower:
                continue

            for res in pkg['resources']:
                fmt = res['format'].upper()
                if fmt in ['CSV', 'API', 'JSON', '']:
                    all_resources.append({
                        'title': pkg['title'],
                        'resource_id': res['id'],
                        'url': res['url'],
                        'format': fmt
                    })

    logging.info(f"Total relevant resources found: {len(all_resources)}")
    if not all_resources:
        logging.warning("No resources found. Check API Key validity or Base URL availability.")
        return

    # 3. Ingest & Normalize
    normalized_data = []
    for res in all_resources:
        logging.info(f"Fetching {res['title']} (ID: {res['resource_id']})...")
        try:
            df = harvester.get_resource_data(res['resource_id'])
            if not df.empty:
                norm_df = normalize_schema(df, res['title'])
                if not norm_df.empty:
                    normalized_data.append(norm_df)
                    logging.info(f"  -> Extracted {len(norm_df)} records.")
                else:
                    logging.warning(f"  -> Schema mismatch. Skipping.")
            else:
                logging.warning(f"  -> Empty data returned.")
        except Exception as e:
            logging.error(f"  -> Failed: {e}")

    # 4. Feature Store & Training
    if normalized_data:
        logging.info("Building Feature Store...")
        final_df = pd.concat(normalized_data, ignore_index=True)
        
        # Standardize Districts
        final_df['District'] = final_df['District'].astype(str).str.upper().str.strip()
        
        # Pivot to Wide Format
        feature_store = final_df.pivot_table(index=['District', 'Year'], 
                                             columns='Feature', 
                                             values='Value', 
                                             aggfunc='mean').reset_index()
        feature_store = feature_store.fillna(0)
        
        output_file = 'district_feature_store.csv'
        feature_store.to_csv(output_file, index=False)
        logging.info(f"💾 Feature Store saved to {output_file} (Shape: {feature_store.shape})")
        
        # 5. Train
        if feature_store.shape[0] > 10: # Only train if enough data
            train_baseline_model(feature_store)
        else:
            logging.warning("Not enough data points to train model.")
    else:
        logging.warning("❌ No data normalized. Exiting.")

if __name__ == "__main__":
    main()
