import pandas as pd
import numpy as np
import requests
import io
import logging
import argparse
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Constants
URL_CROP_PROD = "https://raw.githubusercontent.com/ritveek19/EDA_CropProduction/master/crop_production.csv"
URL_RAINFALL = "https://raw.githubusercontent.com/Osprey-DS/Rainfall-Measurement-in-INDIA-in-the-time-of-1901-to-2015/master/district%20wise%20rainfall%20normal.csv"

def download_csv(url: str, name: str) -> pd.DataFrame:
    """Downloads a CSV from a URL and returns a DataFrame."""
    logging.info(f"⬇️ Downloading {name} from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return pd.read_csv(io.StringIO(response.content.decode('utf-8')))
    except Exception as e:
        logging.error(f"❌ Failed to download {name}: {e}")
        return pd.DataFrame()

def normalize_text(series):
    """Standardizes text to uppercase and stripped."""
    return series.astype(str).str.upper().str.strip()

def process_data(crop_df: pd.DataFrame, rain_df: pd.DataFrame):
    """Merges and processes the datasets."""
    if crop_df.empty:
        logging.error("Crop data is empty. Cannot proceed.")
        return pd.DataFrame()

    # 1. Standardize Crop Data
    logging.info("🧹 Cleaning Crop Data...")
    crop_df['District_Name'] = normalize_text(crop_df['District_Name'])
    crop_df['State_Name'] = normalize_text(crop_df['State_Name'])
    
    # Drop rows with missing values
    crop_df = crop_df.dropna()
    
    # Calculate Yield
    # Yield = Production / Area. (Handle division by zero)
    crop_df = crop_df[crop_df['Area'] > 0]
    crop_df['Yield'] = crop_df['Production'] / crop_df['Area']
    
    # 2. Standardize Rainfall Data (Static Normals)
    if not rain_df.empty:
        logging.info("🧹 Cleaning Rainfall Data...")
        rain_df['DISTRICT'] = normalize_text(rain_df['DISTRICT'])
        
        # Select relevant columns: District, Annual Rainfall
        rain_features = rain_df[['DISTRICT', 'ANNUAL']].rename(columns={
            'DISTRICT': 'District_Name', 
            'ANNUAL': 'Avg_Annual_Rainfall'
        })
        
        # 3. Merge
        logging.info("🔗 Merging Datasets...")
        # Left join because we want to keep all crop records even if rainfall is missing
        merged_df = pd.merge(crop_df, rain_features, on='District_Name', how='left')
        
        # Fill missing rainfall with state average or median (simple imputation)
        median_rain = merged_df['Avg_Annual_Rainfall'].median()
        merged_df['Avg_Annual_Rainfall'] = merged_df['Avg_Annual_Rainfall'].fillna(median_rain)
    else:
        logging.warning("⚠️ Rainfall data missing. Proceeding with Crop data only.")
        merged_df = crop_df
        merged_df['Avg_Annual_Rainfall'] = 0

    return merged_df

def train_baseline_model(df: pd.DataFrame):
    """Trains a simple XGBoost model to predict YIELD."""
    logging.info("🎯 Training Baseline XGBoost Model (Target: Yield)")
    
    # Features: Area, Avg_Annual_Rainfall, Year (encoded?), Crop (encoded)
    # For simplicity in this script, we'll use numeric + OneHot for Crop
    
    # Limit to top crops to avoid massive dimensionality
    top_crops = df['Crop'].value_counts().nlargest(10).index
    df_filtered = df[df['Crop'].isin(top_crops)]
    
    # One-Hot Encode 'Crop' and 'Season'
    X = pd.get_dummies(df_filtered[['Area', 'Avg_Annual_Rainfall', 'Crop_Year', 'Crop', 'Season']], 
                       columns=['Crop', 'Season'], drop_first=True)
    
    y = df_filtered['Yield']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    logging.info(f"✅ Model Results -> MSE: {mse:.2f}, R²: {r2:.2f}")
    
    # Save Feature Store
    output_file = 'district_feature_store_no_api.csv'
    df_filtered.to_csv(output_file, index=False)
    logging.info(f"💾 Feature Store saved to {output_file} (Shape: {df_filtered.shape})")

def main():
    parser = argparse.ArgumentParser(description='DLPE Ingestion (No API)')
    # Optional state filter
    parser.add_argument('--state', help='Filter by State Name (e.g., Karnataka)', default=None)
    args = parser.parse_args()

    # 1. Download
    crop_df = download_csv(URL_CROP_PROD, "Crop Production")
    rain_df = download_csv(URL_RAINFALL, "Rainfall Normals")

    # 2. Filter State if requested
    if args.state and not crop_df.empty:
        state_upper = args.state.upper()
        logging.info(f"🔍 Filtering for State: {state_upper}")
        crop_df = crop_df[crop_df['State_Name'].str.upper() == state_upper]

    # 3. Process
    final_df = process_data(crop_df, rain_df)

    # 4. Train
    if not final_df.empty:
        train_baseline_model(final_df)
    else:
        logging.error("Final dataset is empty.")

if __name__ == "__main__":
    main()
