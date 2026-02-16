import pandas as pd
import numpy as np
import requests
import io
import logging
import re
import argparse

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

URL_MNREGA = "https://raw.githubusercontent.com/Arpita-deb/NREGA-Data-Analysis/main/NREGA%20Data.csv"
INPUT_FEATURE_STORE = "district_feature_store_no_api.csv"
OUTPUT_FEATURE_STORE = "district_feature_store_v2_mnrega.csv"

def download_mnrega_data(url: str) -> pd.DataFrame:
    """Downloads MNREGA CSV (semicolon separated)."""
    logging.info(f"⬇️ Downloading MNREGA data from {url}...")
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        # The preview showed semicolon delimiters
        return pd.read_csv(io.StringIO(response.content.decode('utf-8')), sep=';')
    except Exception as e:
        logging.error(f"❌ Failed to download MNREGA data: {e}")
        return pd.DataFrame()

def normalize_text(series):
    """Standardizes text to uppercase and stripped."""
    return series.astype(str).str.upper().str.strip()

def merge_datasets(mnrega_df: pd.DataFrame, crop_fs_df: pd.DataFrame):
    if mnrega_df.empty or crop_fs_df.empty:
        logging.error("Input DataFrames are empty.")
        return pd.DataFrame()

    logging.info("🧹 Cleaning MNREGA Data...")
    # Preview columns: state_name;district_name;Total No. of JobCards issued;Total No. of Workers;...
    mnrega_df.columns = mnrega_df.columns.str.strip()
    
    # Rename for consistency
    mnrega_df = mnrega_df.rename(columns={
        'state_name': 'State_Name',
        'district_name': 'District_Name',
        'Total No. of JobCards issued': 'MNREGA_JobCards_Issued',
        'Total No. of Workers': 'MNREGA_Workers_Registered',
        'Total No. of Active Job Cards': 'MNREGA_Active_JobCards',
        'Total No. of Active Workers': 'MNREGA_Active_Workers'
    })
    
    mnrega_df['District_Name'] = normalize_text(mnrega_df['District_Name'])
    mnrega_df['State_Name'] = normalize_text(mnrega_df['State_Name'])
    
    # Select subset of features
    mnrega_features = mnrega_df[['State_Name', 'District_Name', 
                                 'MNREGA_JobCards_Issued', 'MNREGA_Active_Workers']]
    
    logging.info("🧹 Cleaning Crop Feature Store...")
    crop_fs_df['District_Name'] = normalize_text(crop_fs_df['District_Name'])
    crop_fs_df['State_Name'] = normalize_text(crop_fs_df['State_Name'])
    
    # MNREGA data in this CSV is likely a snapshot (static), not time-series.
    # We will treat it as a "static district socioeconomic profile" for now.
    # Real-world: needs Year-Month grain.
    
    logging.info("🔗 Merging MNREGA into Feature Store...")
    # Left join to keep all crop history
    merged_df = pd.merge(crop_fs_df, mnrega_features, 
                         on=['State_Name', 'District_Name'], 
                         how='left')
    
    # Fill NA with 0 or median? 
    # If missing, it implies no MNREGA data found for that district name match.
    # We'll fill with NaN for now to see match rate, then fill 0.
    match_rate = 1 - merged_df['MNREGA_JobCards_Issued'].isna().mean()
    logging.info(f"✅ Implementation Match Rate: {match_rate:.2%}")
    
    merged_df[['MNREGA_JobCards_Issued', 'MNREGA_Active_Workers']] = \
        merged_df[['MNREGA_JobCards_Issued', 'MNREGA_Active_Workers']].fillna(0)
        
    return merged_df

def main():
    # 1. Load Crop Feature Store (Layer 1)
    try:
        logging.info(f"📂 Loading {INPUT_FEATURE_STORE}...")
        crop_fs = pd.read_csv(INPUT_FEATURE_STORE)
    except FileNotFoundError:
        logging.error(f"❌ '{INPUT_FEATURE_STORE}' not found. Run Layer 1 ingestion first.")
        return

    # 2. Download MNREGA
    mnrega_df = download_mnrega_data(URL_MNREGA)
    
    # 3. Merge
    final_df = merge_datasets(mnrega_df, crop_fs)
    
    if not final_df.empty:
        # Save
        final_df.to_csv(OUTPUT_FEATURE_STORE, index=False)
        logging.info(f"💾 Saved Layer 2 Feature Store to {OUTPUT_FEATURE_STORE}")
        logging.info(f"Shape: {final_df.shape}")
        logging.info(f"Columns: {final_df.columns.tolist()}")
    else:
        logging.error("❌ Merge failed.")

if __name__ == "__main__":
    main()
