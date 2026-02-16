import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

INPUT_FILE = "district_feature_store_v2_mnrega.csv"
OUTPUT_FILE = "district_feature_store_v3_temporal.csv"

def main():
    logging.info(f"📂 Loading {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        logging.error("File not found.")
        return

    # Sort for shift operations: District -> Crop -> Year
    # We need to be careful. MNREGA data in our V2 store is currently STATIC (snapshot).
    # TO ENABLE TRUE TEMPORAL MODELLING, we must acknowledge that MNREGA column currently repeats for all years.
    # HOWEVER, Crop/Rain data IS time-series (1997-2015).
    # So we can create "Previous Year's Crop Failure" as a feature.
    
    # We will treat the 'Yield' and 'Avg_Annual_Rainfall' as dynamic.
    # Note: 'Avg_Annual_Rainfall' in source was actually a 'Normal' (static average), 
    # but for a REAL system we would use actual yearly rainfall.
    # Since we only have static rainfall 'Normal' in this dataset, we can't lag it meaningfully (it's constant).
    # BUT 'Yield' and 'Production' change every year. We CAN lag those.
    
    logging.info("🔄 Generating Lag Features (Year T-1, T-2)...")
    
    df = df.sort_values(by=['State_Name', 'District_Name', 'Crop', 'Season', 'Crop_Year'])
    
    # Group by unique Time-Series identifier
    # Unique ID = District + Crop + Season
    
    # Lag 1: Last Year's Yield
    df['Yield_Lag1'] = df.groupby(['State_Name', 'District_Name', 'Crop', 'Season'])['Yield'].shift(1)
    
    # Lag 1: Last Year's Production (Volume shock)
    df['Production_Lag1'] = df.groupby(['State_Name', 'District_Name', 'Crop', 'Season'])['Production'].shift(1)
    
    # Delta: Growth or Decline?
    df['Yield_Pct_Change'] = df.groupby(['State_Name', 'District_Name', 'Crop', 'Season'])['Yield'].pct_change()
    
    # Rolling Mean (3 Years) - Sustained Stress
    df['Yield_Rolling_3Y'] = df.groupby(['State_Name', 'District_Name', 'Crop', 'Season'])['Yield'].transform(lambda x: x.rolling(window=3).mean())

    # Fill NA for first years
    df = df.fillna(0)
    
    # Feature Engineering: "Shock" Indicator
    # If Yield dropped by > 20% compared to last year -> severe shock
    df['Yield_Shock'] = np.where(df['Yield_Pct_Change'] < -0.2, 1, 0)
    
    logging.info(f"💾 Saving Temporal Feature Store to {OUTPUT_FILE}")
    logging.info(f"   New Features: Yield_Lag1, Production_Lag1, Yield_Pct_Change, Yield_Shock")
    df.to_csv(OUTPUT_FILE, index=False)
    logging.info(f"Shape: {df.shape}")

if __name__ == "__main__":
    main()
