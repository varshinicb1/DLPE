import pandas as pd
import os
import logging

# Soil Health Intelligence Module
# Data source: Regional norms for Jharkhand (Acidic/Laterite) and Karnataka (Red/Black)
OUTPUT_PATH = "kissan/data/soil_health_store.csv"

def main():
    logging.info("🌱 Ingesting Soil Health Data...")
    
    # Representative Soil profiles for Districts
    districts = [
        {"District": "GARHWA", "State": "Jharkhand", "Soil_Type": "Laterite/Red Sandy", "pH": 5.8, "Nitrogen": "Low", "Phosphorus": "Medium", "Potash": "Medium"},
        {"District": "PALAMU", "State": "Jharkhand", "Soil_Type": "Red Loamy", "pH": 6.2, "Nitrogen": "Medium", "Phosphorus": "Low", "Potash": "Medium"},
        {"District": "GADAG", "State": "Karnataka", "Soil_Type": "Black Cotton", "pH": 7.8, "Nitrogen": "Medium", "Phosphorus": "Medium", "Potash": "High"},
        {"District": "BELAGAVI", "State": "Karnataka", "Soil_Type": "Deep Black", "pH": 8.1, "Nitrogen": "Medium", "Phosphorus": "High", "Potash": "High"},
    ]
    
    df = pd.DataFrame(districts)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"✅ Saved {len(df)} soil health records to {OUTPUT_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
