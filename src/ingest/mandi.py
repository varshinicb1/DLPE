import pandas as pd
import numpy as np
import os
import logging

# Market Price Intelligence Module
# Data source: Simulated based on AGMARKNET/CEDA norms for Jharkhand/Karnataka
OUTPUT_PATH = "kissan/data/mandi_prices_store.csv"

def main():
    logging.info("🏪 Ingesting Mandi Market Prices...")
    
    # Representative Mandis for Jharkhand/Karnataka
    markets = [
        {"District": "Garhwa", "Market": "Garhwa", "Commodity": "Rice", "State": "Jharkhand"},
        {"District": "Palamu", "Market": "Daltonganj", "Commodity": "Rice", "State": "Jharkhand"},
        {"District": "Gadag", "Market": "Gadag", "Commodity": "Rice", "State": "Karnataka"},
        {"District": "Palamu", "Market": "Daltonganj", "Commodity": "Wheat", "State": "Jharkhand"},
    ]
    
    data = []
    # Generate 12 months of daily-ish price data (Simulated Volatility)
    for mk in markets:
        base_price = 2200 if mk["Commodity"] == "Rice" else 1800
        for m in range(1, 13):
            # seasonal variance
            seasonal_factor = 1 + 0.1 * np.sin(2 * np.pi * m / 12)
            for d in range(1, 4): # Sample days per month
                price = base_price * seasonal_factor + np.random.normal(0, 50)
                data.append({
                    "State": mk["State"],
                    "District": mk["District"].upper(),
                    "Market": mk["Market"],
                    "Commodity": mk["Commodity"],
                    "Modal_Price": round(price, 2),
                    "Month": m,
                    "Year": 2025
                })
                
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    logging.info(f"✅ Saved {len(df)} market price records to {OUTPUT_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
