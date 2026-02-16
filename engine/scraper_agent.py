import os
import logging
import requests
import time
import pandas as pd
from urllib.parse import urljoin

# --- Data Hunter Config ---
DOMAINS = {
    "agriculture": [
        "https://raw.githubusercontent.com/ShikhaYadav123/SQL--EDA-on-Food-Prices-in-India/main/RAW%20File%20-%20ind-food-prices.csv",
        "https://raw.githubusercontent.com/arzzahid66/Optimizing_Agricultural_Production/master/Crop_recommendation.csv"
    ],
    "animal_husbandry": [
        "https://raw.githubusercontent.com/Gaiban-Khan/Crop-Recommendation-System/main/Data/crop_recommendation.csv" 
        # (Will be replaced with actual vet datasets as found)
    ]
}

INGEST_DIR = "engine/data/ingest"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🕵️‍♂️ [SCRAPER] - %(message)s')

class DataHungryAgent:
    def __init__(self):
        if not os.path.exists(INGEST_DIR): os.makedirs(INGEST_DIR)
        for domain in DOMAINS:
            d_path = os.path.join(INGEST_DIR, domain)
            if not os.path.exists(d_path): os.makedirs(d_path)

    def download_file(self, url, domain):
        """Downloads a real dataset and drops it into the AGI ingestion zone"""
        filename = url.split('/')[-1]
        target_path = os.path.join(INGEST_DIR, domain, filename)
        
        logging.info(f"🎯 Hunting: {url} -> {target_path}")
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(target_path, 'wb') as f:
                f.write(response.content)
            
            logging.info(f"✅ Success: Dataset captured for {domain}.")
            return True
        except Exception as e:
            logging.error(f"❌ Failed to capture {url}: {e}")
            return False

    def hunt_24_7(self):
        """Infinite loop for data hunting"""
        logging.info("🚀 Data Hungry Agent is now active. Scanning the web for Absolute Real evidence...")
        
        while True:
            for domain, urls in DOMAINS.items():
                for url in urls:
                    # Logic to avoid re-downloading identical files in a short window
                    # (In a real 24/7 env, we'd check headers/ETags)
                    self.download_file(url, domain)
            
            logging.info("💤 Sleeping before next hunt cycle (1 hour intervals for 24/7 mode)...")
            time.sleep(3600) 

if __name__ == "__main__":
    agent = DataHungryAgent()
    # For demonstration/initial seed, we run one pass
    for domain, urls in DOMAINS.items():
        for url in urls:
            agent.download_file(url, domain)
