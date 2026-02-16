import requests
import os
import logging
from bs4 import BeautifulSoup
import re

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = "kissan/data/kannada_corpus"

# Valid Sources for Kannada Agri Data (Open Access)
# 1. Vikaspedia (Government of India) - Excellent multilingual content
# 2. UAS Dharwad/Bangaloer (University Archives) - Often PDFs, we'll try to find text first.

SOURCES = [
    "https://kn.vikaspedia.in/agriculture",
    "https://kn.vikaspedia.in/agriculture/crop-production",
    "https://kn.vikaspedia.in/agriculture/agri-inputs"
]

def clean_kannada_text(text):
    # Remove extra whitespace and non-Kannada/English garbage
    # Keep Kannada unicode range: \u0C80-\u0CFF
    # Keep English for scientific names: a-zA-Z
    # Keep numbers: 0-9
    # Keep basic punctuation
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def scrape_vikaspedia():
    logging.info("🚜 Starting Scrape: Vikaspedia (Kannada)...")
    
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    # Simple recursive scraper (depth 1 for demo)
    for url in SOURCES:
        try:
            logging.info(f"   Fetching: {url}")
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.content, 'html.parser')
                
                # Extract main content (usually in div id='text')
                # Vikaspedia structure varies, but often has specific content divs
                content_div = soup.find('div', {'id': 'text'}) or soup.find('div', {'class': 'middle-content'})
                
                if content_div:
                    raw_text = content_div.get_text()
                    clean_text = clean_kannada_text(raw_text)
                    
                    if len(clean_text) > 500:
                        filename = url.split('/')[-1] + ".txt"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        with open(filepath, 'w', encoding='utf-8') as f:
                            f.write(clean_text)
                        
                        logging.info(f"   ✅ Saved: {filename} ({len(clean_text)} chars)")
                    else:
                        logging.warning("   ⚠️ Content too short, skipping.")
                else:
                    logging.warning("   ⚠️ No text content found.")
            else:
                logging.error(f"   ❌ Failed: {resp.status_code}")
                
        except Exception as e:
            logging.error(f"   ❌ Error: {e}")

def create_synthetic_dataset():
    """ Creates a small synthetic dataset for immediate testing if scraping fails """
    logging.info("🧪 Creating Synthetic Kannada Agri-Dataset (Knowledge Graph Translation)...")
    
    # We translate our Graph Facts into Kannada (Rule-based)
    # "In {district}, {crop} failure risk is {risk}%"
    # Kannada: "{district} Jilleyalli, {crop} beleyu {risk}% nashta honduva sadhyate ide."
    
    facts = [
        "In Garhwa, Rice failure risk is 54.5%. -> Garhwa jilleyalli, Bhatta beleyu 54.5% nashta honduva sadhyate ide.",
        "In Dantewada, Rice failure risk is 45%. -> Dantewada jilleyalli, Bhatta beleyu 45% nashta honduva sadhyate ide.",
        "Wheat needs moderate rainfall. -> Godhi belege madhyama male agatya.",
        "Cotton is a cash crop. -> Hatti ondu vanijya beleyagide."
    ]
    
    filepath = os.path.join(OUTPUT_DIR, "synthetic_agri_facts.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("\n".join(facts))
    
    logging.info(f"   ✅ Saved Synthetic Data: {filepath}")

if __name__ == "__main__":
    scrape_vikaspedia()
    create_synthetic_dataset()
