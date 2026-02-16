import os
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🕵️‍♂️ [INSPECTION] - %(message)s')

def inspect_vaani():
    dataset_name = "ARTPARK-IISc/Vaani"
    language = "kannada"
    
    logging.info(f"🔍 Inspecting Hugging Face Dataset: {dataset_name} (Language: {language})...")
    
    try:
        # Load the dataset in streaming mode to avoid full download (Vaani is massive)
        ds = load_dataset(dataset_name, language, split="train", streaming=True)
        
        # Access the underlying dataset info if available (via features)
        # Note: In streaming mode, features are often accessible on the iterable dataset
        sample_iter = iter(ds)
        first_sample = next(sample_iter)
        
        logging.info("--- Dataset Sample Details ---")
        for key, value in first_sample.items():
            if key == 'audio':
                logging.info(f"Field: {key} | Details: { {k: v for k,v in value.items() if k != 'array'} } (Array data hidden for brevity)")
            else:
                logging.info(f"Field: {key} | Value: {value}")
                
        logging.info("--- Inspecting multiple samples for dialect/content variety ---")
        count = 1
        for i in range(5):
            sample = next(sample_iter)
            logging.info(f"Sample {count+i+1} Transcription: {sample.get('transcription', 'N/A')}")
            logging.info(f"   - District: {sample.get('district', 'Unknown')} | Metadata: {sample.get('metadata', '{}')}")
            
        logging.info("✅ Inspection Complete.")
        
    except Exception as e:
        logging.error(f"❌ Inspection Failed: {e}")

if __name__ == "__main__":
    inspect_vaani()
