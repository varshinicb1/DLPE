import os
import logging
from datasets import load_dataset
import soundfile as sf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🎧 [SAMPLE-EXTRACTOR] - %(message)s')

def save_vaani_sample():
    output_path = r"C:\Users\varsh\.gemini\antigravity\brain\6db64765-7966-408d-a075-37c115d6b037\engine\data\test_input.wav"
    logging.info(f"🧬 Extracting Absolute Real audio sample to: {output_path}...")
    
    try:
        # Load one sample from Vaani
        ds = load_dataset("ARTPARK-IISc/Vaani", "kannada", split="train", streaming=True)
        sample = next(iter(ds))
        
        audio_data = sample['audio']['array']
        sampling_rate = sample['audio']['sampling_rate']
        
        # Save to disk
        if not os.path.exists("engine/data"): os.makedirs("engine/data")
        sf.write(output_path, audio_data, sampling_rate)
        
        logging.info(f"✅ Sample Captured: '{sample['transcription']}'")
        logging.info(f"💾 File saved to: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"❌ Extraction Failed: {e}")
        return None

if __name__ == "__main__":
    save_vaani_sample()
