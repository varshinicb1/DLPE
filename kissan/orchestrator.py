import os
import time
import logging
import subprocess
import shutil

# --- Configuration ---
WATCH_DIR = "kissan/data/ingest"
ARCHIVE_DIR = "kissan/data/archive"
TRAIN_SCRIPT = "kissan/train_real_agi.py"
BRAIN_SCRIPT = "kissan/brain.py"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🚀 [ORCHESTRATOR] - %(message)s')

def run_step(command_list, description):
    logging.info(f"Running: {description}...")
    try:
        result = subprocess.run(command_list, capture_output=True, text=True, check=True)
        logging.info(f"✅ {description} successful.")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ {description} failed: {e.stderr}")
        return False

def orchestrate():
    """Monitors WATCH_DIR and triggers AGI retraining pipeline"""
    if not os.path.exists(WATCH_DIR): os.makedirs(WATCH_DIR)
    if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR)

    logging.info(f"👀 Monitoring {WATCH_DIR} for new city/agricultural data...")

    while True:
        # 1. Detect New Data
        new_files = [f for f in os.listdir(WATCH_DIR) if f.endswith('.csv') or f.endswith('.json')]
        
        if new_files:
            logging.info(f"🏗️ Detected {len(new_files)} new dataset(s): {new_files}")
            
            # 2. Trigger Retraining (Factual Alignment)
            # This incorporates the NEW facts into the model weights
            if run_step(["python", TRAIN_SCRIPT], "Factual Model Retraining"):
                
                # 3. Trigger Brain Re-Indexing
                # This refreshes the Vector DB with the latest facts
                if run_step(["python", BRAIN_SCRIPT], "Knowledge Base Re-Indexing"):
                    
                    # 4. Archive Data
                    for f in new_files:
                        shutil.move(os.path.join(WATCH_DIR, f), os.path.join(ARCHIVE_DIR, f))
                    logging.info("♻️ Data archived. Model is now updated and ready.")
        
        time.sleep(10) # Poll every 10 seconds

if __name__ == "__main__":
    orchestrate()
