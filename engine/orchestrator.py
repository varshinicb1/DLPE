import os
import time
import logging
import subprocess
import shutil

# --- Configuration ---
WATCH_DIR = "engine/data/ingest"
ARCHIVE_DIR = "engine/data/archive"
TRAIN_SCRIPT = "engine/train_real_agi.py"
BRAIN_SCRIPT = "engine/brain.py"

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
    """Monitors WATCH_DIR subdirectories and triggers domain-specific AGI evolution"""
    if not os.path.exists(WATCH_DIR): os.makedirs(WATCH_DIR)
    if not os.path.exists(ARCHIVE_DIR): os.makedirs(ARCHIVE_DIR)

    logging.info(f"👀 Monitoring {WATCH_DIR} subfolders for any profession/domain...")

    while True:
        # Scan for domain subfolders
        domains = [d for d in os.listdir(WATCH_DIR) if os.path.isdir(os.path.join(WATCH_DIR, d))]
        
        for domain in domains:
            domain_path = os.path.join(WATCH_DIR, domain)
            new_files = [f for f in os.listdir(domain_path) if f.endswith('.csv') or f.endswith('.json')]
            
            if new_files:
                logging.info(f"🏗️ Detected {len(new_files)} new files for domain: {domain}")
                
                # 1. Trigger Domain Retraining
                # We pass the domain context if needed, but for now, cumulative scaler is best
                if run_step(["python", TRAIN_SCRIPT], f"Factual Retraining ({domain})"):
                    
                    # 2. Trigger Brain Re-Indexing
                    if run_step(["python", BRAIN_SCRIPT], f"Brain Update ({domain})"):
                        
                        # 3. Move to Permanent Dataset Store
                        target_store = f"engine/data/datasets/{domain}"
                        if not os.path.exists(target_store): os.makedirs(target_store)
                        
                        for f in new_files:
                            shutil.move(os.path.join(domain_path, f), os.path.join(target_store, f))
                        logging.info(f"✅ Domain '{domain}' is now updated and evolved.")
        
        time.sleep(10)

if __name__ == "__main__":
    orchestrate()
