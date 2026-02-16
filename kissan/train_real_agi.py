import torch
import pandas as pd
import os
import logging
import sys
from tqdm import tqdm

# Add local path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import KissanGPT, KissanConfig
from tokenizers import Tokenizer

# Absolute Real Fine-tuner
# Goal: Re-weight the Nano model using actual Indian agricultural data (NPK, Rainfall, pH mappings)

MODEL_PATH = "kissan/models/kissan_gpt_nano.pt"
DATA_DIR = "kissan/data"
TOKENIZER_PATH = "kissan/models/kissan_tokenizer/tokenizer.json"
OUTPUT_PATH = "kissan/models/kissan_gpt_agi_real.pt"

def train():
    logging.info("🧠 Starting Cumulative Factual Alignment (AGI Scaler)...")
    
    # 1. Load ALL Real Data (Cumulative Learning)
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    dataframes = []
    for f in all_files:
        try:
            df_temp = pd.read_csv(f)
            # Only include if it has agricultural mapping columns (N, P, K or label)
            if 'label' in df_temp.columns:
                dataframes.append(df_temp)
                logging.info(f"   📈 Added dataset: {f}")
        except: pass
    
    if not dataframes:
        logging.error("❌ No real agricultural datasets found to train on.")
        return

    df = pd.concat(dataframes, ignore_index=True)
    logging.info(f"📊 Total Training Knowledge Base: {len(df)} points.")

    # 2. Setup Model & Tokenizer
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    
    # Load with base vocab size (155) first
    model = KissanGPT(vocab_size=155) 
    logging.info("   🧩 Loading base weights (Vocab: 155)...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    
    # Resize embeddings and head for the new 5000-vocab AGI tokenizer
    logging.info(f"   ✂️ Resizing model to {vocab_size} for AGI fine-tuning...")
    model.token_embedding_table = torch.nn.Embedding(vocab_size, KissanConfig.n_embd)
    model.lm_head = torch.nn.Linear(KissanConfig.n_embd, vocab_size)
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 3. Fine-tuning Loop (Grounding the weights in Real facts)
    # We turn NPK rows into text: "Soil N50 P40 K30 is best for [LABEL]"
    epochs = 2
    for epoch in range(epochs):
        logging.info(f"🚀 Epoch {epoch+1}/{epochs}...")
        total_loss = 0
        
        # Shuffle for diversity
        df_shuffled = df.sample(frac=1).reset_index(drop=True)
        
        for i, row in tqdm(df_shuffled.iterrows(), total=len(df_shuffled)):
            # Formulate the factual sentence
            text = f"Soil N:{row['N']} P:{row['P']} K:{row['K']} pH:{row['ph']:.1f} Rainfall:{row['rainfall']:.0f}mm is ideal for {row['label']}."
            tokens = tokenizer.encode(text).ids
            
            # Ensure sequence length fits
            if len(tokens) > 64: tokens = tokens[:64]
            
            idx = torch.tensor([tokens], dtype=torch.long)
            targets = idx.clone() # Simple causal LM training
            
            # Forward pass
            logits, loss = model(idx, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                logging.debug(f"   Step {i}, Loss: {loss.item():.4f}")
                
        logging.info(f"✅ Epoch {epoch+1} Avg Loss: {total_loss / len(df_shuffled):.4f}")

    # 4. Save the "Real AGI" weights
    torch.save(model.state_dict(), OUTPUT_PATH)
    logging.info(f"🏆 Saved 'Absolute Real' AGI weights to {OUTPUT_PATH}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train()
