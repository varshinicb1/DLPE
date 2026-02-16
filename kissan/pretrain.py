import torch
import torch.nn as nn
from torch.nn import functional as F
from tokenizers import Tokenizer
import os
import logging
import time
from model import KissanGPT, KissanConfig

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TOKENIZER_PATH = "kissan/models/kissan_tokenizer/tokenizer.json"
DATA_PATH = "kissan/data/kannada_corpus/synthetic_agri_facts.txt"
MODEL_SAVE_PATH = "kissan/models/kissan_gpt_nano.pt"

# Hyperparameters for Demo Training
BATCH_SIZE = 2 # Small batch for CPU demo
MAX_ITERS = 50
LEARNING_RATE = 3e-4
EVAL_INTERVAL = 10
DEVICE = KissanConfig.device

def get_batch(data_tensor):
    # Generate a small batch of data of inputs x and targets y
    ix = torch.randint(len(data_tensor) - KissanConfig.block_size, (BATCH_SIZE,))
    x = torch.stack([data_tensor[i:i+KissanConfig.block_size] for i in ix])
    y = torch.stack([data_tensor[i+1:i+KissanConfig.block_size+1] for i in ix])
    x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

def train():
    logging.info(f"🚜 Initializing KissanGPT Pre-training on {DEVICE}...")
    
    # 1. Load Tokenizer
    if not os.path.exists(TOKENIZER_PATH):
        logging.error(f"❌ Tokenizer not found at {TOKENIZER_PATH}")
        return
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()
    logging.info(f"   Loaded Tokenizer (Vocab: {vocab_size})")
    
    # Update Config with actual vocab size
    KissanConfig.vocab_size = vocab_size

    # 2. Load Data
    logging.info(f"   Loading Data from {DATA_PATH}...")
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    encoded = tokenizer.encode(text)
    data = torch.tensor(encoded.ids, dtype=torch.long)
    logging.info(f"   Total Tokens: {len(data)}")

    if len(data) <= KissanConfig.block_size:
        logging.error("❌ Data too short for block size. Add more text!")
        # Fake duplication for demo if data is tiny
        data = torch.cat([data] * 50) 
        logging.warning("   ⚠️ Multiplied data for demo purposes.")

    # 3. Initialize Model
    model = KissanGPT()
    m = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Loop
    logging.info("🚀 Starting Training Loop...")
    start_time = time.time()
    
    for iter in range(MAX_ITERS):
        # Sample batch
        xb, yb = get_batch(data)

        # Evaluate loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if iter % EVAL_INTERVAL == 0:
            logging.info(f"   Step {iter}: Loss {loss.item():.4f}")

    logging.info(f"✅ Training Finished in {time.time()-start_time:.2f}s")
    
    # 5. Save Model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logging.info(f"💾 Model saved to {MODEL_SAVE_PATH}")

    # 6. Inference Demo
    logging.info("🗣️ Generating Sample Text...")
    context = torch.tensor([tokenizer.encode("Garhwa").ids], dtype=torch.long, device=DEVICE)
    generated_ids = m.generate(context, max_new_tokens=20)[0].tolist()
    generated_text = tokenizer.decode(generated_ids)
    logging.info(f"   🤖 KissanGPT says: {generated_text}")

if __name__ == "__main__":
    train()
