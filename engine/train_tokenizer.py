from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
import os
import logging

# Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
CORPUS_DIR = "kissan/data/kannada_corpus"
MODEL_DIR = "kissan/models/kissan_tokenizer"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_tokenizer():
    logging.info("🔤 Initializing Custom Tokenizer (BPE)...")
    
    # 1. Initialize
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    
    # 2. Gather Data files
    files = [os.path.join(CORPUS_DIR, f) for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]
    if not files:
        logging.error("❌ No training data found in kissan/data/kannada_corpus")
        return

    logging.info(f"   Found {len(files)} files to train on: {files}")

    # 3. Trainer Config
    # Vocab size: 5000 (Small for Nano model, usually 32k-50k for large models)
    # Special tokens: CLS/SEP for structure
    trainer = BpeTrainer(
        vocab_size=5000, 
        special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
    )
    
    # 4. Train
    tokenizer.train(files, trainer)
    logging.info(f"   ✅ Training Complete. Vocab Size: {tokenizer.get_vocab_size()}")
    
    # 5. Post-Processing (for BERT/RoBERTa style)
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[
            ("[CLS]", tokenizer.token_to_id("[CLS]")),
            ("[SEP]", tokenizer.token_to_id("[SEP]")),
        ],
    )
    
    # 6. Save
    save_path = os.path.join(MODEL_DIR, "tokenizer.json")
    tokenizer.save(save_path)
    logging.info(f"   💾 Tokenizer saved to {save_path}")
    
    # 7. Test
    sample_text = "Garhwa jilleyalli Bhatta beleyu nashta"
    encoded = tokenizer.encode(sample_text)
    logging.info(f"   🧪 Test Encoding '{sample_text}': {encoded.tokens}")

if __name__ == "__main__":
    train_tokenizer()
