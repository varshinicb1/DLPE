import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os

# Configuration
# TinyLlama is 1.1B, trained on 3 Trillion tokens. It is verified "Quality".
MODEL_ID = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
DATA_PATH = "kissan/data/kannada_corpus/synthetic_agri_facts.txt"
OUTPUT_DIR = "kissan/models/kissan_1B_lora"

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing data: {DATA_PATH}")
        
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Text Completion Format
    return Dataset.from_dict({"text": lines})

def train_lora():
    print(f"🚀 Initializing Fast-Track Training (LoRA) on {MODEL_ID}...")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Fix padding
    
    # 2. Load Base Model (1.1B)
    print("📥 Downloading Base Model (TinyLlama-1.1B)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load in 16-bit to save memory (requires ~2.5GB VRAM)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16 if device=="cuda" else torch.float32)
    model.to(device)
    
    # 3. Apply LoRA (The "Speed" Secret)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=16,           # Rank (Higher = smarter but slower)
        lora_alpha=32,  # Scaling
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] # Target Attention layers
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters() # Show how efficient this is!
    
    # 4. Prepare Data
    dataset = load_data()
    def tokenize(element):
        return tokenizer(element["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_datasets = dataset.map(tokenize, batched=True)
    
    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=4, # Fits easily in GPU
        gradient_accumulation_steps=4,
        num_train_epochs=3, # Quick runs
        learning_rate=2e-4,
        fp16=(device=="cuda"),
        logging_steps=10,
        save_strategy="epoch"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("🚜 Starting LoRA Training...")
    trainer.train()
    
    # 6. Save
    model.save_pretrained(OUTPUT_DIR)
    print(f"✅ Fast-Track Complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_lora()
