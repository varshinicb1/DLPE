import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
import os

# Configuration - Using CACHED model
# Based on dir output: models--unsloth--gemma-2b-it-bnb-4bit
MODEL_ID = "unsloth/gemma-2b-it-bnb-4bit"
DATA_PATH = "kissan/data/kannada_corpus/synthetic_agri_facts.txt"
OUTPUT_DIR = "kissan/models/kissan_gemma_lora"

def load_data():
    if not os.path.exists(DATA_PATH):
        # Create dummy if missing for some reason, but should be there
        lines = ["ರೈತರಿಗೆ ಕೃಷಿ ಮಾಹಿತಿ ಬೇಕು.", "ಮಳೆಯನ್ನು ಅವಲಂಬಿಸಿ ಕೃಷಿ ಮಾಡಲಾಗುತ್ತದೆ."]
    else:
        with open(DATA_PATH, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
    return Dataset.from_dict({"text": lines})

def train_gemma_cached():
    print(f"🚀 Using CACHED Model: {MODEL_ID}")
    
    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    
    # 2. Load Base Model (Already on disk)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"📥 Loading model onto {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, 
        torch_dtype=torch.float16 if device=="cuda" else torch.float32,
        device_map="auto"
    )
    
    # 3. Apply LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        r=8,           # Lightweight rank
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "o_proj", "k_proj"]
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Prepare Data
    dataset = load_data()
    def tokenize(element):
        return tokenizer(element["text"], truncation=True, padding="max_length", max_length=128)
    
    tokenized_datasets = dataset.map(tokenize, batched=True)
    
    # 5. Trainer
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=1, # Quick proof of work
        learning_rate=2e-4,
        fp16=(device=="cuda"),
        logging_steps=1,
        save_strategy="no"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    print("🚜 Starting Training (Offline Mode)...")
    trainer.train()
    
    # 6. Save
    model.save_pretrained(OUTPUT_DIR)
    print(f"✅ Training Complete. Model saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train_gemma_cached()
