import os
import logging
import torch
from datasets import load_dataset, Audio
from transformers import WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🎙️🎯 [WHISPER-TRAINER] - %(message)s')

# --- Training Config ---
MODEL_NAME = "openai/whisper-tiny" # Small footprint for edge compatibility
LANGUAGE = "kannada"
DATASET_NAME = "ARTPARK-IISc/Vaani"
SUBSET_NAME = "Karnataka_Bijapur"
OUTPUT_DIR = "engine/models/voice/whisper-kannada-bijapur"

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

def train_whisper():
    logging.info(f"🚀 Starting Whisper Fine-tuning on {DATASET_NAME} ({LANGUAGE})...")

    try:
        # 1. Load Processor and Models
        logging.info("   🔍 Loading models and processors...")
        feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_NAME)
        tokenizer = WhisperTokenizer.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe")
        processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task="transcribe")
        model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
        
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []

        # 2. Data Preparation
        logging.info(f"   📦 Loading Vaani {SUBSET_NAME} Subset (Streaming Mode)...")
        # Use SUBSET_NAME for district-specific fine-tuning
        dataset = load_dataset(DATASET_NAME, SUBSET_NAME, split="train", streaming=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        def prepare_dataset(batch):
            audio = batch["audio"]
            # Robust Decoding: Ensure audio array is valid
            audio_array = audio.get("array")
            if audio_array is None:
                # Potential backend issue (e.g. libtorchcodec)
                # We attempt to use soundfile/librosa directly if possible in a production env
                raise ValueError("Audio decoding failed. Ensure FFmpeg or soundfile is available.")
            
            batch["input_features"] = feature_extractor(audio_array, sampling_rate=audio["sampling_rate"]).input_features[0]
            batch["labels"] = tokenizer(batch["transcription"]).input_ids
            return batch

        logging.info("   🛠️ Pre-processing samples...")
        train_samples = []
        it = iter(dataset)
        for i in range(5): # Very small sample for verification
            logging.info(f"      Processing sample {i+1}...")
            try:
                sample = next(it)
                processed = prepare_dataset(sample)
                train_samples.append(processed)
            except Exception as e:
                logging.warning(f"      ⚠️ Failed to process sample {i+1}: {e}")
                continue
                
        if not train_samples:
            logging.error("❌ No samples processed. Training aborted.")
            return

        # 3. Training Arguments
        logging.info("   ⚙️ Setting up training arguments...")
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=2,
            max_steps=5, # Iteration limit for demo
            gradient_checkpointing=True,
            fp16=False, # Set to True if GPU available
            evaluation_strategy="no",
            save_steps=5,
            logging_steps=1,
            report_to=["none"],
            push_to_hub=False,
        )

        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

        # 4. Initialize Trainer
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_samples,
            data_collator=data_collator,
            tokenizer=processor.feature_extractor,
        )

        # 5. Execute Specialization
        logging.info("   🔥 Executing Fine-tuning Loop (Factual Alignment)...")
        trainer.train()
        
        # 6. Save specialized weights
        trainer.save_model(OUTPUT_DIR)
        processor.save_pretrained(OUTPUT_DIR)
        logging.info(f"✅ Specialized Whisper Model saved to: {OUTPUT_DIR}")

    except Exception as e:
        logging.error(f"❌ Training Loop Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    train_whisper()
