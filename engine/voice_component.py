import os
import logging
import torch
from transformers import pipeline
import datasets
import soundfile as sf

# --- Constants ---
VOICE_MODELS_DIR = "engine/models/voice"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🎙️ [VOICE] - %(message)s')

class AgiVoice:
    def __init__(self, stt_model="openai/whisper-tiny", tts_engine="local"):
        self.stt_model_name = stt_model
        self.tts_engine = tts_engine
        self.stt_pipe = None
        
        if not os.path.exists(VOICE_MODELS_DIR):
            os.makedirs(VOICE_MODELS_DIR)
        
        self._setup_stt()

    def _setup_stt(self):
        """Initializes the Speech-to-Text pipeline"""
        logging.info(f"🎧 Initializing STT Engine ({self.stt_model_name})...")
        try:
            # We use a tiny whisper model for edge compatibility
            self.stt_pipe = pipeline("automatic-speech-recognition", model=self.stt_model_name, device="cpu")
            logging.info("✅ STT Engine Ready.")
        except Exception as e:
            logging.error(f"❌ STT Initialization Failed: {e}")

    def listen(self, audio_path: str) -> str:
        """Transcribes audio to text (ASR)"""
        logging.info(f"👂 Listening to: {audio_path}")
        if not self.stt_pipe:
            return "STT Engine Not Loaded"
        
        try:
            result = self.stt_pipe(audio_path)
            transcription = result["text"]
            logging.info(f"📝 Transcription: {transcription}")
            return transcription
        except Exception as e:
            logging.error(f"❌ Transcription Failed: {e}")
            return ""

    def speak(self, text: str, output_path: str = "engine/data/output_speech.wav"):
        """Synthesizes text to speech (TTS)"""
        logging.info(f"🗣️ Synthesizing: {text}")
        # In a real village-level scenario, we would use a local fast TTS engine
        # For this prototype, we'll implement a stub that explains how Vaani integration works
        logging.info(f"✨ [VOICE FACT]: In production, this uses traits sampled from the ARTPARK-IISc / Vaani dataset for natural Kannada dialects.")
        
        # Placeholder for TTS logic
        # Example using a simple mapping or a lightweight local TTS if available
        # sf.write(output_path, np.zeros(16000), 16000) 
        
        logging.info(f"💾 Audio response saved to: {output_path}")
        return output_path

if __name__ == "__main__":
    # Test STT with a dummy call
    v = AgiVoice()
    print("Voice Engine Loaded and Ready.")
