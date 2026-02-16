import os
import logging
import argparse
from engine.brain import AgiBrain, DEFAULT_PERSONALITY
from engine.voice_component import AgiVoice
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - 🏛️ [S2S-SYSTEM] - %(message)s')

class AgiS2SOrchestrator:
    def __init__(self, domain="agriculture"):
        # 1. Load Domain Personality
        domain_file = os.path.join("domains", f"{domain}.json")
        personality = DEFAULT_PERSONALITY
        if os.path.exists(domain_file):
            with open(domain_file, 'r', encoding='utf-8') as f:
                personality = json.load(f)
        
        # 2. Setup Brain
        M_PATH = os.path.join("engine", "models", "agi_core_real.pt")
        if not os.path.exists(M_PATH): M_PATH = "engine/models/kissan_gpt_agi_real.pt"
        
        self.brain = AgiBrain(model_path=M_PATH, data_path="engine/data", domain_config=personality)
        
        # 3. Setup Voice
        self.voice = AgiVoice()
        
    def run_voice_loop(self, audio_input_path: str):
        """Processes a Speech-to-Speech query loop"""
        logging.info("🏁 Starting S2S Inference Loop...")
        
        # Step A: Listen (STT)
        user_text = self.voice.listen(audio_input_path)
        if not user_text:
            logging.error("🔇 No speech detected.")
            return
        
        # Step B: Consult Brain (RAG + LLM)
        response_text = self.brain.query(user_text)
        
        # Step C: Speak (TTS)
        # We strip markdown for the TTS engine
        clean_response = response_text.replace("*", "").replace("#", "").replace("=", "")
        output_audio = self.voice.speak(clean_response)
        
        print("\n" + "🚀" * 20)
        print(f"👂 USER SAID: {user_text}")
        print("-" * 40)
        print(f"🧠 BRAIN RESPONSE:\n{response_text}")
        print("-" * 40)
        print(f"🔊 AUDIO OUTPUT: {output_audio}")
        print("🚀" * 20 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgiEngine S2S CLI")
    parser.add_argument("--domain", type=str, default="agriculture", help="Domain: agriculture, animal_husbandry")
    parser.add_argument("--input", type=str, required=True, help="Path to input .wav file")
    args = parser.parse_args()

    orchestrator = AgiS2SOrchestrator(domain=args.domain)
    orchestrator.run_voice_loop(args.input)
