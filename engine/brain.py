import os
import logging
import json
import networkx as nx
import pandas as pd
from typing import List, Dict
import torch
import sys

# Ensure engine directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import AgiCoreModel, KissanConfig
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# --- Domain Personality Loader ---
# This allows the AI to pivot between Agriculture, Animal Husbandry, etc.
DEFAULT_PERSONALITY = {
    "name": "General AGI",
    "advice_header": "Expert Advice",
    "mappings": {
        "HIGH": {"status": "🔴 HIGH RISK", "advice": "Action not recommended. Extreme caution required."},
        "MODERATE": {"status": "🟡 CAUTION", "advice": "Proceed with care. Continuous monitoring recommended."},
        "LOW": {"status": "🟢 SAFE", "advice": "Optimal conditions detected. System clearance granted."}
    }
}

class AgiBrain:
    def __init__(self, model_path: str, data_path: str, domain_config=None):
        self.model_path = model_path
        self.data_path = data_path
        self.personality = domain_config if domain_config else DEFAULT_PERSONALITY
        
        self.vector_db = None
        self.llm_model = None
        self.embedding_model = None
        
        self._setup_knowledge_base()
        self._load_llm()

    def _setup_knowledge_base(self):
        """Dynamic Knowledge Ingestion: Scales to any profession based on CSV columns"""
        logging.info(f"🧠 Initializing '{self.personality['name']}' Brain...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        chroma_client = chromadb.Client(Settings(persist_directory=os.path.join(self.data_path, "chroma_db"), is_persistent=True))
        collection_name = "agi_knowledge_base"
        
        try:
            chroma_client.delete_collection(collection_name)
        except: pass
            
        self.vector_db = chroma_client.create_collection(collection_name)
        
        # 1. Ingest Relationships (Graphs)
        graph_path = os.path.join(self.data_path, "knowledge_graph.gml")
        if os.path.exists(graph_path):
            G = nx.read_gml(graph_path)
            logging.info(f"   🕸️ Ingesting Graph Relationships...")
            docs, ids = [], []
            for u, v, data in G.edges(data=True):
                fact = f"Relation: {u} -> {v} (Weight: {data.get('weight', 0)})"
                docs.append(fact); ids.append(f"G_{u}_{v}")
            if docs: self.vector_db.add(documents=docs, embeddings=self.embedding_model.encode(docs).tolist(), ids=ids)

        # 2. Dynamic Data Scaling (Profession-Agnostic, Domain-Specific)
        # We only ingest data relevant to the current domain (e.g. agriculture, animal_husbandry)
        domain_name = self.personality.get('domain', 'general')
        data_dir = os.path.join(self.data_path, "datasets", domain_name)
        
        if os.path.exists(data_dir):
            all_csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            logging.info(f"   📂 Scanning {len(all_csvs)} {domain_name} datasets for Knowledge Scaling...")
            
            for csv_file in all_csvs:
                csv_path = os.path.join(data_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    logging.info(f"      📖 Learning from: {csv_file}")
                    
                    # Pattern-based ingestion
                    docs, ids = [], []
                    for idx, row in df.head(500).iterrows():
                        # Create a curated fact from the row
                        fact_parts = [f"{col}: {val}" for col, val in row.to_dict().items()]
                        fact = f"Domain {domain_name} fact: " + ", ".join(fact_parts)
                        docs.append(fact); ids.append(f"F_{domain_name}_{csv_file}_{idx}")
                    
                    if docs: self.vector_db.add(documents=docs, embeddings=self.embedding_model.encode(docs).tolist(), ids=ids)
                except Exception as e:
                    logging.warning(f"      ⚠️ Failed to learn from {csv_file}: {e}")
            
        logging.info(f"✅ Framework Ready with {self.vector_db.count()} factual indices.")

    def _load_llm(self):
        """Loads Local AgiCoreModel weights"""
        if os.path.exists(self.model_path):
            # Base logic for Nano core
            self.llm_model = AgiCoreModel(vocab_size=5000) 
            try:
                self.llm_model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.llm_model.eval()
            except: pass

    def query(self, user_text: str) -> str:
        logging.info(f"❓ User Engagement: {user_text}")
        query_vec = self.embedding_model.encode([user_text]).tolist()
        results = self.vector_db.query(query_embeddings=query_vec, n_results=5)
        retrieved_facts = results['documents'][0]
        context = "\n".join(retrieved_facts)
        
        # Determine Severity/Action based on core context (Placeholder for advanced gating)
        level = "LOW" 
        if "risk" in context.lower() or "high" in context.lower(): level = "HIGH"
        
        info = self.personality['mappings'].get(level, self.personality['mappings']['LOW'])
        
        return f"""
{info['status']}
========================================
📢 **{self.personality['advice_header']}**:
{info['advice']}

🔍 **Ground Truth Evidence**:
{'- ' + context.replace('\n', '\n- ')}
"""

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)
    
    parser = argparse.ArgumentParser(description="AgiEngine Brain CLI")
    parser.add_argument("--domain", type=str, default="agriculture", help="Domain to load (agriculture, animal_husbandry)")
    parser.add_argument("--query", type=str, default="What is the real market price of Rice?", help="Query to run")
    args = parser.parse_args()

    # Load Domain Personality
    domain_file = os.path.join("domains", f"{args.domain}.json")
    personality = DEFAULT_PERSONALITY
    if os.path.exists(domain_file):
        with open(domain_file, 'r', encoding='utf-8') as f:
            personality = json.load(f)
            logging.info(f"✨ Loaded Personality: {personality['name']}")

    M_PATH = os.path.join(CURRENT_DIR, "models", "agi_core_real.pt")
    if not os.path.exists(M_PATH): M_PATH = "engine/models/kissan_gpt_agi_real.pt"
    
    BRAIN = AgiBrain(model_path=M_PATH, data_path="engine/data", domain_config=personality)
    
    print("\n" + "="*40)
    print(f"🏛️ AGI Personality: {personality['name']}")
    print("="*40)
    print(BRAIN.query(args.query))
    print("="*40 + "\n")
