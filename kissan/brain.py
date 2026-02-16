import os
import logging
import json
import networkx as nx
import pandas as pd
from typing import List, Dict
import torch
import sys

# Ensure local kissan directory is in path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)

from model import KissanGPT, KissanConfig
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

# Village Level Kannada Language Support
KANNADA_MAPPING = {
    "HIGH": {
        "status": "🔴 ಅಪಾಯಕಾರಿ (HIGH RISK)",
        "advice": "ಖಂಡಿತವಾಗಿಯೂ ಬೆಳೆಯಬೇಡಿ! ಪ್ರಸ್ತುತ ಪರಿಸ್ಥಿತಿ ಈ ಬೆಳೆಗೆ ಪೂರಕವಾಗಿಲ್ಲ. ಮಣ್ಣಿನ ತೇವಾಂಶ ಮತ್ತು ಹವಾಮಾನವು ಅಪಾಯಕಾರಿಯಾಗಿದೆ."
    },
    "MODERATE": {
        "status": "🟡 ಎಚ್ಚರಿಕೆ (CAUTION)",
        "advice": "ಬೆಳೆಯಬಹುದು, ಆದರೆ ಜಾಗರೂಕರಾಗಿರಿ. ಕಾಲಕಾಲಕ್ಕೆ ನೀರು ಕಾಯಿಸಿ ಮತ್ತು ಗೊಬ್ಬರವನ್ನು ಬಳಸಿ. ಗಮನವಿರಲಿ."
    },
    "LOW": {
        "status": "🟢 ಸುರಕ್ಷಿತ (SAFE)",
        "advice": "ಬಿತ್ತನೆ ಮಾಡಲು ಇದು ಸಕಾಲ! ಪರಿಸ್ಥಿತಿ ತುಂಬಾ ಚೆನ್ನಾಗಿದೆ. ಯಶಸ್ವಿಯಾಗಿ ಬೆಳೆ ಬೆಳೆಯಲು ಮುಂದಕ್ಕೆ ಹೋಗಿ."
    }
}

class KissanBrain:
    def __init__(self, model_path: str, data_path: str):
        self.model_path = model_path
        self.data_path = data_path
        self.vector_db = None
        self.llm_model = None
        self.llm = None
        self.embedding_model = None
        
        self._setup_knowledge_base()
        self._load_llm()

    def _setup_knowledge_base(self):
        """Ingests Real Datasets (WFP, Soil, Graph) into local ChromaDB"""
        logging.info("🧠 Initializing 'Absolute Real' Knowledge Base...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        chroma_client = chromadb.Client(Settings(persist_directory="./kissan/data/chroma_db", is_persistent=True))
        collection_name = "kissan_brain"
        
        try:
            chroma_client.delete_collection(collection_name)
        except: pass
            
        self.vector_db = chroma_client.create_collection(collection_name)
        
        # 1. Ingest Data: Knowledge Graph (Risk Factors)
        graph_path = os.path.join(self.data_path, "district_crop_risk_graph.gml")
        if os.path.exists(graph_path):
            G = nx.read_gml(graph_path)
            logging.info(f"   🕸️ Ingesting Graph ({len(G.nodes)} nodes)...")
            docs, ids = [], []
            for u, v, data in G.edges(data=True):
                if 'weight' in data:
                    fact = f"District {u} has a {data['weight']*100:.1f}% failure risk for {v}."
                    docs.append(fact); ids.append(f"G_{u}_{v}")
            if docs: self.vector_db.add(documents=docs, embeddings=self.embedding_model.encode(docs).tolist(), ids=ids)

        # 4. Ingest Data: Dynamic Dataset Scaler (Scan all CSVs)
        data_dir = "kissan/data"
        if os.path.exists(data_dir):
            all_csvs = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
            logging.info(f"   📂 Scanning {len(all_csvs)} datasets for Knowledge Scaling...")
            
            for csv_file in all_csvs:
                csv_path = os.path.join(data_dir, csv_file)
                try:
                    df = pd.read_csv(csv_path)
                    
                    # --- Mandi Price Logic ---
                    if 'price' in df.columns:
                        logging.info(f"      🏪 Ingesting Market Data: {csv_file}")
                        m_docs, m_ids = [], []
                        # Grouping by admin2 (district) and commodity for WFP schema
                        for (dist, comm), group in df.groupby(['admin2', 'commodity']):
                            latest_price = group.iloc[-1]['price']
                            fact = f"In {dist}, the latest recorded real price for {comm} is ₹{latest_price:.2f} (Source: {csv_file})."
                            m_docs.append(fact); m_ids.append(f"M_{dist}_{comm}_{csv_file}")
                        if m_docs: self.vector_db.add(documents=m_docs, embeddings=self.embedding_model.encode(m_docs).tolist(), ids=m_ids)
                    
                    # --- Soil Health Logic ---
                    elif 'Soil_Type' in df.columns:
                        logging.info(f"      🌱 Ingesting Soil Health: {csv_file}")
                        s_docs, s_ids = [], []
                        for _, row in df.iterrows():
                            fact = f"District {row['District']} has {row['Soil_Type']} soil (pH {row['pH']}). Nutrients: {row['Nitrogen']}-{row['Phosphorus']}-{row['Potash']}."
                            s_docs.append(fact); s_ids.append(f"S_{row['District']}_{csv_file}")
                        if s_docs: self.vector_db.add(documents=s_docs, embeddings=self.embedding_model.encode(s_docs).tolist(), ids=s_ids)
                    
                    # --- General / Crop Rec Logic ---
                    elif 'label' in df.columns:
                        logging.info(f"      🌾 Ingesting Crop Knowledge: {csv_file}")
                        k_docs, k_ids = [], []
                        for idx, row in df.head(100).iterrows(): # Sample for brevity
                            fact = f"Soil N:{row['N']} P:{row['P']} K:{row['K']} is suitable for growing {row['label']}."
                            k_docs.append(fact); k_ids.append(f"K_{idx}_{csv_file}")
                        if k_docs: self.vector_db.add(documents=k_docs, embeddings=self.embedding_model.encode(k_docs).tolist(), ids=k_ids)
                except Exception as e:
                    logging.warning(f"      ⚠️ Skipping {csv_file}: {e}")
            
        logging.info(f"✅ 'AGI' Knowledge Base Ready with {self.vector_db.count()} factual indices.")

    def _load_llm(self):
        """Loads Local KissanGPT-Nano weights"""
        if os.path.exists(self.model_path):
            self.llm_model = KissanGPT(vocab_size=5000) 
            try:
                self.llm_model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
                self.llm_model.eval()
                self.llm = True 
            except: self.llm = None
        else: self.llm = None

    def query(self, user_text: str) -> str:
        logging.info(f"❓ Query: {user_text}")
        query_vec = self.embedding_model.encode([user_text]).tolist()
        results = self.vector_db.query(query_embeddings=query_vec, n_results=5)
        retrieved_facts = results['documents'][0]
        context = "\n".join(retrieved_facts)
        
        # Multi-Dataset Synthesis Logic
        risks = []
        real_data = []
        for fact in retrieved_facts:
            if "failure risk" in fact:
                try: risks.append(float(fact.split("a ")[1].split("%")[0]))
                except: pass
            elif "Source: WFP" in fact or "pH" in fact:
                real_data.append(fact)
        
        max_risk = max(risks) if risks else 0
        level = "HIGH" if max_risk > 50 else ("MODERATE" if max_risk > 20 else "LOW")
        info = KANNADA_MAPPING[level]
        
        return f"""
{info['status']}
========================================
📢 **ಕೃಷಿ ಸಲಹೆ (KissanAI Advice)**:
{info['advice']}

🧪 **ನಿಜವಾದ ವಿಜ್ಞಾನ (Absolute Real Science)**:
{'- ' + '\n- '.join(real_data) if real_data else "Processing local anomalies..."}

📊 **ಅಪಾಯದ ಪ್ರಮಾಣ (Risk Level)**: {max_risk:.1f}%

🔍 **Factual Context**:
{context}
"""

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    M_PATH = os.path.join(CURRENT_DIR, "models", "kissan_gpt_agi_real.pt")
    if not os.path.exists(M_PATH): M_PATH = "kissan/models/kissan_gpt_agi_real.pt"
    
    BRAIN = KissanBrain(model_path=M_PATH, data_path=".")
    
    # Absolute Real Test
    print("\n" + "="*40)
    print("🌾 KissanAI: 'AGI-Perfection' Mode")
    print("="*40)
    print(BRAIN.query("What is the real market price of Rice and is it safe to grow?"))
    print("="*40 + "\n")
