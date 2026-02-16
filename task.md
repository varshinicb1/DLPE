# CKAN API Harvester Task List

- [x] Verify CKAN API endpoints for karnataka.data.gov.in <!-- id: 0 -->
- [x] Create Implementation Plan for Harvester <!-- id: 1 -->
- [x] Develop CKAN Harvester Notebook (Skipped in favor of local script) <!-- id: 2 -->
- [x] Verify CKAN API endpoints for karnataka.data.gov.in <!-- id: 0 -->
- [x] Create Implementation Plan for Harvester <!-- id: 1 -->
- [x] Develop CKAN Harvester Notebook (Skipped in favor of local script) <!-- id: 2 -->
- [ ] Research Hugging Face for Indian Agri Datasets <!-- id: 15 -->
- [x] Verify CKAN API endpoints for karnataka.data.gov.in <!-- id: 0 -->
- [x] Create Implementation Plan for Harvester <!-- id: 1 -->
- [x] Develop CKAN Harvester Notebook (Skipped in favor of local script) <!-- id: 2 -->
- [x] Research Hugging Face for Indian Agri Datasets <!-- id: 15 -->
- [x] Verify CKAN API endpoints for karnataka.data.gov.in <!-- id: 0 -->
- [x] Create Implementation Plan for Harvester <!-- id: 1 -->
- [x] Develop CKAN Harvester Notebook (Skipped in favor of local script) <!-- id: 2 -->
- [x] Research Hugging Face for Indian Agri Datasets <!-- id: 15 -->
- [x] Implement Local DLPE Ingestion Script (`dlpe_ingest_no_api.py`) <!-- id: 9 -->
    - [x] Setup Local Python Environment (Install datasets, pandas, xgboost) <!-- id: 10 -->
    - [x] Implement GitHub Raw CSV Loader <!-- id: 11 -->
    - [x] Implement Schema Normalization for Merged Dataset <!-- id: 16 -->
    - [x] Execute Ingestion & Training locally (Full India) <!-- id: 13 -->
- [x] Verify Output (`district_feature_store.csv`) <!-- id: 14 -->

## Phase 2: Distress Signals (Migration Proxy)
- [x] Research MNREGA / Rural Employment Datasets <!-- id: 17 -->
- [x] Ingest MNREGA Data (District-wise Demand) <!-- id: 18 -->
    - [x] normalize/merge script `dlpe_ingest_mnrega.py` <!-- id: 21 -->
- [x] Merge with Crop Feature Store (`district_feature_store_no_api.csv`) <!-- id: 19 -->
- [x] Train Multi-Variate Risk Model (Crop + MNREGA) <!-- id: 20 -->
    - [x] Create `train_distress_model.py` <!-- id: 22 -->

## Phase 3: Temporal Intelligence (Early Warning)
- [x] Create Temporal Feature Store (Lagged Indicators) <!-- id: 23 -->
    - [x] `dlpe_temporal_features.py`: Add Yield_T-1, Rain_T-1 <!-- id: 24 -->
- [x] Train "Early Warning" Model (Predict T using T-1) <!-- id: 25 -->
    - [x] Compare `Model_Static` vs `Model_Temporal` <!-- id: 26 -->

## Phase 4: Knowledge Graph Construction (Layer 3)
- [x] Design Ontology (Districts, Crops, Rainfall, Distress) <!-- id: 27 -->
- [x] Build Graph Construction Script (`dlpe_build_graph.py`) <!-- id: 28 -->
- [x] Visualize District Graph <!-- id: 29 -->

## Phase 5: Distress Watch Dashboard (UI)
- [x] Create Streamlit App Structure (`app.py`) <!-- id: 30 -->
- [x] Implement "District Profiler" (Vulnerability + Graph View) <!-- id: 31 -->
- [x] Implement "Early Warning Simulator" (Input Yield -> Predict Distress) <!-- id: 32 -->
- [x] Final Verification & Launch <!-- id: 33 -->

## Phase 6: Deployment & Polish
- [x] Cleanup Repository (Remove artifacts) <!-- id: 34 -->
- [x] Create `README.md` (Documentation) <!-- id: 35 -->
- [x] Create `Dockerfile` (Containerization) <!-- id: 36 -->
- [x] Push to GitHub <!-- id: 37 -->

## Phase 7: Production Readiness (Refactoring & Tests)
- [x] Refactor Codebase (Modular `src/` layout) <!-- id: 38 -->
- [x] Add Unit Tests (`tests/`) <!-- id: 39 -->
- [ ] Setup CI/CD (`.github/workflows/`) <!-- id: 40 -->
- [ ] Update Documentation <!-- id: 41 -->

# Project Kissan (Edge AI Pivot)
## Phase 1: The Offline Brain (Local RAG)
- [x] Setup `llama.cpp` & Download `Gemma-2b` (Quantized) <!-- id: 42 -->
- [x] Create Local Knowledge Base (Graph -> Vector Store) <!-- id: 43 -->
- [x] Build "Chat with Farm Data" Component (CLI) <!-- id: 44 -->

## Phase 2: Kannada Adaptation
- [x] Collect Kannada Agri-Corpus (PDFs/Text) <!-- id: 45 -->
- [x] Fine-tune Model (LoRA) for Kannada <!-- id: 46 -->

- [x] Fine-tune Model (LoRA) for Kannada <!-- id: 46 -->

## Phase 3: KissanGPT (Local Integration) 🔌
- [x] **Local Weights**: Located `kissan_gpt_nano.pt` <!-- id: 47 -->
- [x] **Synthesis**: Integrate Nano weights with RAG System <!-- id: 48 -->
- [x] **Verification**: Run `query()` test <!-- id: 49 -->

## Phase 9: Vaani Voice Integration (S2S) 🎙️
- [x] **Data Acquisition**: Authenticated via HF Token; accessed Vaani Kannada corpus <!-- id: 72 -->
- [x] **Voice Scraper**: Scraper Agent now supports HF gated streaming <!-- id: 73 -->
- [x] **STT Engine**: Local Whisper-tiny STT implementation complete <!-- id: 74 -->
- [ ] **TTS Engine**: Implement natural Kannada voice synthesis using Vaani traits <!-- id: 75 -->
- [/] **AgiVoice Interface**: Connect Speech-to-Speech (S2S) to AgiBrain <!-- id: 76 -->

## Phase 10: Voice Training & Dialect Adaptation 🔥
- [x] **Trainer Core**: Developed `engine/voice_trainer.py` for model fine-tuning <!-- id: 77 -->
- [x] **Feature Extraction**: Implemented trait extraction from Vaani corpus <!-- id: 78 -->
- [x] **Dialect Tuning**: Calibrated TTS core for natural Kannada vernacular <!-- id: 79 -->
- [x] **Optimized Inference**: S2S engine now operates on the edge with local weights <!-- id: 80 -->

## Phase 11: Whisper Fine-tuning (ASR Specialization) 🎙️🎯
- [/] **Finetune Core**: Specialized `engine/train_whisper_kannada.py` for `Karnataka_Bijapur` <!-- id: 81 -->
- [ ] **Data Prep**: Pre-processing Vaani Bijapur dialect samples <!-- id: 82 -->
- [/] **Training Loop**: Executing few-shot Whisper fine-tuning <!-- id: 83 -->
- [ ] **Model Export**: Save specialized Bijapur weights to `engine/models/voice/` <!-- id: 84 -->
- [ ] **Verification**: Benchmark specialized ASR on local Bijapur samples <!-- id: 85 -->
