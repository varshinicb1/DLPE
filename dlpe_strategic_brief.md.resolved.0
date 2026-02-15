# District-Level Predictive Engine (DLPE) - Strategic Brief

## 🎯 Core Vision
A unified, geospatial, time-aware engine that predicts district-level risk and opportunity signals across India. It fuses satellite, weather, infrastructure, demographic, economic, and telecom data to serve as a **national-scale decision intelligence system**.

## 🧠 MVP Scope: Crop Stress → Migration Risk
**Goal**: Predict distinct crop stress events and their cascading effect on migration/urban pressure.

### Primary Predictive Target: **Crop Failure / Yield Stress**
*   **Signals**:
    *   Rainfall anomalies (IMD/State Data)
    *   Soil moisture & NDVI (Satellite)
    *   Agricultural Production/Yield (CKAN Data)
*   **Impact**:
    *   Features for Migration Risk (Outbound migration spikes, MNREGA demand)
    *   Early compensation & storage planning.

## 🏗️ Architecture

### Layer 1: Unified District Feature Store (Current Focus)
*   **Granularity**: District × Month (or Year for initial agricultural baseline).
*   **Schema**: `[District, Time, Feature, Value]`
*   **Data Sources**:
    *   `data.gov.in` (Karnataka Pilot): Production, Rainfall, Yield.
    *   *Future*: Satellite (Sentinel/Landsat), Telecom (aggregates), MNREGA (public portal).

### Layer 2: Temporal & Graph Models
*   **Baseline**: XGBoost (Structured data).
*   **Advanced**: Graph Neural Networks (GNN) to model district adjacency and cascading risks (e.g., drought in District A affects labor in District B).

### Layer 3: Policy Simulation
*   "What-if" scenarios for rainfall deficits or policy interventions.

## 🚀 Execution Roadmap
1.  **Ingestion (Current)**: Build CKAN Harvester for agricultural & rainfall data (DONE).
2.  **Feature Store**: Normalize district names and align temporal resolutions.
3.  **Baseline Model**: Train XGBoost to predict Yield Stress (DONE in V1 Notebook).
4.  **Expansion**: Integrate Migration/MNREGA signals.
5.  **Graph Scale-up**: Implement GNN for inter-district dependencies.
