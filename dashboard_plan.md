# Distress Watch Dashboard Plan

## Goal
Build an interactive **Streamlit** application to visualize the District-Level Predictive Engine (DLPE).

## Architecture
*   **Framework**: Streamlit (Python)
*   **Data Sources**:
    1.  `district_feature_store_v3_temporal.csv` (Historical Data & Temporal Features)
    2.  `district_vulnerability_ranking.csv` (Risk Scores)
    3.  `district_crop_risk_graph.gml` (Knowledge Graph)
*   **Models**:
    *   Load the XGBoost models (saved as JSON/Pickle in previous steps - *Correction*: I need to save the trained models first!).

## Features
1.  **🚨 National Risk Map**:
    *   Choropleth map of India showing District Vulnerability Scores.
    *   Top 10 "Red Alert" Districts list.

2.  **🔍 District Profiler**:
    *   User selects State -> District.
    *   **Metrics**: Avg Yield, Avg Rainfall, Vulnerability Score.
    *   **Graph View**: Interactive network graph of the district's crop portfolio (using `streamlit-agraph` or `pyvis`).

3.  **🔮 Early Warning Simulator**:
    *   "What if?" Analysis.
    *   Sliders to adjust current year's Yield for major crops.
    *   **Output**: Predicted MNREGA Demand (Distress) for next year.

## Tasks
1.  **Save Models**: Update training scripts to save model files (`.json`).
2.  **App Skeleton**: `app.py` with sidebar navigation.
3.  **Visualization Components**: Plotly for charts, PyVis/NetworkX for graphs.
