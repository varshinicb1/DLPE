# Knowledge Graph Ontology (Phase 4)

## Goal
Transform the flat temporal feature store (`district_feature_store_v3_temporal.csv`) into a **Knowledge Graph** to enable semantic reasoning (e.g., "Find all districts growing Rice that had a Yield Shock and High Distress").

## Ontology Design

### Nodes (Entities)
1.  **District**
    *   Properties: `Name`, `State`
    *   ID: `State_District` (e.g., "KARNATAKA_BELGAUM")
2.  **Crop**
    *   Properties: `Name`, `Season`
    *   ID: `Crop_Season` (e.g., "Rice_Kharif")
3.  **Year**
    *   Properties: `Year`
    *   ID: `Year` (e.g., "2005")

### Edges (Relationships)
1.  **(District) -[:GROWS {area, production, yield}]-> (Crop)**
    *   Represents agricultural activity.
    *   Attributes: `Area`, `Production`, `Yield`, `Yield_Shock` (Boolean)
2.  **(District) -[:EXPERIENCED {rainfall}]-> (Year)**
    *   Represents climatic conditions.
    *   Attributes: `Avg_Annual_Rainfall`
3.  **(District) -[:HAS_DEMAND {job_cards}]-> (DistressSignal)**
    *   *Simplification*: Direct property on District or Edge to a Distress Node?
    *   *Decision*: **(District) -[:REPORTED_DISTRESS {job_cards, workers}]-> (Year)**
        *   This links the distress to the specific year (or static snapshot if data is limited).

### Graph Schema (NetworkX)
*   We will use `networkx` for construction and initial traversal.
*   **Query Capabilities**:
    *   "Get neighbor crops of Districts with High Distress"
    *   "Find correlation path: Rain -> Yield -> Distress"

## Implementation Steps
1.  **Build Script (`dlpe_build_graph.py`)**:
    *   Load `district_feature_store_v3_temporal.csv`.
    *   Iterate rows and create Nodes/Edges.
    *   Save graph object (Pickle/GML).
2.  **Analysis**:
    *   Compute Centrality (Most critical districts).
    *   Find "Shock Clusters" (Districts with shared yield shocks).
