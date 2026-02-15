# generic_ckan_harvester_plan

## Goal
Build a generic CKAN API harvester to ingest agricultural data (rainfall, production, crop yields) from `data.gov.in` (specifically Karnataka subdomain) and create a district-level feature store for modeling.

## User Review Required
> [!IMPORTANT]
> **API Access**: The notebook will run in Google Colab. Ensure the Colab runtime has access to `karnataka.data.gov.in`. If the state portal is flaky, the fallback is the central `data.gov.in` with state filters.
> **Scope**: focusing on **Option A (Karnataka Pilot)** as requested to ensure schema normalization is perfect before scaling.

## Proposed Changes

### [New] [ckan_harvester.ipynb](file:///C:/Users/varsh/.gemini/antigravity/brain/6db64765-7966-408d-a075-37c115d6b037/ckan_harvester.ipynb)
A Jupyter notebook containing:
1.  **Configuration**:
    - `API_KEY` (User input)
    - `BASE_URL = "https://karnataka.data.gov.in"` (Configurable)
2.  **CKANHarvester Class**:
    - `__init__(base_url, api_key)`
    - `package_search(query, params)`: Wraps `/api/3/action/package_search`
    - `get_resource_data(resource_id)`: Wraps `/api/3/action/datastore_search` (with pagination/limit handling)
3.  **Data Processing Pipeline**:
    - **Step 1: Search**: Query for "district", "crop", "rainfall", "production".
    - **Step 2: Filter**: Select relevant packages (avoiding unrelated search results).
    - **Step 3: Ingest**: Loop through resources, fetch data.
    - **Step 4: Normalize**:
        - Standardize column names (regex for 'dist.*' -> 'district', 'prod.*' -> 'production').
        - Handle district name mismatched (fuzzy matching if needed, or simple cleaning).
        - Convert numeric columns.
4.  **Feature Store Construction**:
    - Merge variable datasets into a single DataFrame on `(District, Year)`.
    - Handle missing values (imputation or drop).
5.  **Modeling**:
    - Training a baseline XGBoost regressor (Target: Yield/Production).

## Verification Plan
### Automated Tests within Notebook
- **API Check**: A cell to verify connection to `package_search`.
- **Schema Check**: A cell asserting that the normalized DataFrame has strictly `[District, Year, Feature, Value]` or wide-format equivalent.
- **Model Check**: Minimal train/test run to ensure pipeline end-to-end success.

### Manual Verification
- User runs the notebook in Colab.
- User inspects `district_feature_store.csv` output.
