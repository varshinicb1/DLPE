# AgiEngine: The Autonomous AGI Framework 🏛️

I have transformed the KissanGPT project into **AgiEngine**, a generic, multi-profession AGI framework that evolves autonomously using real-world data.

## 🚀 Architectural Evolution
1.  **Domain Agnostic Core**: The implementation has been migrated from `kissan/` to `engine/`. Key classes are now `AgiCoreModel` and `AgiBrain`, allowing the AI to be repurposed for any profession.
2.  **Domain Personalities**: Configurable through `domains/*.json`. Currently supports:
    - **Agriculture**: Professional-grade market and soil advice in Kannada.
    - **Animal Husbandry**: Veterinary health diagnostics and stock management.
3.  **The "Data Hungry" Loop**:
    - **Scraper Agent**: `engine/scraper_agent.py` autonomously hunts the web for CSV/JSON datasets.
    - **Orchestrator**: Detects new data in `engine/data/ingest/`, triggers retraining, and refreshes the knowledge base.

## 🕵️‍♂️ Autonomous Ingestion Verified
I verified the scraper by running an initial hunt cycle. It successfully captured:
- **WFP India Food Prices** (Historical & Recent)
- **Crop Recommendation Mappings** (NPK-based science)
- **Cattle Health Snapshots** (For Animal Husbandry)

The system is now capable of **evolving 24/7** as new production data is discovered.

## 🛠️ Developer Usage
- **Add a Profession**: Create a new JSON in `domains/` and drop relevant data into `engine/data/ingest/<profession>/`.
- **Run the Engine**: 
  ```bash
  python engine/brain.py --domain agriculture --query "Price of rice in Belgaum?"
  python engine/brain.py --domain animal_husbandry --query "Fever in Gir cattle?"
  ```

This completes the project goal of building a perfect, data-hungry AGI for the real world!
