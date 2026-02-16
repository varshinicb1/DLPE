# Contributing to DLPE / KissanGPT 🚜

Thank you for your interest in helping Indian farmers! We follow a structured AGI-first development process.

## How to Contribute

### 1. Per-City Data Ingestion
The most impactful way to contribute is by adding real-world datasets for your district.
- Create a CSV following the `kissan/data/city_template_example.csv` format.
- Submit a Pull Request. The `kissan/orchestrator.py` will automatically integrate your data into the next model version.

### 2. Code Improvements
- **Coding Style**: We use PEP 8 for Python. Please run `black` or `autopep8` before submitting.
- **Testing**: Add unit tests in `tests/` for any new modules.
- **Branching**: Use feature branches (`feat/feature-name`) and submit PRs to `main`.

## Developer Setup
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`.
3. Setup pre-commit: `pre-commit install`.

## Community & Ethics
We build for offline-first, village-level accessibility. Ensure your contributions prioritize light-weight execution and Kannada language compatibility.
