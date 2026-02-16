# Refactoring Plan

## Goal
Move flat scripts into a `src` package structure.

## Structure
```
src/
    __init__.py
    ingest/
        __init__.py
        mnrega.py (was dlpe_ingest_mnrega.py)
        agri.py (was dlpe_ingest_no_api.py)
        temporal.py (was dlpe_temporal_features.py)
    model/
        __init__.py
        distress.py (was train_distress_model.py)
        early_warning.py (was train_early_warning.py)
        graph_builder.py (was dlpe_build_graph.py)
    utils/
        __init__.py
        validation.py (was validate_*.py)
```

## Action Items
1.  Move files.
2.  Update imports in `app.py`.
3.  Verify `app.py` runs.
