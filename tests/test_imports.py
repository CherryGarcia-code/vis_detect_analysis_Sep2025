def test_basic_imports():
    import importlib
    import sys
    from pathlib import Path

    # Ensure repository root is on sys.path so `src` is importable when pytest runs
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    importlib.import_module("visdetect.analysis.optotagging")
    importlib.import_module("visdetect.analysis.tracking")
    importlib.import_module("visdetect.analysis.responsive_analysis")
    # ensure the runner script is syntactically valid
    importlib.import_module("scripts.run_demo_pipeline")
    importlib.import_module("scripts.run_batch")
    importlib.import_module("visdetect.core.kilosort")

    importlib.import_module("visdetect.integrations.bombcell_wrapper")
    importlib.import_module("scripts.run_bombcell_wrapper")
