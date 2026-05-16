def test_basic_imports():
    import importlib
    import sys
    from pathlib import Path

    # Ensure repository root is on sys.path so `src` is importable when pytest runs
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))

    importlib.import_module("visdetect.analysis.optotagging")
    # Note: visdetect.analysis.tracking and responsive_analysis were removed/archived.
    # Note: scripts.pipelines.run_demo_pipeline and run_bombcell_wrapper moved to
    #       archive/pipelines_archive/ and are no longer importable from the default path.
    importlib.import_module("visdetect.core.kilosort")
    importlib.import_module("visdetect.core.session")
    importlib.import_module("visdetect.analysis.behavior")