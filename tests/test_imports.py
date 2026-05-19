def test_basic_imports():
    import importlib

    importlib.import_module("visdetect.analysis.optotagging")
    # Note: visdetect.analysis.tracking and responsive_analysis were removed/archived.
    # Note: scripts.pipelines.run_demo_pipeline and run_bombcell_wrapper moved to
    #       archive/pipelines_archive/ and are no longer importable from the default path.
    importlib.import_module("visdetect.core.kilosort")
    importlib.import_module("visdetect.core.session")
    importlib.import_module("visdetect.analysis.behavior")