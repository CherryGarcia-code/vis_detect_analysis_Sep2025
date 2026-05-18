"""visdetect.suite — analysis-suite infrastructure.

Shared infrastructure for the figure/analysis pipeline: configuration and
output paths (:mod:`config`), unified data loading (:mod:`loader`), and
publication plotting helpers (:mod:`plotting`).

This is project glue, not core library logic — it wires the figure scripts
to the :mod:`visdetect` library. Scripts import from here, e.g.::

    from visdetect.suite.config import STAGE_ORDER
    from visdetect.suite.loader import load_session
    from visdetect.suite.plotting import save_figure
"""
