import pathlib
import sys

# Make the tracking_dant package importable by bare module name in tests.
_PKG = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "tracking_dant"
if str(_PKG) not in sys.path:
    sys.path.insert(0, str(_PKG))
