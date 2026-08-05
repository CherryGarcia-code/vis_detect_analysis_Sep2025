"""Guard the interactive-backend contract of the tagger entry points.

Regression test for the A1-pilot bug: `tag_session.py` selected TkAgg BEFORE
importing visdetect, but several library modules (`visdetect.core.qc`,
`visdetect.suite.plotting`, `visdetect.analysis.tf_pulse`, ...) call
`matplotlib.use("Agg")` at import time and silently clobbered it. The window
then refused to show ("FigureCanvasAgg is non-interactive"). Every headless
check passed because they all set MPLBACKEND=Agg deliberately, so nothing
caught it until a human ran the GUI.

The invariant: with MPLBACKEND unset, importing the tool must leave an
interactive backend live; with MPLBACKEND=Agg it must stay headless-safe.
Backend selection is process-global, so each case runs in a subprocess.
"""
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")

TOOLS = [
    os.path.join("scripts", "video", "tag_session.py"),
    os.path.join("scripts", "video", "click_anchor.py"),
]

# Only tag_session promises to honour an explicit MPLBACKEND=Agg. Legacy
# click_anchor calls matplotlib.use("TkAgg", force=True) unconditionally — it
# predates this contract and is deliberately left alone (changing a working
# tool's backend handling is out of scope), so it is excluded from the headless
# case but still covered by the interactive-ordering case above.
HEADLESS_AWARE_TOOLS = [os.path.join("scripts", "video", "tag_session.py")]

_PROBE = (
    "import importlib.util as u, matplotlib\n"
    "s = u.spec_from_file_location('probe', r'{path}')\n"
    "m = u.module_from_spec(s)\n"
    "s.loader.exec_module(m)\n"
    "print(matplotlib.get_backend())\n"
)


def _import_and_report_backend(rel_path, mplbackend=None):
    """Import the tool in a clean subprocess; return the live backend name."""
    env = dict(os.environ)
    env["PYTHONPATH"] = SRC
    env.pop("MPLBACKEND", None)
    if mplbackend is not None:
        env["MPLBACKEND"] = mplbackend
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE.format(path=os.path.join(REPO_ROOT, rel_path))],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, (
        f"importing {rel_path} failed (MPLBACKEND={mplbackend!r}):\n{proc.stderr}"
    )
    return proc.stdout.strip().splitlines()[-1].strip()


@pytest.mark.parametrize("rel_path", TOOLS)
def test_tool_keeps_interactive_backend_after_visdetect_imports(rel_path):
    """With MPLBACKEND unset the GUI tools must end up on an interactive backend.

    Guards the import-ORDER invariant: the tool's matplotlib.use("TkAgg") must
    come after the visdetect imports that call matplotlib.use("Agg").
    """
    pytest.importorskip("tkinter", reason="TkAgg needs tkinter")
    backend = _import_and_report_backend(rel_path, mplbackend=None)
    assert backend.lower() != "agg", (
        f"{rel_path} left backend={backend!r}: a visdetect import clobbered the "
        "interactive backend. Move the matplotlib.use('TkAgg', force=True) block "
        "AFTER the visdetect imports."
    )


@pytest.mark.parametrize("rel_path", HEADLESS_AWARE_TOOLS)
def test_tool_honours_explicit_headless_request(rel_path):
    """MPLBACKEND=Agg must be honoured so --help / spec-import need no display."""
    backend = _import_and_report_backend(rel_path, mplbackend="Agg")
    assert backend.lower() == "agg", (
        f"{rel_path} overrode an explicit MPLBACKEND=Agg (got {backend!r}); "
        "headless verification would then require a display."
    )
