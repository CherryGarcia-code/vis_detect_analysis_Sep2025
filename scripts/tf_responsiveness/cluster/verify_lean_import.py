"""Prove the cluster worker imports under the REAL `tfglm` conda-env package set.

The first cluster run died because importing ``visdetect.analysis.tf_glm`` fired
the package ``__init__`` chain (``core/qc`` -> PyYAML, ``analysis/hmm`` -> ssm,
``align`` -> h5py), none of which the lean ``tfglm`` env has. The staging step
now empties the two heavy ``__init__.py`` files; this test PROVES that fix is
sufficient, independent of the cluster, by simulating the lean env:

  1. Derive the set of top-level packages ACTUALLY installed in the env's
     site-packages (live, from ceph).
  2. Install an import hook that allows only those (+ stdlib + visdetect) and
     raises ModuleNotFoundError for anything else -- exactly what the cluster
     would do.
  3. Import the STAGED, STUBBED ``visdetect.analysis.tf_glm`` + ``.tf_glm_data``
     and assert the heavy chain (core.qc / analysis.hmm / align / core) never
     loaded.

Exit 0 + "PASS" => the worker will import on the cluster. Non-zero => it won't.

Usage:
  py verify_lean_import.py
  STAGED_SRC=... TFGLM_SITE_PKGS=... py verify_lean_import.py
"""
from __future__ import annotations
import os
import sys
import importlib.abc
from pathlib import Path

STAGED_SRC = Path(os.environ.get(
    "STAGED_SRC",
    "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/code/src"))
TFGLM_SITE_PKGS = Path(os.environ.get(
    "TFGLM_SITE_PKGS",
    "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/conda_envs/tfglm/"
    "lib/python3.10/site-packages"))


def env_top_levels(sp: Path) -> set:
    """Top-level importable names actually present in a site-packages dir."""
    names = set()
    for p in sp.iterdir():
        n = p.name
        if (n.endswith((".dist-info", ".egg-info", "__pycache__", ".pth"))):
            continue
        if p.is_dir():
            names.add(n)
        elif n.endswith(".py"):
            names.add(n[:-3])
        elif n.endswith((".so", ".pyd")) or ".cpython-" in n:
            names.add(n.split(".")[0])
    return names


def main():
    if not (STAGED_SRC / "visdetect" / "analysis" / "tf_glm.py").is_file():
        print(f"FAIL: staged tf_glm.py not found under {STAGED_SRC}")
        return 2

    # The allowlist of env top-level packages: either passed in via ALLOW_PKGS
    # (comma-separated; lets the test run fully offline / off the slow ceph
    # mount) or derived live from the env's site-packages dir.
    allow_env = os.environ.get("ALLOW_PKGS")
    if allow_env:
        env_pkgs = {p.strip() for p in allow_env.split(",") if p.strip()}
    else:
        if not TFGLM_SITE_PKGS.is_dir():
            print(f"FAIL: env site-packages not found: {TFGLM_SITE_PKGS} "
                  f"(or pass ALLOW_PKGS=...)")
            return 2
        env_pkgs = env_top_levels(TFGLM_SITE_PKGS)
    allow = env_pkgs | {"visdetect", "pkg_resources", "_distutils_hack"}
    std = set(getattr(sys, "stdlib_module_names", set()))
    print(f"env top-level packages: {len(allow)} allowed; stdlib: {len(std)}")
    blocked_probe = [m for m in ("yaml", "h5py", "statsmodels", "ssm")
                     if m not in allow]
    print(f"will block (heavy-chain deps absent from env): {blocked_probe}")

    class LeanEnvBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            top = name.split(".")[0]
            if top in allow or top in std:
                return None  # allow normal resolution
            raise ModuleNotFoundError(f"[SIMULATED-LEAN-ENV] not installed: {name}")

    # Staged stubbed src must win; clear any pre-imported visdetect first.
    sys.path.insert(0, str(STAGED_SRC))
    for m in [k for k in list(sys.modules)
              if k == "visdetect" or k.startswith("visdetect.")]:
        del sys.modules[m]
    sys.meta_path.insert(0, LeanEnvBlocker())

    import visdetect
    import visdetect.analysis.tf_glm as tg
    import visdetect.analysis.tf_glm_data as td

    vfile = visdetect.__file__ or ""
    try:
        under = Path(vfile).resolve().is_relative_to(STAGED_SRC.resolve())
    except Exception:
        under = str(STAGED_SRC.resolve()) in str(Path(vfile).resolve())
    if not under:
        print(f"FAIL: imported the wrong visdetect: {vfile} (want under {STAGED_SRC})")
        return 1
    if Path(vfile).stat().st_size != 0:
        print(f"FAIL: visdetect/__init__.py is not the empty stub: {vfile}")
        return 1
    heavy = [h for h in ("visdetect.core", "visdetect.core.qc",
                         "visdetect.analysis.hmm", "visdetect.analysis.align")
             if h in sys.modules]
    if heavy:
        print(f"FAIL: heavy __init__ chain loaded: {heavy}")
        return 1

    print(f"visdetect:    {vfile}")
    print(f"tf_glm:       {tg.__file__}")
    print(f"tf_glm_data:  {td.__file__}")
    print("PASS: leaf modules import under the simulated lean env; "
          "heavy __init__ chain isolated. The worker will import on the cluster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
