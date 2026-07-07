"""Prove the BG worker imports under the REAL lean `tfglm` conda-env package set.

Like the MoHa ``cluster/verify_lean_import.py`` but for the BG worker, which
ALSO imports ``visdetect.core.session`` (load_session for pkls). That means
``visdetect.core`` legitimately loads -- but only as the EMPTY stub; the heavy
``core.qc`` (PyYAML) / ``core.io`` / ``core.kilosort`` / ``core.spikeglx`` and
``analysis.hmm`` / ``analysis.align`` must NOT load. This catches a missing
``core/__init__`` stub offline, before burning a cluster job.

Simulates the lean env by blocking every top-level package not in the env's
site-packages, then imports the three leaves from the STAGED, STUBBED tree.

Exit 0 + "PASS" => the worker will import on the cluster.

Usage:
  py verify_lean_import_bg.py
  STAGED_SRC=... ALLOW_PKGS=numpy,pandas,scipy,sklearn,pyarrow py verify_lean_import_bg.py
"""
from __future__ import annotations
import os
import sys
import importlib.abc
from pathlib import Path

STAGED_SRC = Path(os.environ.get(
    "STAGED_SRC",
    "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/tf_glm_cluster/"
    "bg_mice/code/src"))
TFGLM_SITE_PKGS = Path(os.environ.get(
    "TFGLM_SITE_PKGS",
    "X:/public/projects/BeJG_20230130_VisDetect/wEPhys/conda_envs/tfglm/"
    "lib/python3.10/site-packages"))

STUBS = ["visdetect/__init__.py", "visdetect/analysis/__init__.py",
         "visdetect/core/__init__.py"]
HEAVY = ["visdetect.core.qc", "visdetect.core.io", "visdetect.core.kilosort",
         "visdetect.core.spikeglx", "visdetect.analysis.hmm",
         "visdetect.analysis.align"]


def env_top_levels(sp: Path) -> set:
    names = set()
    for p in sp.iterdir():
        n = p.name
        if n.endswith((".dist-info", ".egg-info", "__pycache__", ".pth")):
            continue
        if p.is_dir():
            names.add(n)
        elif n.endswith(".py"):
            names.add(n[:-3])
        elif n.endswith((".so", ".pyd")) or ".cpython-" in n:
            names.add(n.split(".")[0])
    return names


def main():
    if not (STAGED_SRC / "visdetect" / "core" / "session.py").is_file():
        print(f"FAIL: staged core/session.py not found under {STAGED_SRC}")
        return 2
    for s in STUBS:
        f = STAGED_SRC / s
        if not f.is_file():
            print(f"FAIL: missing stub {f}")
            return 2
        if f.stat().st_size != 0:
            print(f"FAIL: {s} is not the empty stub ({f.stat().st_size} bytes) "
                  f"-- staging did not stub it")
            return 1

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
    blocked_probe = [m for m in ("yaml", "h5py", "statsmodels", "ssm")
                     if m not in allow]
    print(f"env top-level packages: {len(allow)} allowed; stdlib: {len(std)}")
    print(f"will block (heavy-chain deps absent from env): {blocked_probe}")

    class LeanEnvBlocker(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            top = name.split(".")[0]
            if top in allow or top in std:
                return None
            raise ModuleNotFoundError(f"[SIMULATED-LEAN-ENV] not installed: {name}")

    sys.path.insert(0, str(STAGED_SRC))
    for m in [k for k in list(sys.modules)
              if k == "visdetect" or k.startswith("visdetect.")]:
        del sys.modules[m]
    sys.meta_path.insert(0, LeanEnvBlocker())

    import visdetect.core.session as cs  # noqa: F401
    import visdetect.analysis.tf_glm as tg
    import visdetect.analysis.tf_glm_data as td

    heavy = [h for h in HEAVY if h in sys.modules]
    if heavy:
        print(f"FAIL: heavy chain loaded despite stubs: {heavy}")
        return 1

    print(f"core.session: {cs.__file__}")
    print(f"tf_glm:       {tg.__file__}")
    print(f"tf_glm_data:  {td.__file__}")
    print("PASS: core.session + analysis leaves import under the simulated lean "
          "env; heavy core/analysis chain isolated. The worker will import on "
          "the cluster.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
