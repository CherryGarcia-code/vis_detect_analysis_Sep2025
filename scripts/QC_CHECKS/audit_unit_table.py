"""Audit the per-unit label table spine: which inputs exist, and what build_unit_table produces.

Usage:
    py scripts/QC_CHECKS/audit_unit_table.py
    py scripts/QC_CHECKS/audit_unit_table.py --out FIGURES/qc/unit_table_audit.txt
"""
import argparse
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

# suite.config re-exports the analysis.config paths AND defines CACHE_DIR/FIGURE_DIR,
# so import all four from suite.config (CACHE_DIR is NOT in analysis.config).
from visdetect.suite.config import (  # noqa: E402
    GLT_PATH, WAVEFORM_LABELS_PATH, LICK_DIR, CACHE_DIR,
)
from visdetect.suite.unit_table_schema import (  # noqa: E402
    CONTRACT_COLUMNS, KEY_COLUMNS,
)


def _exists(label, path):
    ok = os.path.exists(path)
    return f"  [{'OK ' if ok else 'MISSING'}] {label}: {path}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="Optional path to write the report")
    args = ap.parse_args()

    lines = ["=== UNIT-LABEL-TABLE AUDIT ===", "", "Inputs:"]
    lines.append(_exists("GLT", GLT_PATH))
    lines.append(_exists("waveform labels", WAVEFORM_LABELS_PATH))
    lines.append(_exists("lick dir", LICK_DIR))
    lines.append(_exists("detrended TF resp", os.path.join(CACHE_DIR, "tf_responsiveness_detrended.csv")))
    lines.append(_exists("detrended tier", os.path.join(CACHE_DIR, "tf_cell_classification_detrended.csv")))

    lines += ["", "build_unit_table():"]
    try:
        from visdetect.suite.loader import build_unit_table
        df = build_unit_table(qc_only=True)
        lines.append(f"  ROWS: {len(df)}")
        dup = int(df.duplicated(subset=[c for c in KEY_COLUMNS if c in df.columns]).sum())
        lines.append(f"  DUPLICATE KEYS: {dup}")
        lines.append(f"  COLUMNS ({len(df.columns)}): {list(df.columns)}")
        lines.append("  CONTRACT COVERAGE:")
        for c in CONTRACT_COLUMNS:
            if c in df.columns:
                nan_frac = float(df[c].isna().mean())
                lines.append(f"    {c}: present, NaN-fraction={nan_frac:.3f}")
            else:
                lines.append(f"    {c}: MISSING")
    except Exception as exc:  # noqa: BLE001 — audit must report, not crash
        lines.append(f"  FAILED: {type(exc).__name__}: {exc}")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
