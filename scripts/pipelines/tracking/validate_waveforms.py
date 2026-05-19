#!/usr/bin/env python3
"""Validate that UnitMatch input RawWaveforms are RAW-AVERAGED, not KS templates.

UnitMatch's similarity metrics are calibrated for raw averaged spike waveforms.
If the input were Kilosort *templates* (low-rank, noiseless reconstructions),
matching would be compromised. This script decides which they are, from the
physical signatures in the data itself:

  signature              raw averaged spikes        KS template
  ---------------------  -------------------------  -----------------------
  pre-spike baseline     real noise floor (>0.5%    ~flat zero (<~0.1% of
                         of peak amplitude)         peak)
  CV half-1 vs half-2    differ by white-ish noise  near-identical / no
                         present in the baseline    natural CV split
  off-footprint channels noise floor everywhere     many channels exactly 0

Usage:
    py scripts/pipelines/tracking/validate_waveforms.py
    py scripts/pipelines/tracking/validate_waveforms.py --input <dir> --no-fig
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "data" / "unit_match" / "input" / "BG_046"
OUT_DIR = REPO_ROOT / "FIGURES" / "tracking_qc"

BASELINE = slice(2, 25)      # pre-spike samples (spike width 82, peak ~40)
N_SESSIONS = 6               # sample this many sessions across the span
N_UNITS_PER_SESSION = 8


def analyse_unit(wave: np.ndarray):
    """wave: (n_t, n_chan, 2). Return per-unit diagnostics."""
    h1, h2 = wave[:, :, 0], wave[:, :, 1]
    avg = wave.mean(axis=2)
    ptp = avg.max(axis=0) - avg.min(axis=0)
    pk = int(np.argmax(ptp))
    peak_amp = float(ptp[pk])
    if peak_amp < 1e-6:
        return None

    # 1. baseline noise floor on the peak channel (% of peak amplitude)
    base_rms = float(np.std(avg[BASELINE, pk]))
    base_frac = base_rms / peak_amp

    # 2. CV half-1 vs half-2 difference in the baseline (pure sampling noise
    #    if raw; ~0 if template)
    cv_diff_base = float(np.std((h1 - h2)[BASELINE, pk]))
    cv_diff_frac = cv_diff_base / peak_amp

    # 3. off-footprint channels: fraction of channels that are *exactly* zero
    exact_zero = float(np.mean(np.all(avg == 0.0, axis=0)))
    # near-zero baseline RMS across ALL channels (template -> ~0 everywhere
    # off-footprint; raw -> consistent noise floor everywhere)
    per_chan_base = np.std(avg[BASELINE, :], axis=0)
    med_base_all = float(np.median(per_chan_base))

    return dict(peak_amp=peak_amp, base_frac=base_frac,
                cv_diff_frac=cv_diff_frac, exact_zero_frac=exact_zero,
                med_baseline_rms=med_base_all, peak_chan=pk)


def main():
    ap = argparse.ArgumentParser(description="Raw-vs-template waveform validator")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--no-fig", action="store_true")
    args = ap.parse_args()

    sess = sorted([d for d in args.input.iterdir() if d.is_dir()])
    pick = sess[:: max(1, len(sess) // N_SESSIONS)][:N_SESSIONS]
    print(f"Sampling {len(pick)} sessions: {[d.name for d in pick]}\n")

    rows, examples = [], []
    for d in pick:
        wf = sorted((d / "RawWaveforms").glob("*.npy"))[:N_UNITS_PER_SESSION]
        for f in wf:
            w = np.load(f)
            if w.ndim != 3 or w.shape[2] != 2:
                continue
            r = analyse_unit(w)
            if r is None:
                continue
            r["session"] = d.name
            rows.append(r)
            if len(examples) < 4:
                examples.append((f"{d.name}/{f.stem}", w, r["peak_chan"]))

    base_frac = np.array([r["base_frac"] for r in rows])
    cv_frac = np.array([r["cv_diff_frac"] for r in rows])
    exact_zero = np.array([r["exact_zero_frac"] for r in rows])

    print("=" * 64)
    print(f"WAVEFORM VALIDATION  ({len(rows)} units)")
    print("=" * 64)
    print(f"  baseline noise / peak amplitude : median {np.median(base_frac)*100:.2f}%"
          f"  (range {base_frac.min()*100:.2f}-{base_frac.max()*100:.2f}%)")
    print(f"  CV half-diff / peak amplitude   : median {np.median(cv_frac)*100:.2f}%")
    print(f"  channels exactly zero per unit  : median {np.median(exact_zero)*100:.1f}%")
    print()

    # verdict
    raw_like = (np.median(base_frac) > 0.005 and np.median(cv_frac) > 0.003)
    template_like = (np.median(base_frac) < 0.001 and np.median(exact_zero) > 0.3)
    print("  VERDICT:")
    if raw_like and not template_like:
        print("  -> RAW-AVERAGED spike waveforms. A real baseline noise floor")
        print("     and noise-like CV half differences are present -- this is")
        print("     averaged raw data, exactly what UnitMatch expects.")
    elif template_like:
        print("  -> KS TEMPLATE-like. Baseline is ~flat and many channels are")
        print("     exactly zero. NOT suitable -- re-extract with raw averaging.")
    else:
        print("  -> AMBIGUOUS. Inspect the figure manually.")
    print(f"     (raw_like={raw_like}, template_like={template_like})")

    if not args.no_fig:
        _figure(examples)
    print(f"\n  (n exactly-zero-channel units: "
          f"{int(np.sum(exact_zero > 0.5))}/{len(rows)} have >50% zero channels)")


def _figure(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(examples), 2, figsize=(13, 3 * len(examples)))
    if len(examples) == 1:
        axes = axes[None, :]
    for i, (name, w, pk) in enumerate(examples):
        h1, h2 = w[:, pk, 0], w[:, pk, 1]
        # full peak-channel trace, both CV halves
        axes[i, 0].plot(h1, lw=1, label="CV half 1")
        axes[i, 0].plot(h2, lw=1, label="CV half 2")
        axes[i, 0].axvspan(BASELINE.start, BASELINE.stop, color="gray", alpha=0.2)
        axes[i, 0].set_title(f"{name} peak ch{pk} - full")
        axes[i, 0].legend(fontsize=7)
        # baseline zoom -- raw shows noise here, templates show a flat line
        axes[i, 1].plot(range(BASELINE.start, BASELINE.stop),
                        h1[BASELINE], "o-", ms=3, label="CV1")
        axes[i, 1].plot(range(BASELINE.start, BASELINE.stop),
                        h2[BASELINE], "o-", ms=3, label="CV2")
        axes[i, 1].set_title(f"{name} - baseline zoom (noise floor?)")
        axes[i, 1].legend(fontsize=7)
    fig.tight_layout()
    out = OUT_DIR / "waveform_validation.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  figure -> {out}")


if __name__ == "__main__":
    main()
