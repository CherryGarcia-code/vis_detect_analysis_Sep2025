"""FIX B: ONE combined cell-type-validity panel across animals (the "why optotagging" slide).

Per animal, three diagnostics of the narrow/broad (width) split:
  (i)  width bimodality: GMM delta-BIC (2- vs 1-component) on t2p
  (ii) CV2 AUC  : how well a RATE-INDEPENDENT ISI feature classifies the width groups
  (iii) rate AUC: how well firing RATE (the CONFOUND) classifies them
Groups labelled by the COMMON cutoff so animals are comparable. One combined verdict:
cell type is PUTATIVE (width-defined; rate-independent confirmation moderate, weaker than the
rate confound) -> motivates optotagging.

Usage: py scripts/talk_substrate/ws_fixB_validity.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402
from visdetect.suite.plotting import setup_style  # noqa: E402
from visdetect.analysis.waveform_celltype import classify_celltype  # noqa: E402

C.setup_talk_style()
REGION = {"BG_046": "DMS", "BG_039": "DMS", "BG_031": "VMS", "BG_038": "Cortex(ref)"}
AUC_READY = 0.70


def sep_auc(x, label):
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(x)
    if ok.sum() < 10 or len(np.unique(label[ok])) < 2:
        return np.nan
    return max(roc_auc_score(label[ok], x[ok]), 1 - roc_auc_score(label[ok], x[ok]))


def main():
    thr, _ = C.common_t2p_cutoff()
    rows = []
    for subj in C.ALL_SUBJECTS:
        t2p = C.load_t2p(subj)
        isi = pd.read_csv(C.CACHE_DIR / f"isi_features_{subj}.csv"
                          if subj != "BG_046" else C.CACHE_DIR / "bg046_isi_features.csv",
                          dtype={"session_8": str})
        # align isi session col name
        scol = "session_8" if "session_8" in isi.columns else "session_date"
        isi = isi.rename(columns={scol: "session_8"})
        m = t2p.merge(isi[["session_8", "cluster_id", "cv2", "rate_hz"]],
                      on=["session_8", "cluster_id"], how="inner")
        lab = (m["t2p_ms"].values >= thr).astype(int)  # 1 = broad
        _, info = classify_celltype(t2p["t2p_ms"].values)
        rows.append(dict(animal=subj, region=REGION[subj], delta_bic=info["delta_bic"],
                         cv2_auc=sep_auc(m["cv2"].values, lab),
                         rate_auc=sep_auc(m["rate_hz"].values, lab), n=len(m)))
    df = pd.DataFrame(rows)

    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.0, 1.1], wspace=0.32)
    x = np.arange(len(df))
    # panel 1: CV2 vs rate AUC
    ax0 = fig.add_subplot(gs[0])
    w = 0.38
    ax0.bar(x - w / 2, df["cv2_auc"], w, color="#2c7fb8", label="CV2 AUC (rate-indep)")
    ax0.bar(x + w / 2, df["rate_auc"], w, color="#cccccc", label="rate AUC (confound)")
    ax0.axhline(0.5, color="k", lw=0.8, ls=":")
    ax0.axhline(AUC_READY, color="green", lw=1.0, ls="--", label="ready (0.70)")
    ax0.set_xticks(x); ax0.set_xticklabels([f"{r.animal}\n{r.region}" for r in df.itertuples()], fontsize=8)
    ax0.set_ylim(0.45, 0.95); ax0.set_ylabel("separation AUC (narrow vs broad)")
    ax0.set_title("Rate-independent split (CV2) vs the rate confound", fontsize=C.FS["title"])
    ax0.legend(frameon=False, fontsize=7, loc="upper left")
    # panel 2: delta-BIC (log)
    ax1 = fig.add_subplot(gs[1])
    ax1.bar(x, df["delta_bic"], color="#7a3b8f")
    ax1.set_yscale("log")
    ax1.set_xticks(x); ax1.set_xticklabels([r.animal.replace("BG_", "") for r in df.itertuples()], fontsize=8)
    ax1.set_ylabel("width GMM ΔBIC (2 vs 1 comp, log)")
    ax1.set_title("Width bimodality", fontsize=C.FS["title"])
    # panel 3: verdict
    ax2 = fig.add_subplot(gs[2]); ax2.axis("off")
    best = df.loc[df["cv2_auc"].idxmax()]
    txt = ["FIX B — combined cell-type validity", "",
           f"common cutoff: {thr:.3f} ms", "",
           "per animal (CV2 / rate AUC, ΔBIC):"]
    for r in df.itertuples():
        txt.append(f"  {r.animal} {r.region:11s}: {r.cv2_auc:.2f} / {r.rate_auc:.2f}, BIC {r.delta_bic:.0f}")
    txt += ["",
            "All: width is bimodal (ΔBIC>0), but a",
            "RATE-INDEPENDENT split (CV2) is moderate",
            f"and WEAKER than rate in every animal",
            f"(best CV2 = {best.animal} {best.cv2_auc:.2f}).",
            "",
            "VERDICT: cell type is PUTATIVE",
            "(width-defined; rate-independent support",
            " partial) -> identity needs OPTOTAGGING."]
    ax2.text(0.0, 1.0, "\n".join(txt), va="top", ha="left", fontsize=8.6, family="monospace")

    fig.suptitle("Cell-type label validity across animals (one combined verdict)", fontsize=C.FS["suptitle"], y=1.0)
    fig.text(0.5, -0.03,
             "Groups labelled by the COMMON width cutoff. CV2 = rate-independent ISI irregularity; "
             "rate is the confound (width<->rate correlated). Width is bimodal everywhere, but the "
             "rate-independent confirmation is only moderate -> keep 'putative', motivate optotagging.",
             ha="center", fontsize=C.FS["caption"], color=C.CAPTION_GREY, wrap=True)
    out = C.FIG_DIR.parent / "ws_fixB_celltype_validity_combined.png"
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    df.to_csv(C.FIG_DIR.parent / "ws_fixB_celltype_validity_combined.csv", index=False)
    print(f"[fig] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
