# tests/anatomy/test_plot_tf_cells_on_atlas.py
import matplotlib
matplotlib.use("Agg")
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "anatomy"))
from plot_tf_cells_on_atlas import load_tf_labels, plot_tf_cells, METRIC_INFO
from visdetect.anatomy.atlas import AllenAtlas
from visdetect.anatomy.tracks import TrackArtifact, ShankTrack


def _atlas():
    ann = np.ones((200, 160, 160), dtype=int)
    return AllenAtlas(annotation=ann, resolution_um=25.0, id_to_acronym={1: "CP"},
                      id_to_name={1: "Caudoputamen"}, id_to_coarse={1: "CP"})


def _art():
    mls = [1600.0, 1850.0, 2100.0, 2350.0]
    sh = [ShankTrack(i, np.array([[2500., mls[i], 2900.], [2500., mls[i], 2000.]]),
                     0.0, "brainreg_traced", 30., 30., 0.0) for i in range(4)]
    return TrackArtifact("BG_046", "allen_mouse_25um", "left", "forward", "t", "2026-06-24", sh)


def _setup(tmp_path, n=12):
    anat_dir = tmp_path / "BG_046"; anat_dir.mkdir()
    # session_name int with leading zero dropped (01072025 -> 1072025)
    pd.DataFrame({
        "session_name": [1072025] * n, "cluster_id": range(n),
        "ccf_ap": [2500.0] * n, "ccf_ml": np.linspace(1600, 2350, n),
        "ccf_dv": np.linspace(2600, 3200, n), "region_coarse": ["CP"] * n,
    }).to_csv(anat_dir / "unit_anatomy.csv", index=False)
    # TF registry: first 6 units responsive
    tf_dir = tmp_path / "tf"; tf_dir.mkdir()
    pd.DataFrame({
        "session": ["BG_046_01072025"] * n, "session_date": ["01072025"] * n,
        "unit": range(n), "resp_log2": [True] * 6 + [False] * (n - 6),
    }).to_csv(tf_dir / "bg046_tf_responsive.csv", index=False)
    # kernel width for the responsive ones (a spread of FWHM)
    kw = tmp_path / "kw.csv"
    pd.DataFrame({
        "subject": ["BG_046"] * 6, "session": ["BG_046_01072025"] * 6,
        "unit": range(6), "interp_fwhm": [0.03, 0.06, 0.09, 0.12, 0.2, 0.4],
    }).to_csv(kw, index=False)
    return anat_dir, tf_dir, kw


def test_load_tf_labels_joins(tmp_path):
    anat_dir, tf_dir, kw = _setup(tmp_path)
    df = load_tf_labels("BG_046", anatomy_dir=str(anat_dir), tf_dir=str(tf_dir), kw_csv=str(kw))
    assert (df["resp_log2"] == True).sum() == 6          # noqa: E712
    assert df["interp_fwhm"].notna().sum() == 6
    # leading-zero session id joined correctly (no silent miss)
    assert df.loc[df.cluster_id == 0, "resp_log2"].iloc[0] == True   # noqa: E712


def test_all_metrics_render(tmp_path):
    anat_dir, tf_dir, kw = _setup(tmp_path)
    df = load_tf_labels("BG_046", anatomy_dir=str(anat_dir), tf_dir=str(tf_dir), kw_csv=str(kw))
    for m in METRIC_INFO:
        for coords in ("ccf", "stereotaxic"):
            out = tmp_path / f"{m}_{coords}.png"
            assert os.path.exists(plot_tf_cells("BG_046", df, _art(), m, out,
                                                atlas=_atlas(), coords=coords))


def test_no_tf_returns_none(tmp_path):
    # a subject with no registry / no responsive cells -> None, no crash
    anat_dir = tmp_path / "BG_038"; anat_dir.mkdir()
    pd.DataFrame({"session_name": [1072025] * 4, "cluster_id": range(4),
                  "ccf_ap": [2500.0] * 4, "ccf_ml": [1600.0, 1850, 2100, 2350],
                  "ccf_dv": [2600.0] * 4, "region_coarse": ["CTX"] * 4}).to_csv(
        anat_dir / "unit_anatomy.csv", index=False)
    df = load_tf_labels("BG_038", anatomy_dir=str(anat_dir),
                        tf_dir=str(tmp_path / "none"), kw_csv=str(tmp_path / "none.csv"))
    assert plot_tf_cells("BG_038", df, _art(), "kernel_width", tmp_path / "x.png",
                         atlas=_atlas()) is None
