import os, sys
TRACK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRACK)
from eval_deepum_checkpoints import summarize_run, BASELINE_ROWS


def test_summarize_run_row():
    summary = {"n_tracked_ids": 1000, "ge_2": 80, "ge_5": 10, "ge_10": 2,
               "ge_15": 0, "ge_20": 0, "max_span": 12}
    row = summarize_run("warmstart_ep10", summary)
    assert row["label"] == "warmstart_ep10"
    assert abs(row["ge_2_pct"] - 8.0) < 1e-9      # 80/1000
    assert row["max_span"] == 12
    print("test_summarize_run_row PASS")


def test_baseline_rows_present():
    labels = {r["label"] for r in BASELINE_ROWS}
    assert {"UM 3.2.9", "DeepUM stock"} <= labels
    print("test_baseline_rows_present PASS")


if __name__ == "__main__":
    test_summarize_run_row()
    test_baseline_rows_present()
    print("ALL EVAL-ASSEMBLY TESTS PASSED")
