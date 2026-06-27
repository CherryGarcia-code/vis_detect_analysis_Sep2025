"""Re-derive the cluster recovery verdict with the PER-DIAL shrunk veto.

Why this exists: the Jun-26 cluster run (job 3202468, harness <= commit 6de25a3)
computed the shrunk veto as a SCALAR ``any(|rec[d]| < 0.5*|true[d]|)`` in
``recover_true_difference``; ``recovery_gate`` then applied that scalar to EVERY dial
(gate code path for a non-dict ``shrunk``). Because only sharpness (v) was crushed by
the L2 prior (recovered 0.27 vs true 0.95) while caution (z) recovered faithfully
(-0.83 vs -0.85), the scalar wrongly vetoed caution+timing too -> an "all descriptive"
ARTIFACT. gate_criteria.md line 11 ratifies a PER-DIAL veto ("shrunk -> *that dial*
'descriptive'"), and the library fix makes ``recover_true_difference`` return a
per-dial ``{dial: bool}`` mapping. The expensive point-recovery + confusion are
already in the cluster JSON, so we DON'T re-run the 4.6h job: we recompute the
per-dial shrunk from the cached ``recovered_delta``/``true_delta`` and re-apply
``recovery_gate``.

It preserves the raw cluster output as ``*.cluster_raw.json`` and stamps provenance
into ``meta``. Idempotent: re-running on an already-corrected file reproduces it.

Run:  PYTHONPATH=<repo>/src py scripts/analysis/decision_latents/_regate_perdial_shrunk.py \
          [data/cache/decision_latents/recovery_results.json]
"""
import json
import os
import shutil
import sys

from visdetect.analysis.decision_latents_generative import recovery_gate

DEFAULT_JSON = os.path.join("data", "cache", "decision_latents", "recovery_results.json")


def per_dial_shrunk(recovered_delta, true_delta):
    """The fixed criterion: shrunk[d] = |recovered[d]| < 0.5*|true[d]| (per dial)."""
    return {d: bool(abs(float(recovered_delta[d])) < 0.5 * abs(float(true_delta[d])))
            for d in true_delta}


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    json_path = argv[0] if argv else DEFAULT_JSON
    if not os.path.exists(json_path):
        raise SystemExit(f"FATAL: not found: {json_path}")

    with open(json_path, encoding="utf-8") as fh:
        r = json.load(fh)

    td = r["veto"]["truediff"]
    rec = td["recovered_delta"]
    true = td["true_delta"]
    shrunk_pd = per_dial_shrunk(rec, true)
    print(f"[regate] recovered_delta = {rec}")
    print(f"[regate] true_delta      = {true}")
    print(f"[regate] PER-DIAL shrunk = {shrunk_pd}")

    # preserve the raw cluster output once (don't clobber an existing backup)
    raw_backup = json_path.replace(".json", ".cluster_raw.json")
    if not os.path.exists(raw_backup):
        shutil.copy2(json_path, raw_backup)
        print(f"[regate] preserved raw cluster output -> {raw_backup}")

    # re-apply the gate per regime with the per-dial shrunk mapping
    td["shrunk"] = shrunk_pd                       # canonicalize to per-dial in-place
    regimes = [k for k in ("expert", "naive") if k in r.get("gate", {})]
    print("\n[regate] verdict (per-dial shrunk):")
    for reg in regimes:
        truediff_res = {"recovered_delta": rec, "shrunk": shrunk_pd}
        g = recovery_gate(r["point"][reg], r["confusion"][reg], truediff_res,
                          r["veto"]["cond"][reg], regime=reg)
        r["gate"][reg] = g
        pdt = g["per_dial_trust"]
        print("  {:7s}: ".format(reg)
              + "  ".join(f"{d}={pdt.get(d)}" for d in ("sharpness", "caution", "timing")))

    r.setdefault("meta", {})["shrunk_veto"] = (
        "per-dial (gate_criteria.md L11); re-derived from the cached cluster "
        "recovered_delta/true_delta by _regate_perdial_shrunk.py. The raw cluster "
        "output (scalar-shrunk artifact) is preserved alongside as *.cluster_raw.json.")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(r, fh, indent=2)
    print(f"\n[regate] wrote corrected verdict -> {json_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    sys.exit(main())
