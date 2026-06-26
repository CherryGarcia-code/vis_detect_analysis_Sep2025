import pandas as pd

import inclusive_trusted as it


def test_compute_inclusive_tier_rules():
    # span<2 -> suspect regardless
    assert it.compute_inclusive_tier(1, 0) == "suspect"
    # span>=3, no warn -> trusted
    assert it.compute_inclusive_tier(3, 0) == "trusted"
    # span>=3, exactly one warn -> still trusted (the relaxation)
    assert it.compute_inclusive_tier(5, 1) == "trusted"
    # span>=3, two warns -> review (exceeds max_warn=1)
    assert it.compute_inclusive_tier(5, 2) == "review"
    # span==2 -> review (below min_span, not suspect)
    assert it.compute_inclusive_tier(2, 0) == "review"


def test_assign_inclusive_tiers_counts_keep_warns_only():
    # uid 1: span 3, one warn on a KEEP link -> inclusive trusted (shipped rule
    # would demote it to review). uid 2: span 3, two warn KEEP links -> review.
    # A warn on a SKIP link must NOT count.
    tracks = pd.DataFrame({
        "curated_uid": [1, 2],
        "kept_sessions": ["a;b;c", "a;b;c"],
    })
    links = pd.DataFrame({
        "liberal_uid": [1, 1, 1, 2, 2],
        "link_decision": ["KEEP", "KEEP", "SKIP", "KEEP", "KEEP"],
        "review_flag": [True, False, True, True, True],  # uid1: 1 KEEP-warn (+1 SKIP-warn ignored)
    })
    tiers = it.assign_inclusive_tiers(tracks, links, max_warn=1)
    assert tiers.tolist() == ["trusted", "review"]


def test_kept_pairs_from_normalizes_leading_zero_sessions():
    # Registry stores sessions zero-padded ("08092025"); curate_tracks writes
    # kept_sessions with the leading zero stripped ("8092025"). A raw string-equality
    # join would drop the single-digit-day session; the normalized join must keep it.
    tracks = pd.DataFrame({
        "curated_uid": [5],
        "kept_sessions": ["8092025;23062025"],   # 7-digit (stripped) + 8-digit
    })
    reg = pd.DataFrame({
        "session": ["08092025", "23062025"],      # padded, as written on disk
        "ks_unit_id": [11, 22],
        "dant_uid": [5, 5],
    })
    # norm collapses both forms to the same key (mimics session_date_key)
    norm = lambda s: str(s).strip().zfill(8)
    pairs = it.kept_pairs_from(tracks, reg, norm)
    assert pairs == {(5, "8092025"): 11, (5, "23062025"): 22}   # both kept, original tokens


def test_assign_inclusive_tiers_bridged_track_can_be_trusted():
    # A track with a bridge (n_bridged>0) but zero warns is trusted under the
    # inclusive rule (the shipped rule would force it to review).
    tracks = pd.DataFrame({"curated_uid": [7], "kept_sessions": ["a;b;c;d"]})
    links = pd.DataFrame({
        "liberal_uid": [7, 7, 7, 7],
        "link_decision": ["KEEP", "SKIP", "KEEP", "KEEP"],
        "review_flag": [False, False, False, False],
    })
    assert it.assign_inclusive_tiers(tracks, links, max_warn=1).tolist() == ["trusted"]
