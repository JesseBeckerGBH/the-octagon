import pandas as pd

from features.leak_safe_features import (
    build_fighter_lookup,
    build_prediction_row,
    compute_medians,
    engineer_features,
    get_feature_cols,
)


def _synthetic_raw_fights() -> pd.DataFrame:
    """Four fights: A beats B, A beats C, B beats C, then A vs B again.
    Every fight after the first has at least one debutant on one side
    (dropped by the leak-safe dropna) EXCEPT the final A-vs-B rematch,
    where both fighters already have prior history — that's the row used
    to check the career-stat shift is truly leak-safe.
    """
    dates = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01", "2020-04-01"])
    rows = []
    fights = [("A", "B", True), ("A", "C", True), ("B", "C", True), ("A", "B", False)]
    for i, (a, b, a_won) in enumerate(fights):
        rows.append({
            "event_date_parsed": dates[i], "event_name": f"Event {i}", "weight_class": "Lightweight",
            "header_fighter_a_name": a, "header_fighter_b_name": b,
            "header_fighter_a_outcome": "W" if a_won else "L",
            "header_fighter_b_outcome": "L" if a_won else "W",
            "winner_is_a": int(a_won),
            "header_finish_details_detailed": "Decision (Unanimous)",
            "header_round_detailed": 3,
            "fighter_a_kd": 1.0, "fighter_a_sig_str_landed": 40.0, "fighter_a_sig_str_attempted": 80.0,
            "fighter_a_sig_str_pct": 0.5, "fighter_a_td_landed": 1.0, "fighter_a_td_attempted": 2.0,
            "fighter_a_td_pct": 0.5, "fighter_a_sub_att": 0.0, "fighter_a_rev": 0.0,
            "fighter_a_ctrl_time_seconds": 60.0,
            "fighter_b_kd": 0.0, "fighter_b_sig_str_landed": 30.0, "fighter_b_sig_str_attempted": 70.0,
            "fighter_b_sig_str_pct": 0.4, "fighter_b_td_landed": 0.0, "fighter_b_td_attempted": 1.0,
            "fighter_b_td_pct": 0.0, "fighter_b_sub_att": 0.0, "fighter_b_rev": 0.0,
            "fighter_b_ctrl_time_seconds": 30.0,
        })
    return pd.DataFrame(rows)


def test_engineer_features_drops_debut_fights_with_no_history():
    raw = _synthetic_raw_fights()
    features_ml = engineer_features(raw)
    # Fight 0 (A vs B, both debuts) has no prior history for either side
    # and must be dropped — that's the leak-safety guarantee.
    assert len(features_ml) < len(raw)
    assert "fa_career_won" in features_ml.columns


def test_leak_safety_career_stats_only_reflect_strictly_prior_fights():
    raw = _synthetic_raw_fights()
    features_ml = engineer_features(raw)
    # By the A-vs-B rematch (fight 3), A has two priors (beat B, beat C
    # -> career_won=1.0) and B has two priors (lost to A, beat C ->
    # career_won=0.5) — neither reflects fight 3's own outcome (A loses
    # the rematch), which is what "leak-safe" means here.
    row = features_ml[
        (features_ml["header_fighter_a_name"] == "A")
        & (features_ml["header_fighter_b_name"] == "B")
        & (features_ml["event_date_parsed"] == pd.Timestamp("2020-04-01"))
    ]
    assert not row.empty
    assert row.iloc[0]["fa_career_won"] == 1.0
    assert row.iloc[0]["fb_career_won"] == 0.5


def test_get_feature_cols_excludes_metadata_and_target():
    raw = _synthetic_raw_fights()
    features_ml = engineer_features(raw)
    cols = get_feature_cols(features_ml)
    assert "winner_is_a" not in cols
    assert "header_fighter_a_name" not in cols
    assert "event_date_parsed" not in cols
    assert len(cols) > 0


def test_fighter_lookup_and_prediction_row_roundtrip():
    raw = _synthetic_raw_fights()
    features_ml = engineer_features(raw)
    cols = get_feature_cols(features_ml)
    lookup = build_fighter_lookup(features_ml)
    medians = compute_medians(features_ml, cols)

    assert "A" in lookup or "B" in lookup  # at least one fighter has a post-history row

    row = build_prediction_row(cols, "A", "Someone New", lookup, medians)
    assert list(row.columns) == cols
    assert len(row) == 1
