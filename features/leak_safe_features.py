"""
Leak-safe feature engineering — the real feature set behind the Council.

Ported from the live production engine (ufc-predictions-saas/engine/
ingestion/ingest.py::engineer_features()), which is what the "70+ leak-safe
features" claim in the analysis of thebeastufc.com actually refers to.
Octagon's original features/engineer.py produced 3 placeholder columns
(reach_diff/age_diff/slpm_diff, always zero) because the scraper never
captured per-fight stats — this is the real thing, built on top of
ingestion/hf_pipeline.py's fight_stats_raw table.

Every feature here is computed from a fighter's history STRICTLY BEFORE
the fight being featurized (expanding/rolling stats are shifted by one
row), which is what makes a walk-forward backtest against this feature set
honest rather than leaky. Read the inline comments before changing the
shift/ordering logic — that's the one thing in this file that's easy to
silently break.
"""

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from features.markov_form import add_markov_features
from features.ou_process import add_ou_features_placeholder
from features.style_cluster import add_style_features
from models.method_classifier import add_method_labels

DB_PATH = Path("data/processed/octagon.duckdb")

STAT_COLS = [
    "kd", "sig_str_landed", "sig_str_attempted", "sig_str_pct",
    "td_landed", "td_attempted", "td_pct", "sub_att", "rev", "ctrl_time_seconds",
]

WEIGHT_ORDER = {
    "Strawweight": 115, "Women's Strawweight": 115,
    "Flyweight": 125, "Women's Flyweight": 125,
    "Bantamweight": 135, "Women's Bantamweight": 135,
    "Featherweight": 145, "Women's Featherweight": 145,
    "Lightweight": 155, "Welterweight": 170, "Middleweight": 185,
    "Light Heavyweight": 205, "Heavyweight": 265, "Catch Weight": 0,
}


def load_raw_fights(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pull fight_stats_raw into the flat, pandas shape engineer_features()
    expects (header_fighter_a_name / header_fighter_a_outcome / etc. — the
    same column convention the production pipeline used, kept so this port
    stays a faithful transcription rather than a from-scratch rewrite).
    """
    df = con.execute("SELECT * FROM fight_stats_raw ORDER BY event_date").df()
    if df.empty:
        return df

    df = df.rename(columns={
        "event_date": "event_date_parsed",
        "fighter_a_name": "header_fighter_a_name",
        "fighter_b_name": "header_fighter_b_name",
        "finish_method": "header_finish_details_detailed",
        "finish_round": "header_round_detailed",
    })
    df["event_date_parsed"] = pd.to_datetime(df["event_date_parsed"])
    df["winner_is_a"] = df["winner_is_a"].astype(int)
    df["header_fighter_a_outcome"] = np.where(df["winner_is_a"] == 1, "W", "L")
    df["header_fighter_b_outcome"] = np.where(df["winner_is_a"] == 1, "L", "W")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the fight-level feature matrix. Each row = one fight with
    pre-fight features for both fighters; target = winner_is_a.
    """
    df = df.dropna(subset=["fighter_a_sig_str_landed", "fighter_b_sig_str_landed"]).copy()
    df = df.sort_values("event_date_parsed").reset_index(drop=True)

    records = []
    for idx, row in df.iterrows():
        for side, opp in (("a", "b"), ("b", "a")):
            records.append({
                "fight_idx": idx,
                "fighter": row[f"header_fighter_{side}_name"],
                "date": row["event_date_parsed"],
                "side": side,
                "won": 1 if row[f"header_fighter_{side}_outcome"] == "W" else 0,
                **{c: (row.get(f"fighter_{side}_{c}", 0) or 0) for c in STAT_COLS},
                "opp_sig_str_landed": row.get(f"fighter_{opp}_sig_str_landed", 0) or 0,
                "opp_kd": row.get(f"fighter_{opp}_kd", 0) or 0,
                "opp_td_landed": row.get(f"fighter_{opp}_td_landed", 0) or 0,
            })
    rec = pd.DataFrame(records).sort_values(["fighter", "date"]).reset_index(drop=True)

    career_stats = STAT_COLS + ["opp_sig_str_landed", "opp_kd", "opp_td_landed"]
    grouped = rec.groupby("fighter")

    # Expanding mean of ALL prior fights, shifted so the current fight is
    # excluded — this shift is the entire leak-safety guarantee.
    for col in career_stats:
        rec[f"career_{col}"] = grouped[col].transform(lambda x: x.expanding().mean().shift(1))
    rec["career_won"] = grouped["won"].transform(lambda x: x.expanding().mean().shift(1))
    rec["career_fights"] = grouped.cumcount()

    for col in ["won", "sig_str_pct", "td_pct", "kd"]:
        rec[f"recent3_{col}"] = grouped[col].transform(lambda x: x.rolling(3, min_periods=1).mean().shift(1))

    def _win_streak(series):
        streaks, current = [], 0
        for val in series:
            streaks.append(current)  # append BEFORE updating: leak-safe
            current = current + 1 if val == 1 else 0
        return pd.Series(streaks, index=series.index)

    rec["win_streak"] = grouped["won"].transform(_win_streak)
    rec["days_since_last"] = grouped["date"].transform(lambda x: x.diff().dt.days)

    feature_cols = (
        [f"career_{c}" for c in career_stats] + ["career_won"]
        + [f"recent3_{c}" for c in ["won", "sig_str_pct", "td_pct", "kd"]]
        + ["career_fights", "win_streak", "days_since_last"]
    )

    fa_stats = rec[rec["side"] == "a"].set_index("fight_idx")[feature_cols]
    fb_stats = rec[rec["side"] == "b"].set_index("fight_idx")[feature_cols]
    fa = fa_stats.rename(columns={c: f"fa_{c}" for c in feature_cols})
    fb = fb_stats.rename(columns={c: f"fb_{c}" for c in feature_cols})

    features = df[[
        "event_date_parsed", "event_name", "weight_class",
        "header_fighter_a_name", "header_fighter_b_name", "winner_is_a",
        "header_finish_details_detailed", "header_round_detailed",
    ]].copy()
    features = features.join(fa, how="left").join(fb, how="left")

    diff_pairs = [
        "career_won", "career_sig_str_pct", "career_kd", "career_td_pct",
        "career_sig_str_landed", "career_ctrl_time_seconds", "career_sub_att",
        "career_opp_sig_str_landed", "recent3_won", "win_streak", "career_fights",
    ]
    for col in diff_pairs:
        features[f"diff_{col}"] = features[f"fa_{col}"] - features[f"fb_{col}"]

    def _safe_ratio(a, b):
        return np.where(b > 0, a / b, 0)

    features["fa_strike_efficiency"] = _safe_ratio(
        features["fa_career_sig_str_landed"].values, features["fa_career_sig_str_attempted"].values
    )
    features["fb_strike_efficiency"] = _safe_ratio(
        features["fb_career_sig_str_landed"].values, features["fb_career_sig_str_attempted"].values
    )

    features["weight_lbs"] = features["weight_class"].map(WEIGHT_ORDER).fillna(0)
    features["fight_year"] = features["event_date_parsed"].dt.year
    features["fight_month"] = features["event_date_parsed"].dt.month

    features_ml = features.dropna(subset=["fa_career_won", "fb_career_won"])

    # Bolt-on feature modules — each is defensive (skips itself on error
    # rather than aborting the whole pipeline), matching production
    # behavior: a broken extra feature module should degrade gracefully,
    # not take down feature engineering entirely.
    for name, fn in (
        ("Markov form", lambda d: add_markov_features(d)),
        ("Style clustering", lambda d: add_style_features(d)),
        ("Method-of-victory labels", lambda d: add_method_labels(d)),
        ("OU placeholder columns", lambda d: add_ou_features_placeholder(d)),
    ):
        try:
            features = fn(features)
            features_ml = features.dropna(subset=["fa_career_won", "fb_career_won"])
        except Exception as e:  # noqa: BLE001
            print(f"  [leak_safe_features] {name} skipped: {e}")

    return features_ml


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric columns safe to train on — no target, no metadata, no
    leakage. Identical selection rule to the production backtest/train
    scripts so ported hyperparameters behave the same way here.
    """
    exclude_prefixes = ("event_", "header_", "winner_", "weight_class")
    exclude_exact = {
        "winner_is_a", "event_date_parsed", "event_name", "weight_class",
        "header_fighter_a_name", "header_fighter_b_name", "header_winner_name",
        "header_finish_details_detailed", "header_round_detailed",
        "method_class", "method_class_idx",
    }
    cols = []
    for c in df.columns:
        if c in exclude_exact or any(c.startswith(p) for p in exclude_prefixes):
            continue
        if df[c].dtype in (np.float64, np.int64, float, int):
            cols.append(c)
    return cols


def _num(v) -> float:
    try:
        f = float(v)
        return 0.0 if np.isnan(f) else f
    except (ValueError, TypeError):
        return 0.0


def build_fighter_lookup(features_ml: pd.DataFrame) -> dict[str, dict]:
    """Most-recent per-fighter feature vector (fa_/fb_ prefix stripped),
    keyed by fighter name. Ported from production's
    engine/model/train_ensemble.py::build_fighter_lookup — this is what
    lets /predict answer "Fighter A vs Fighter B" without re-running the
    whole ingestion+feature pipeline per request; a fighter's row is just
    whichever side (fa_/fb_) they were on in their most recent fight.
    """
    if features_ml.empty:
        return {}
    df = features_ml.sort_values("event_date_parsed")
    fa_cols = [c for c in df.columns if c.startswith("fa_")]
    fb_cols = [c for c in df.columns if c.startswith("fb_")]
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        a, b = row.get("header_fighter_a_name"), row.get("header_fighter_b_name")
        if isinstance(a, str) and a:
            lookup[a] = {c[3:]: _num(row[c]) for c in fa_cols}
        if isinstance(b, str) and b:
            lookup[b] = {c[3:]: _num(row[c]) for c in fb_cols}
    return lookup


def compute_medians(features_ml: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Column medians for filling gaps at prediction time (an unseen
    fighter, a feature that wasn't computed for some historical reason).
    """
    return {c: _num(np.nanmedian(features_ml[c].values)) if c in features_ml.columns else 0.0
            for c in feature_cols}


def build_prediction_row(
    feature_cols: list[str], fighter_a: str, fighter_b: str,
    lookup: dict[str, dict], medians: dict[str, float],
) -> pd.DataFrame:
    """One-row feature frame for a hypothetical Fighter A vs Fighter B
    matchup, built from each fighter's latest known stats. Falls back to
    the training-set medians for any fighter (or feature) we don't have
    data for — a "we don't know this fighter" fighter still gets a
    plausible, roughly-neutral row rather than an error.
    """
    A, B = lookup.get(fighter_a), lookup.get(fighter_b)
    row = {}
    for c in feature_cols:
        if A is None or B is None:
            row[c] = medians.get(c, 0.0)
        elif c.startswith("fa_"):
            row[c] = A.get(c[3:], medians.get(c, 0.0))
        elif c.startswith("fb_"):
            row[c] = B.get(c[3:], medians.get(c, 0.0))
        elif c.startswith("diff_"):
            base = c[5:]
            va, vb = A.get(base), B.get(base)
            row[c] = (va - vb) if (va is not None and vb is not None) else medians.get(c, 0.0)
        else:
            row[c] = medians.get(c, 0.0)
    return pd.DataFrame([row])


def build_and_store(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Full pipeline entry point: fight_stats_raw -> engineered features,
    also persisted as data/processed/ufc_features_ml_ready.csv for the
    backtesting engine and the FastAPI service to load without a DuckDB
    round-trip on every request.
    """
    raw = load_raw_fights(con)
    if raw.empty:
        return raw
    features_ml = engineer_features(raw)
    out_path = Path("data/processed/ufc_features_ml_ready.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    features_ml.to_csv(out_path, index=False)
    return features_ml


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        features = build_and_store(con)
        print(f"Built feature matrix: {features.shape[0]} fights x {features.shape[1]} columns")
        if not features.empty:
            print(f"ML feature columns: {len(get_feature_cols(features))}")
    finally:
        con.close()


if __name__ == "__main__":
    main()
