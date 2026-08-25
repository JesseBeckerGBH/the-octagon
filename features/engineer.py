#!/usr/bin/env python3
"""
THE OCTAGON — feature engineering entry point.

Two feature sets exist, selected automatically by build_feature_table():

  1. The real one (features/leak_safe_features.py, ported from the live
     production engine) — career/rolling stat differentials, Markov form,
     style-cluster matchups, method-of-victory labels — built from
     ingestion/hf_pipeline.py's `fight_stats_raw` table. This is what the
     Council actually trains on once you've run the HuggingFace ingestion.

  2. The original placeholder (reach_diff/age_diff/slpm_diff, always 0.0)
     — kept as a graceful fallback for a fresh clone that has only run
     ingestion/ufc_scraper.py and hasn't populated fight_stats_raw yet, so
     the rest of the pipeline (schema, council, validator, tests) still
     runs end-to-end without erroring on missing data.

Prophets should get their feature columns from get_feature_cols() below
rather than hardcoding a list — that's what lets the same prophet code
work against either feature set without a code change.
"""

from pathlib import Path

import duckdb
import polars as pl

DB_PATH = Path("data/processed/octagon.duckdb")

# Only used by the placeholder fallback path.
LEGACY_FEATURE_COLUMNS = ["reach_diff", "age_diff", "slpm_diff"]


def load_fights(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    df = con.execute(
        """
        SELECT f.fight_id, f.event_id, f.fighter_a, f.fighter_b, f.winner,
               f.method, f.round, e.date
        FROM fights f
        JOIN events e ON f.event_id = e.event_id
        """
    ).pl()
    return df


def rolling_form(fighter_history: pl.DataFrame, window: int = 5) -> pl.DataFrame:
    """Win rate over a fighter's last `window` fights. Superseded by
    features/markov_form.py's fuller 4-state chain for anything
    council-facing, but kept as a standalone utility — it's a simpler,
    dependency-free building block that's handy on its own (e.g. for a
    quick "how hot is this fighter" display without pulling in the full
    Markov machinery).
    """
    return fighter_history.sort("date").with_columns(
        pl.col("won")
        .rolling_mean(window_size=window, min_periods=1)
        .shift(1)  # never leak the outcome of the fight being predicted
        .over("fighter")
        .alias(f"win_rate_last_{window}")
    )


def _placeholder_feature_table(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """The original honest-but-empty differential table — see module
    docstring. Only reached when fight_stats_raw has no rows yet.
    """
    fights = load_fights(con)
    if fights.is_empty():
        return fights
    return fights.with_columns(
        [
            pl.lit(0.0).alias("reach_diff"),
            pl.lit(0.0).alias("age_diff"),
            pl.lit(0.0).alias("slpm_diff"),
            (pl.col("winner") == pl.col("fighter_a")).cast(pl.Int8).alias("label_a_won"),
        ]
    )


def build_feature_table(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    """Producers one row per fight with A-vs-B features and a
    `label_a_won` target column. Tries the real leak-safe feature set
    first (features/leak_safe_features.py, sourced from
    ingestion/hf_pipeline.py's fight_stats_raw); falls back to the
    placeholder set if that table is empty.
    """
    try:
        n_raw = con.execute("SELECT COUNT(*) FROM fight_stats_raw").fetchone()[0]
    except duckdb.CatalogException:
        n_raw = 0  # schema not migrated to include fight_stats_raw yet

    if n_raw > 0:
        from features.leak_safe_features import build_and_store

        pdf = build_and_store(con)
        if not pdf.empty:
            pdf = pdf.rename(columns={"winner_is_a": "label_a_won"})
            return pl.from_pandas(pdf)

    return _placeholder_feature_table(con)


def get_feature_cols(features: pl.DataFrame) -> list[str]:
    """Numeric columns safe to train a prophet on — excludes the target
    (`label_a_won`) and any identifier/metadata column. Works against
    either feature set: on the placeholder table this returns
    LEGACY_FEATURE_COLUMNS; on the real leak-safe table it returns the
    full fa_/fb_/diff_/style_/form_/ou_ column set.
    """
    exclude_exact = {
        "label_a_won", "winner_is_a", "fight_id", "event_id", "event_name",
        "event_date_parsed", "weight_class", "header_fighter_a_name",
        "header_fighter_b_name", "header_finish_details_detailed",
        "header_round_detailed", "method_class", "method_class_idx",
        "fighter_a", "fighter_b", "winner", "method", "round", "date",
    }
    numeric_dtypes = {
        pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
        pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    }
    return [
        name for name, dtype in zip(features.columns, features.dtypes)
        if name not in exclude_exact and dtype in numeric_dtypes
    ]


def main() -> None:
    con = duckdb.connect(str(DB_PATH))
    try:
        table = build_feature_table(con)
        print(f"Built feature table with {len(table)} rows, {len(get_feature_cols(table))} feature columns.")
        print(table.head())
    finally:
        con.close()


if __name__ == "__main__":
    main()
