#!/usr/bin/env python3
"""
Create (or upgrade) THE OCTAGON's DuckDB schema.

This is the single source of truth for table shapes. Import init_schema()
from here rather than duplicating CREATE TABLE statements elsewhere —
the original project grew several slightly-different copies of this schema
across ad-hoc scripts, which is exactly the drift this module exists to stop.
"""

from pathlib import Path

import duckdb

DB_PATH = Path("data/processed/octagon.duckdb")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT PRIMARY KEY,
    name          TEXT,
    date          TEXT,
    location      TEXT,
    scraped_at    TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fights (
    fight_id      TEXT PRIMARY KEY,
    event_id      TEXT REFERENCES events(event_id),
    fighter_a     TEXT,
    fighter_b     TEXT,
    winner        TEXT,
    method        TEXT,
    round         INTEGER,
    time          TEXT
);

-- Rich per-fight stats from the HuggingFace/Kaggle ingestion pipeline
-- (ingestion/hf_pipeline.py) — strikes, takedowns, control time, etc. per
-- fighter per fight. This is what features/leak_safe_features.py builds
-- the production-grade feature set from; the scraper-based `fights` table
-- above stays as the lighter-weight, always-available fallback.
CREATE TABLE IF NOT EXISTS fight_stats_raw (
    fight_id                   TEXT PRIMARY KEY,
    event_name                 TEXT,
    event_date                 DATE,
    weight_class               TEXT,
    fighter_a_name              TEXT,
    fighter_b_name              TEXT,
    winner_is_a                 BOOLEAN,
    finish_method               TEXT,
    finish_round                INTEGER,
    fighter_a_kd                 REAL,
    fighter_a_sig_str_landed      REAL,
    fighter_a_sig_str_attempted   REAL,
    fighter_a_sig_str_pct         REAL,
    fighter_a_td_landed           REAL,
    fighter_a_td_attempted        REAL,
    fighter_a_td_pct              REAL,
    fighter_a_sub_att             REAL,
    fighter_a_rev                 REAL,
    fighter_a_ctrl_time_seconds   REAL,
    fighter_b_kd                 REAL,
    fighter_b_sig_str_landed      REAL,
    fighter_b_sig_str_attempted   REAL,
    fighter_b_sig_str_pct         REAL,
    fighter_b_td_landed           REAL,
    fighter_b_td_attempted        REAL,
    fighter_b_td_pct              REAL,
    fighter_b_sub_att             REAL,
    fighter_b_rev                 REAL,
    fighter_b_ctrl_time_seconds   REAL,
    source                      TEXT,
    ingested_at                 TIMESTAMP
);

CREATE TABLE IF NOT EXISTS odds_history (
    odds_id       TEXT PRIMARY KEY,
    fight_id      TEXT REFERENCES fights(fight_id),
    timestamp     TIMESTAMP,
    book          TEXT,
    moneyline_a   REAL,
    moneyline_b   REAL
);

-- Every prediction is written here BEFORE the fight happens.
-- This table is append-only in practice: it is the audit trail that proves
-- predictions weren't tuned after the fact.
CREATE TABLE IF NOT EXISTS predictions (
    pred_id               TEXT PRIMARY KEY,
    fight_id              TEXT REFERENCES fights(fight_id),
    generated_at          TIMESTAMP,
    blended_win_prob_a    REAL,
    predicted_winner      TEXT,
    confidence_tier       TEXT,
    method_probs          JSON,
    edge_percent          REAL,
    paper_bet_size        REAL,
    ou_timing_signal      BOOLEAN,
    sharp_alignment       BOOLEAN,
    model_versions        JSON
);

-- Every prediction is resolved here AFTER the fight, and never edited again.
CREATE TABLE IF NOT EXISTS validation_log (
    validation_id      TEXT PRIMARY KEY,
    pred_id            TEXT REFERENCES predictions(pred_id),
    actual_winner      TEXT,
    actual_method      TEXT,
    actual_round       INTEGER,
    delta_win_prob     REAL,
    brier_score_contrib REAL,
    log_loss_contrib   REAL,
    clv_realized       REAL,
    paper_outcome      TEXT,
    roi_contrib        REAL,
    validated_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monthly_reports (
    report_id       TEXT PRIMARY KEY,
    period_start    DATE,
    period_end      DATE,
    n_predictions   INTEGER,
    win_rate        REAL,
    brier_score     REAL,
    avg_clv         REAL,
    roi             REAL,
    generated_at    TIMESTAMP
);
"""


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(SCHEMA_SQL)


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        init_schema(con)
        print("Schema created/verified successfully.")
    finally:
        con.close()


if __name__ == "__main__":
    main()
