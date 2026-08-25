#!/usr/bin/env python3
"""
UFC data ingestion — HuggingFace + Kaggle historical merge.

Ported from the live production engine (ufc-predictions-saas/engine/
ingestion/ingest.py), which is the actual source of the fight-level
striking/grappling/control-time data that features/leak_safe_features.py
needs. Octagon's own ingestion/ufc_scraper.py only ever captured
fighter/winner/method/round — never the per-fight stat breakdown the
Council's real feature set depends on — so this isn't a second, competing
ingestion path; it's the one that unblocks everything downstream of it.

Downloads 2 HuggingFace datasets (fight-level + fighter-profile) and,
optionally, a locally-supplied Kaggle "UFC Complete Dataset" CSV pair for
pre-2013 history (not auto-downloaded — Kaggle requires auth; drop
large_dataset.csv/medium_dataset.csv into data/raw/kaggle/ yourself if you
want that extra depth). Writes into DuckDB's `fight_stats_raw` table
(scripts/create_schema.py) rather than only CSVs, per Octagon's "DuckDB is
the single source of truth" principle — CSV intermediates are still
written to data/processed/ for parity with the production pipeline and
for quick inspection.

Usage:
    python -m ingestion.hf_pipeline              # download + load into DuckDB
    python -m ingestion.hf_pipeline --skip-download
    python -m ingestion.hf_pipeline --info
"""

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"
DB_PATH = PROC_DIR / "octagon.duckdb"

DOWNLOADS = {
    "ufc_2013_2025.jsonl": {
        "url": "https://huggingface.co/datasets/01x3ATM3/UFC_FIGHT_DATA_2013_TO_2025/resolve/main/output.jsonl",
        "desc": "5,900+ fights (2013-2025) with full striking/grappling breakdown",
    },
    "ufc_fighters_stats.json": {
        "url": "https://huggingface.co/datasets/tawhidmonowar/ufc-fighters-stats-and-records-dataset/resolve/main/ufc_fighters_stats_and_records.json",
        "desc": "2,990 fighter profiles with career stats and fight history",
    },
}


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def download_all() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname, meta in DOWNLOADS.items():
        dest = RAW_DIR / fname
        if dest.exists() and dest.stat().st_size > 1000:
            log(f"  SKIP {fname} (already downloaded, {dest.stat().st_size:,} bytes)")
            continue
        log(f"  Downloading {fname}...  ({meta['desc']})")
        try:
            urllib.request.urlretrieve(meta["url"], str(dest))
            log(f"    OK ({dest.stat().st_size:,} bytes)")
        except Exception as e:  # noqa: BLE001 - deliberately broad, this is best-effort ingestion
            log(f"    ERROR: {e} — continuing without this source")


def load_and_clean() -> pd.DataFrame:
    """Load raw HuggingFace (+ optional Kaggle) data into one clean,
    chronologically-sorted fight-level DataFrame.
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    fights_path = RAW_DIR / "ufc_2013_2025.jsonl"
    if not fights_path.exists():
        raise FileNotFoundError(
            f"{fights_path} not found — run download_all() first, or supply it manually."
        )

    log("Loading fights (2013-2025)...")
    fights = []
    with open(fights_path) as f:
        for line in f:
            fights.append(json.loads(line))
    df = pd.DataFrame(fights)

    df["event_date_parsed"] = pd.to_datetime(df["event_date"], format="mixed", errors="coerce")
    df = df.sort_values("event_date_parsed").reset_index(drop=True)
    df["winner_is_a"] = (df["header_fighter_a_outcome"] == "W").astype(int)

    log(f"  {len(df)} fights loaded ({df['event_date_parsed'].min().date()} to {df['event_date_parsed'].max().date()})")

    kaggle_dir = RAW_DIR / "kaggle"
    if (kaggle_dir / "large_dataset.csv").exists() and (kaggle_dir / "medium_dataset.csv").exists():
        log("Merging Kaggle UFC Complete Dataset (pre-2013 history)...")
        df = _merge_kaggle(df, kaggle_dir / "large_dataset.csv", kaggle_dir / "medium_dataset.csv")
    else:
        log("  Kaggle data not found in data/raw/kaggle/ — skipping (HuggingFace-only, 2013+)")

    df.to_csv(PROC_DIR / "ufc_fights_clean.csv", index=False)
    return df


def _merge_kaggle(df_hf: pd.DataFrame, large_path: Path, medium_path: Path) -> pd.DataFrame:
    """Best-effort merge of Kaggle's 1994-2024 dataset for older history.
    Deliberately forgiving: any failure here just means we keep
    HuggingFace-only coverage rather than losing the whole pipeline.
    """
    try:
        df_large = pd.read_csv(large_path)
        df_medium = pd.read_csv(medium_path)
        df_medium["date_parsed"] = pd.to_datetime(df_medium["date"], format="mixed", errors="coerce")

        merged = df_large.merge(
            df_medium[["event", "r_fighter", "b_fighter", "date_parsed", "method_detailed"]]
            .rename(columns={"event": "event_name"}),
            on=["event_name", "r_fighter", "b_fighter"], how="left",
        )
        merged = merged.dropna(subset=["date_parsed"])

        col_map = {
            "event_name": "event_name", "r_fighter": "header_fighter_a_name",
            "b_fighter": "header_fighter_b_name", "weight_class": "weight_class",
            "method": "header_finish_details_detailed", "date_parsed": "event_date_parsed",
            "r_kd": "fighter_a_kd", "r_sig_str": "fighter_a_sig_str_landed",
            "r_sig_str_att": "fighter_a_sig_str_attempted", "r_sig_str_acc": "fighter_a_sig_str_pct",
            "r_td": "fighter_a_td_landed", "r_td_att": "fighter_a_td_attempted",
            "r_td_acc": "fighter_a_td_pct", "r_sub_att": "fighter_a_sub_att",
            "r_rev": "fighter_a_rev", "r_ctrl_sec": "fighter_a_ctrl_time_seconds",
            "b_kd": "fighter_b_kd", "b_sig_str": "fighter_b_sig_str_landed",
            "b_sig_str_att": "fighter_b_sig_str_attempted", "b_sig_str_acc": "fighter_b_sig_str_pct",
            "b_td": "fighter_b_td_landed", "b_td_att": "fighter_b_td_attempted",
            "b_td_acc": "fighter_b_td_pct", "b_sub_att": "fighter_b_sub_att",
            "b_rev": "fighter_b_rev", "b_ctrl_sec": "fighter_b_ctrl_time_seconds",
        }
        converted = merged.rename(columns=col_map)
        converted["winner_is_a"] = (converted["winner"] == "Red").astype(int)
        converted["header_fighter_a_outcome"] = np.where(converted["winner"] == "Red", "W", "L")
        converted["header_fighter_b_outcome"] = np.where(converted["winner"] == "Blue", "W", "L")
        keep = [v for v in col_map.values() if v in converted.columns] + [
            "winner_is_a", "header_fighter_a_outcome", "header_fighter_b_outcome",
        ]
        converted = converted[[c for c in keep if c in converted.columns]]

        hf_keys = set(zip(
            df_hf["event_name"].str.lower().str.strip(),
            df_hf["header_fighter_a_name"].str.lower().str.strip(),
            df_hf["header_fighter_b_name"].str.lower().str.strip(),
        ))
        is_new = [
            (e.lower().strip(), a.lower().strip(), b.lower().strip()) not in hf_keys
            for e, a, b in zip(converted["event_name"], converted["header_fighter_a_name"], converted["header_fighter_b_name"])
        ]
        new_rows = converted[is_new]
        log(f"  Kaggle: {len(converted)} total, {len(new_rows)} new (not already in HuggingFace)")

        combined = pd.concat([new_rows, df_hf], ignore_index=True)
        return combined.sort_values("event_date_parsed").reset_index(drop=True)
    except Exception as e:  # noqa: BLE001
        log(f"  Kaggle merge failed ({e}) — continuing with HuggingFace data only")
        return df_hf


def _clean(v):
    """NaN/NaT -> None (DuckDB will reject a bare float('nan') going into a
    typed column like finish_round INTEGER — this is what actually blew up
    the first real run against the live 5,902-fight HuggingFace dataset,
    where plenty of fights have a missing round/weight-class/etc.).
    """
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _clean_stat(v) -> float:
    cleaned = _clean(v)
    return 0.0 if cleaned is None else float(cleaned)


def load_into_duckdb(df: pd.DataFrame, con: duckdb.DuckDBPyConnection) -> int:
    """Upsert cleaned fights into fight_stats_raw. Returns rows written."""
    stat_cols = [
        "fighter_a_kd", "fighter_a_sig_str_landed", "fighter_a_sig_str_attempted", "fighter_a_sig_str_pct",
        "fighter_a_td_landed", "fighter_a_td_attempted", "fighter_a_td_pct", "fighter_a_sub_att",
        "fighter_a_rev", "fighter_a_ctrl_time_seconds",
        "fighter_b_kd", "fighter_b_sig_str_landed", "fighter_b_sig_str_attempted", "fighter_b_sig_str_pct",
        "fighter_b_td_landed", "fighter_b_td_attempted", "fighter_b_td_pct", "fighter_b_sub_att",
        "fighter_b_rev", "fighter_b_ctrl_time_seconds",
    ]
    for c in stat_cols:
        if c not in df.columns:
            df[c] = 0.0

    ingested_at = datetime.utcnow()
    records = []
    for _, row in df.iterrows():
        fight_id = f"{row.get('event_name', '')}_{row['header_fighter_a_name']}_{row['header_fighter_b_name']}"
        finish_round = _clean(row.get("header_round_detailed"))
        records.append((
            fight_id, _clean(row.get("event_name")), _clean(row.get("event_date_parsed")),
            _clean(row.get("weight_class")),
            row["header_fighter_a_name"], row["header_fighter_b_name"], bool(row["winner_is_a"]),
            _clean(row.get("header_finish_details_detailed")),
            int(finish_round) if finish_round is not None else None,
            *[_clean_stat(row.get(c)) for c in stat_cols],
            "huggingface+kaggle", ingested_at,
        ))

    con.executemany(
        """
        INSERT OR REPLACE INTO fight_stats_raw
        (fight_id, event_name, event_date, weight_class, fighter_a_name, fighter_b_name,
         winner_is_a, finish_method, finish_round,
         fighter_a_kd, fighter_a_sig_str_landed, fighter_a_sig_str_attempted, fighter_a_sig_str_pct,
         fighter_a_td_landed, fighter_a_td_attempted, fighter_a_td_pct, fighter_a_sub_att,
         fighter_a_rev, fighter_a_ctrl_time_seconds,
         fighter_b_kd, fighter_b_sig_str_landed, fighter_b_sig_str_attempted, fighter_b_sig_str_pct,
         fighter_b_td_landed, fighter_b_td_attempted, fighter_b_td_pct, fighter_b_sub_att,
         fighter_b_rev, fighter_b_ctrl_time_seconds,
         source, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        records,
    )
    return len(records)


def main() -> None:
    skip_dl = "--skip-download" in sys.argv
    info_only = "--info" in sys.argv

    if info_only:
        con = duckdb.connect(str(DB_PATH))
        try:
            n = con.execute("SELECT COUNT(*) FROM fight_stats_raw").fetchone()[0]
            log(f"fight_stats_raw: {n} fights")
        finally:
            con.close()
        return

    if not skip_dl:
        log("Downloading datasets from Hugging Face...")
        download_all()

    df = load_and_clean()

    from scripts.create_schema import init_schema
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    try:
        init_schema(con)
        n = load_into_duckdb(df, con)
        log(f"Loaded {n} fights into fight_stats_raw")
    finally:
        con.close()


if __name__ == "__main__":
    main()
