#!/usr/bin/env python3
"""
Container startup bootstrap.

The DuckDB file lives on a Railway volume, so it survives redeploys — but
the FIRST boot against a fresh volume has an empty database. This checks
`fight_stats_raw` and, only if it's empty, runs the ingestion pipeline once
(downloads the public HuggingFace UFC datasets, no auth needed) before
starting the API. Later boots find the volume already populated and skip
straight to serving, so restarts stay fast and don't hit HuggingFace on
every deploy.
"""
import os
import subprocess
import sys

import duckdb

DB_PATH = "data/processed/octagon.duckdb"


def needs_ingestion() -> bool:
    os.makedirs("data/processed", exist_ok=True)
    con = duckdb.connect(DB_PATH)
    try:
        try:
            n = con.execute("SELECT COUNT(*) FROM fight_stats_raw").fetchone()[0]
        except duckdb.CatalogException:
            n = 0
        return n == 0
    finally:
        con.close()


def main() -> None:
    if needs_ingestion():
        print("[bootstrap] fight_stats_raw empty -- running ingestion pipeline...", flush=True)
        subprocess.run([sys.executable, "-m", "ingestion.hf_pipeline"], check=True)
    else:
        print("[bootstrap] fight_stats_raw already populated -- skipping ingestion.", flush=True)

    import uvicorn
    uvicorn.run(
        "inference_onnx.predict:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080)),
    )


if __name__ == "__main__":
    main()
