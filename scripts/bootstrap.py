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
import time

import duckdb

DB_PATH = "data/processed/octagon.duckdb"


def wait_for_volume(tries: int = 10, delay: float = 1.0) -> None:
    """Block until the mounted volume is actually stable.

    Railway's bind-mount for a persistent volume can finish attaching a
    beat AFTER this process has already started -- if we write anything to
    the mount path before that happens, the real mount lands on top of it
    and silently hides it. Detect that by writing a marker, sleeping, and
    checking it's still there; if the mount swapped underneath us, the
    marker vanishes and we retry on the (now real) filesystem.
    """
    marker_dir = "data"
    marker = os.path.join(marker_dir, ".bootstrap_marker")
    for attempt in range(1, tries + 1):
        os.makedirs(marker_dir, exist_ok=True)
        with open(marker, "w") as f:
            f.write(str(time.time()))
        time.sleep(delay)
        if os.path.exists(marker):
            print(f"[bootstrap] volume stable after {attempt} check(s).", flush=True)
            return
        print(f"[bootstrap] volume not stable yet (attempt {attempt}/{tries}) -- retrying...", flush=True)
    print("[bootstrap] WARNING: volume never stabilized -- proceeding anyway.", flush=True)


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
    wait_for_volume()

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
