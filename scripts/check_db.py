#!/usr/bin/env python3
"""Quick diagnostic: what tables exist and how many rows are in each."""

from pathlib import Path

import duckdb

DB_PATH = Path("data/processed/octagon.duckdb")


def main() -> None:
    if not DB_PATH.exists():
        print(f"No database found at {DB_PATH}. Run scripts/create_schema.py first.")
        return

    con = duckdb.connect(str(DB_PATH))
    try:
        tables = [t[0] for t in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
        ).fetchall()]

        print("=== DATABASE STATUS ===")
        print(f"Path: {DB_PATH}")
        print(f"Tables: {len(tables)}")
        for table in sorted(tables):
            count = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            print(f"  - {table}: {count} rows")
    finally:
        con.close()


if __name__ == "__main__":
    main()
