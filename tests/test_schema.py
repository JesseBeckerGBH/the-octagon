import duckdb

from scripts.create_schema import init_schema


def test_schema_creates_all_expected_tables():
    con = duckdb.connect(":memory:")
    try:
        init_schema(con)
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
            ).fetchall()
        }
        expected = {
            "events", "fights", "odds_history",
            "predictions", "validation_log", "monthly_reports",
        }
        assert expected.issubset(tables)
    finally:
        con.close()


def test_schema_is_idempotent():
    con = duckdb.connect(":memory:")
    try:
        init_schema(con)
        init_schema(con)  # must not raise on second call
    finally:
        con.close()
