"""
Precompute a small demo dataset for the Streamlit dashboard.
Uses DuckDB to extract sessions without loading the full CSV into RAM.
"""

from pathlib import Path

import duckdb
import polars as pl

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def prepare_demo_data(n_sessions: int = 200, seed: int = 42) -> None:
    print("Connecting to DuckDB...")
    con = duckdb.connect()
    con.execute("SET memory_limit='3GB'")
    con.execute("SET threads=2")
    con.execute("SET temp_directory='data/tmp'")

    print("Sampling sessions...")
    sessions = con.execute(f"""
        WITH counts AS (
            SELECT user_session,
                COUNT(*) AS n,
                MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
            FROM read_csv('data/raw/2019-Oct.csv', header=true, ignore_errors=true)
            WHERE user_session IS NOT NULL
              AND product_id IS NOT NULL
              AND price > 0
            GROUP BY user_session
            HAVING n BETWEEN 5 AND 15
        ),
        purchased AS (
            SELECT user_session FROM counts
            WHERE purchased = 1
            ORDER BY random()
            LIMIT {n_sessions // 2}
        ),
        not_purchased AS (
            SELECT user_session FROM counts
            WHERE purchased = 0
            ORDER BY random()
            LIMIT {n_sessions // 2}
        )
        SELECT user_session FROM purchased
        UNION ALL
        SELECT user_session FROM not_purchased
    """).arrow()

    selected = pl.from_arrow(sessions)
    print(f"Selected {len(selected)} sessions")

    # Save session list as temp file for second query
    selected.write_parquet("data/tmp/selected_sessions.parquet")

    print("Extracting events...")
    events = con.execute("""
        SELECT
            e.user_session,
            e.event_time,
            e.event_type,
            CAST(e.product_id AS BIGINT) AS product_id,
            e.category_code,
            e.brand,
            CAST(e.price AS FLOAT) AS price
        FROM read_csv('data/raw/2019-Oct.csv', header=true, ignore_errors=true) e
        JOIN read_parquet('data/tmp/selected_sessions.parquet') s
          ON e.user_session = s.user_session
        WHERE e.product_id IS NOT NULL
          AND e.price > 0
        ORDER BY e.user_session, e.event_time
    """).arrow()

    demo = pl.from_arrow(events)

    catalog = (
        demo.select(["product_id", "category_code", "brand", "price"])
        .unique("product_id")
        .drop_nulls()
    )

    PROCESSED_DIR.mkdir(exist_ok=True)
    demo.write_parquet(PROCESSED_DIR / "demo_sessions.parquet")
    catalog.write_parquet(PROCESSED_DIR / "demo_catalog.parquet")

    print(f"Events saved: {len(demo):,}")
    print(f"Products in catalog: {len(catalog):,}")
    print("Done → data/processed/demo_sessions.parquet")
    print("Done → data/processed/demo_catalog.parquet")


if __name__ == "__main__":
    prepare_demo_data()
