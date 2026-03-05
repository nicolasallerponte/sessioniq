"""
Session-level feature engineering.
Groups raw events into sessions and computes predictive features.
Only uses the first `max_events` events per session to simulate
real-time prediction mid-session (no leakage from future events).
"""

from pathlib import Path

import polars as pl

PROCESSED_DIR = Path("data/processed")


def build_session_features(lf: pl.LazyFrame, max_events: int = 3) -> pl.LazyFrame:
    """
    Aggregate raw events into one row per session with predictive features.
    Only uses the first `max_events` events per session to simulate
    real-time prediction mid-session (no leakage from future events).
    """
    return (
        lf.sort("event_time")
        .with_columns(
            pl.col("event_time")
            .rank("ordinal")
            .over("user_session")
            .alias("event_rank")
        )
        .filter(pl.col("event_rank") <= max_events)
        .group_by("user_session")
        .agg(
            [
                pl.len().alias("n_events_observed"),
                (pl.col("event_type") == "view").sum().alias("n_views"),
                (pl.col("event_type") == "cart").sum().alias("n_carts"),
                (pl.col("event_type") == "remove_from_cart").sum().alias("n_removals"),
                pl.col("product_id").n_unique().alias("n_unique_products"),
                pl.col("category_code").n_unique().alias("n_unique_categories"),
                pl.col("brand").n_unique().alias("n_unique_brands"),
                pl.col("price").mean().alias("avg_price"),
                pl.col("price").max().alias("max_price"),
                (pl.col("event_time").max() - pl.col("event_time").min())
                .dt.total_seconds()
                .alias("session_duration_seconds"),
                pl.col("event_time")
                .filter(pl.col("event_type") == "cart")
                .min()
                .alias("first_cart_time"),
                pl.col("event_time").min().alias("session_start"),
                pl.col("user_id").first().alias("user_id"),
            ]
        )
        .with_columns(
            [
                (pl.col("n_carts") / (pl.col("n_views") + 1)).alias("cart_view_ratio"),
                (pl.col("first_cart_time") - pl.col("session_start"))
                .dt.total_seconds()
                .alias("seconds_to_first_cart"),
            ]
        )
        .drop(["first_cart_time", "session_start"])
        .fill_null(0)
    )


def build_labels(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Compute ground truth labels from the FULL session.
    Kept separate to avoid leakage into features.
    """
    return lf.group_by("user_session").agg(
        (pl.col("event_type") == "purchase").any().cast(pl.Int8).alias("purchased")
    )


def save_features(df: pl.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"Saved {len(df):,} sessions → {path}")


if __name__ == "__main__":
    import duckdb

    from sessioniq.pipeline.loader import load_all

    print("Loading data...")
    train_lf, _ = load_all()

    print("Building train features (first 3 events only)...")
    train_features = build_session_features(train_lf, max_events=3).collect(
        engine="streaming"
    )
    train_labels = build_labels(train_lf).collect(engine="streaming")
    train_df = train_features.join(train_labels, on="user_session", how="inner")
    print(train_df.describe())
    save_features(train_df, "train_sessions")
    print("\nClass balance (train):")
    print(train_df["purchased"].value_counts())
    del train_df, train_features, train_labels, train_lf

    print("\nBuilding test features with DuckDB (first 3 events only)...")
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=2")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET temp_directory='data/tmp'")

    features_result = con.execute("""
        WITH parsed AS (
            SELECT user_session, user_id,
                CAST(event_time AS TIMESTAMP) AS event_time,
                event_type, product_id, category_code, brand,
                CAST(price AS FLOAT) AS price
            FROM read_csv('data/raw/2019-Nov.csv', header=true, ignore_errors=true)
            WHERE user_session IS NOT NULL
              AND user_id IS NOT NULL
              AND product_id IS NOT NULL
        ),
        session_starts AS (
            SELECT user_session, MIN(event_time) AS session_start
            FROM parsed
            GROUP BY user_session
        ),
        with_offset AS (
            SELECT p.*,
                EPOCH(p.event_time - s.session_start) AS seconds_from_start
            FROM parsed p
            JOIN session_starts s ON p.user_session = s.user_session
        ),
        early_events AS (
            SELECT user_session
            FROM with_offset
            WHERE seconds_from_start <= 300
            GROUP BY user_session
            HAVING COUNT(*) <= 3
        )
        SELECT
            w.user_session,
            FIRST(w.user_id) AS user_id,
            COUNT(*) AS n_events_observed,
            SUM(w.event_type = 'view') AS n_views,
            SUM(w.event_type = 'cart') AS n_carts,
            SUM(w.event_type = 'remove_from_cart') AS n_removals,
            COUNT(DISTINCT w.product_id) AS n_unique_products,
            COUNT(DISTINCT w.category_code) AS n_unique_categories,
            COUNT(DISTINCT w.brand) AS n_unique_brands,
            AVG(w.price) AS avg_price,
            MAX(w.price) AS max_price,
            EPOCH(MAX(w.event_time) - MIN(w.event_time))
                AS session_duration_seconds,
            SUM(w.event_type = 'cart') / (SUM(w.event_type = 'view') + 1.0)
                AS cart_view_ratio,
            EPOCH(
                MIN(CASE WHEN w.event_type = 'cart' THEN w.event_time END)
                - MIN(w.event_time)
            ) AS seconds_to_first_cart
        FROM with_offset w
        JOIN early_events e ON w.user_session = e.user_session
        WHERE w.seconds_from_start <= 300
        GROUP BY w.user_session
    """).arrow()

    test_features = pl.from_arrow(features_result).fill_null(0)

    labels_result = con.execute("""
        SELECT
            user_session,
            MAX(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchased
        FROM read_csv('data/raw/2019-Nov.csv', header=true, ignore_errors=true)
        WHERE user_session IS NOT NULL
        GROUP BY user_session
    """).arrow()

    test_labels = pl.from_arrow(labels_result)
    test_features = test_features.join(test_labels, on="user_session", how="inner")

    print(test_features.describe())
    save_features(test_features, "test_sessions")
    print("\nClass balance (test):")
    print(test_features["purchased"].value_counts())
