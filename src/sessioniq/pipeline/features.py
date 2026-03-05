"""
Session-level feature engineering.
Groups raw events into sessions and computes predictive features.
"""

from pathlib import Path

import polars as pl

PROCESSED_DIR = Path("data/processed")


def build_session_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Aggregate raw events into one row per session with predictive features.

    Features:
    - n_events: total clicks in session
    - n_views, n_carts, n_purchases, n_removals: counts by event type
    - cart_view_ratio: carts / views (intent signal)
    - n_unique_products: product diversity
    - n_unique_categories: category diversity
    - category_entropy: how spread is the user across categories
    - avg_price, max_price: price range explored
    - session_duration_seconds: time from first to last event
    - seconds_to_first_cart: how fast user adds to cart (null if never)
    - purchased: label (1 if session has a purchase, 0 otherwise)
    """
    return (
        lf.sort("event_time")
        .group_by("user_session")
        .agg(
            [
                # Label
                (pl.col("event_type") == "purchase")
                .any()
                .cast(pl.Int8)
                .alias("purchased"),
                # Volume
                pl.len().alias("n_events"),
                (pl.col("event_type") == "view").sum().alias("n_views"),
                (pl.col("event_type") == "cart").sum().alias("n_carts"),
                (pl.col("event_type") == "purchase").sum().alias("n_purchases"),
                (pl.col("event_type") == "remove_from_cart").sum().alias("n_removals"),
                # Diversity
                pl.col("product_id").n_unique().alias("n_unique_products"),
                pl.col("category_code").n_unique().alias("n_unique_categories"),
                pl.col("brand").n_unique().alias("n_unique_brands"),
                # Price
                pl.col("price").mean().alias("avg_price"),
                pl.col("price").max().alias("max_price"),
                # Temporal
                (pl.col("event_time").max() - pl.col("event_time").min())
                .dt.total_seconds()
                .alias("session_duration_seconds"),
                # Time to first cart
                pl.col("event_time")
                .filter(pl.col("event_type") == "cart")
                .min()
                .alias("first_cart_time"),
                pl.col("event_time").min().alias("session_start"),
                # User id (keep for joins)
                pl.col("user_id").first().alias("user_id"),
            ]
        )
        .with_columns(
            [
                # cart/view ratio — avoid division by zero
                (pl.col("n_carts") / (pl.col("n_views") + 1)).alias("cart_view_ratio"),
                # seconds from session start to first cart
                (pl.col("first_cart_time") - pl.col("session_start"))
                .dt.total_seconds()
                .alias("seconds_to_first_cart"),
            ]
        )
        .drop(["first_cart_time", "session_start"])
        .fill_null(0)
    )


def save_features(df: pl.DataFrame, name: str) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{name}.parquet"
    df.write_parquet(path)
    print(f"Saved {len(df):,} sessions → {path}")


if __name__ == "__main__":
    import duckdb

    from sessioniq.pipeline.loader import load_all

    # TRAIN
    print("Building train features...")
    train_lf, _ = load_all()
    train_features = build_session_features(train_lf).collect(engine="streaming")
    print(train_features.describe())
    save_features(train_features, "train_sessions")
    print("\nClass balance (train):")
    print(train_features["purchased"].value_counts())
    del train_features, train_lf

    # TEST
    print("\nBuilding test features with DuckDB...")
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    con.execute("SET temp_directory='data/tmp'")

    result = con.execute("""
        WITH parsed AS (
            SELECT
                user_session,
                user_id,
                CAST(event_time AS TIMESTAMP) AS event_time,
                event_type,
                product_id,
                category_code,
                brand,
                CAST(price AS FLOAT) AS price
            FROM read_csv(
                'data/raw/2019-Nov.csv',
                header = true,
                ignore_errors = true
            )
            WHERE user_session IS NOT NULL
              AND user_id IS NOT NULL
              AND product_id IS NOT NULL
        )
        SELECT
            user_session,
            FIRST(user_id)                                         AS user_id,
            MAX(event_type = 'purchase')::INT                      AS purchased,
            COUNT(*)                                               AS n_events,
            SUM(event_type = 'view')                               AS n_views,
            SUM(event_type = 'cart')                               AS n_carts,
            SUM(event_type = 'purchase')                           AS n_purchases,
            SUM(event_type = 'remove_from_cart')                   AS n_removals,
            COUNT(DISTINCT product_id)                             AS n_unique_products,
            COUNT(DISTINCT category_code)                          AS n_unique_categories,
            COUNT(DISTINCT brand)                                  AS n_unique_brands,
            AVG(price)                                             AS avg_price,
            MAX(price)                                             AS max_price,
            EPOCH(MAX(event_time) - MIN(event_time))               AS session_duration_seconds,
            SUM(event_type = 'cart') / (SUM(event_type = 'view') + 1.0) AS cart_view_ratio,
            EPOCH(
                MIN(CASE WHEN event_type = 'cart' THEN event_time END) - MIN(event_time)
            )                                                      AS seconds_to_first_cart
        FROM parsed
        GROUP BY user_session
    """).arrow()

    test_features = pl.from_arrow(result)
    test_features = test_features.fill_null(0)
    print(test_features.describe())
    save_features(test_features, "test_sessions")
    print("\nClass balance (test):")
    print(test_features["purchased"].value_counts())
