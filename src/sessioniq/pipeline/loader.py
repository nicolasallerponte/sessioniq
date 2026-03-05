"""
Raw data loader using Polars lazy evaluation.
October → train split, November → test split.
"""

from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

SCHEMA = {
    "event_time": pl.Utf8,
    "event_type": pl.Categorical,
    "product_id": pl.Int64,
    "category_id": pl.Int64,
    "category_code": pl.Utf8,
    "brand": pl.Utf8,
    "price": pl.Float32,
    "user_id": pl.Int64,
    "user_session": pl.Utf8,
}


def load_month(filename: str) -> pl.LazyFrame:
    """Load a single month CSV as a LazyFrame."""
    path = RAW_DIR / filename
    return (
        pl.scan_csv(path, schema=SCHEMA)
        .with_columns(
            pl.col("event_time")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S UTC")
            .alias("event_time")
        )
        .drop_nulls(subset=["user_session", "user_id", "product_id"])
    )


def load_all() -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Returns (train, test) as LazyFrames.
    Train = October 2019, Test = November 2019.
    """
    train = load_month("2019-Oct.csv")
    test = load_month("2019-Nov.csv")
    return train, test


if __name__ == "__main__":
    train, test = load_all()
    print("Train schema:", train.schema)
    print("Train rows (sample):", train.fetch(5))
    print("Test rows (sample):", test.fetch(5))
