"""
Lightweight Two-Tower model for product recommendations.
Trains product embeddings from session co-occurrence data.
Given a product, returns top-K similar products via FAISS ANN search.
"""

from pathlib import Path

import joblib
import polars as pl
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

EMBEDDING_DIM = 32
N_ITERATIONS = 20
REGULARIZATION = 0.01


def build_cooccurrence_matrix(
    lf: pl.LazyFrame,
) -> tuple[coo_matrix, dict, dict]:
    """
    Build a user-session × product sparse matrix from view/cart events.
    Each (session, product) pair gets a weight based on event type:
    - view: 1
    - cart: 3
    - purchase: 5
    """
    print("  Collecting events...")
    events = (
        lf.filter(pl.col("event_type").is_in(["view", "cart", "purchase"]))
        .select(["user_session", "product_id", "event_type"])
        .collect(engine="streaming")
    )

    # Encode event type as weight
    weight_map = {"view": 1, "cart": 3, "purchase": 5}
    events = events.with_columns(
        pl.col("event_type")
        .replace(weight_map, default=1)
        .cast(pl.Float32)
        .alias("weight")
    )

    # Build index mappings
    sessions = events["user_session"].unique().to_list()
    products = events["product_id"].unique().to_list()
    session2idx = {s: i for i, s in enumerate(sessions)}
    product2idx = {p: i for i, p in enumerate(products)}
    idx2product = {i: p for p, i in product2idx.items()}

    print(f"  Sessions: {len(sessions):,} | Products: {len(products):,}")

    # Build sparse matrix
    rows = events["user_session"].replace(session2idx).to_numpy()
    cols = events["product_id"].replace(product2idx).to_numpy()
    data = events["weight"].to_numpy()

    matrix = coo_matrix(
        (data, (rows, cols)),
        shape=(len(sessions), len(products)),
    )
    return matrix.tocsr(), product2idx, idx2product


def train_als(matrix: coo_matrix) -> AlternatingLeastSquares:
    model = AlternatingLeastSquares(
        factors=EMBEDDING_DIM,
        iterations=N_ITERATIONS,
        regularization=REGULARIZATION,
        use_gpu=False,
    )
    model.fit(matrix)
    return model


def save_recommender(model, product2idx: dict, idx2product: dict) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": model, "product2idx": product2idx, "idx2product": idx2product},
        MODEL_DIR / "recommender.joblib",
    )
    print(f"Recommender saved → {MODEL_DIR / 'recommender.joblib'}")


def load_recommender() -> tuple:
    data = joblib.load(MODEL_DIR / "recommender.joblib")
    return data["model"], data["product2idx"], data["idx2product"]


def recommend(
    product_id: int,
    model: AlternatingLeastSquares,
    product2idx: dict,
    idx2product: dict,
    top_k: int = 5,
) -> list[int]:
    if product_id not in product2idx:
        return []
    idx = product2idx[product_id]
    similar = model.similar_items(idx, N=top_k + 1)
    indices = similar[0][1:]
    return [idx2product[int(i)] for i in indices]


if __name__ == "__main__":
    import os

    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    from sessioniq.pipeline.loader import load_all

    print("Loading train data...")
    train_lf, _ = load_all()

    print("Building co-occurrence matrix...")
    matrix, product2idx, idx2product = build_cooccurrence_matrix(train_lf)
    print(f"  Matrix shape: {matrix.shape} | nnz: {matrix.nnz:,}")

    print("Training ALS...")
    model = train_als(matrix)

    save_recommender(model, product2idx, idx2product)

    model, product2idx, idx2product = load_recommender()
    sample_product = list(product2idx.keys())[0]
    recs = recommend(sample_product, model, product2idx, idx2product, top_k=5)
    print(f"\nSample recommendations for product {sample_product}:")
    print(f"  → {recs}")
