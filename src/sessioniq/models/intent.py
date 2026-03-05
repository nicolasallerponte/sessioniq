"""
Purchase intent classifier trained on session-level features.
Uses LightGBM with calibrated probabilities for reliable scores.
Trained on first 3 events of each session — no leakage.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")

FEATURE_COLS = [
    "n_events_observed",
    "n_views",
    "n_carts",
    "n_removals",
    "n_unique_products",
    "n_unique_categories",
    "n_unique_brands",
    "avg_price",
    "max_price",
    "session_duration_seconds",
    "cart_view_ratio",
    "seconds_to_first_cart",
]


def load_splits() -> tuple:
    train = pl.read_parquet(PROCESSED_DIR / "train_sessions.parquet")
    test = pl.read_parquet(PROCESSED_DIR / "test_sessions.parquet")

    X_train = train.select(FEATURE_COLS).to_pandas().astype(float)
    y_train = train["purchased"].to_numpy()
    X_test = test.select(FEATURE_COLS).to_pandas().astype(float)
    y_test = test["purchased"].to_numpy()

    return X_train, y_train, X_test, y_test


def build_model() -> CalibratedClassifierCV:
    # Load tuned params if available, else use sensible defaults
    params_path = MODEL_DIR / "best_params.joblib"
    if params_path.exists():
        params = joblib.load(params_path)
        print(f"Loaded tuned params from {params_path}")
    else:
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 100,
            "scale_pos_weight": 13,
            "n_jobs": -1,
            "verbose": -1,
        }

    base = lgb.LGBMClassifier(**params)
    return CalibratedClassifierCV(
        base,
        cv=StratifiedKFold(n_splits=3),
        method="isotonic",
    )


def evaluate(model, X_test, y_test) -> None:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    print("\nEvaluation")
    print(
        classification_report(y_test, y_pred, target_names=["no_purchase", "purchase"])
    )
    print(f"ROC-AUC:          {roc_auc_score(y_test, y_prob):.4f}")
    print(f"Avg Precision:    {average_precision_score(y_test, y_prob):.4f}")
    print("\n")


def save_model(model) -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    path = MODEL_DIR / "intent_classifier.joblib"
    joblib.dump(model, path)
    print(f"Model saved → {path}")


def load_model() -> CalibratedClassifierCV:
    return joblib.load(MODEL_DIR / "intent_classifier.joblib")


if __name__ == "__main__":
    print("Loading splits...")
    X_train, y_train, X_test, y_test = load_splits()
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Train conversion: {y_train.mean():.3%} | Test: {y_test.mean():.3%}")

    print("\nTraining calibrated LightGBM...")
    model = build_model()
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)
    save_model(model)
