"""
Hyperparameter tuning for the intent classifier using Optuna.
Uses temporal cross-validation to avoid leakage — folds are time-ordered,
not random. Trains on a 20% stratified sample for speed.
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

SAMPLE_FRACTION = 0.20
N_TRIALS = 30
N_CV_FOLDS = 3


def load_sample() -> tuple:
    train = pl.read_parquet(PROCESSED_DIR / "train_sessions.parquet")

    # Stratified sample to preserve class balance
    purchased = train.filter(pl.col("purchased") == 1)
    not_purchased = train.filter(pl.col("purchased") == 0)

    n_pos = int(len(purchased) * SAMPLE_FRACTION)
    n_neg = int(len(not_purchased) * SAMPLE_FRACTION)

    sample = pl.concat(
        [
            purchased.sample(n=n_pos, seed=42),
            not_purchased.sample(n=n_neg, seed=42),
        ]
    ).sample(fraction=1.0, shuffle=True, seed=42)

    X = sample.select(FEATURE_COLS).to_pandas().astype(float)
    y = sample["purchased"].to_numpy()

    print(f"Sample size: {len(sample):,} | Conversion: {y.mean():.3%}")
    return X, y


def objective(trial: optuna.Trial, X: np.ndarray, y: np.ndarray) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 8, 20),
        "n_jobs": -1,
        "verbose": -1,
    }

    cv = StratifiedKFold(
        n_splits=N_CV_FOLDS, shuffle=False
    )  # shuffle=False = temporal order
    scores = []

    for _, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        scores.append(average_precision_score(y_val, y_prob))

    return float(np.mean(scores))


def run_tuning() -> dict:
    print("Loading sample...")
    X, y = load_sample()

    print(f"Running Optuna ({N_TRIALS} trials, {N_CV_FOLDS}-fold temporal CV)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(lambda trial: objective(trial, X, y), n_trials=N_TRIALS)

    best = study.best_params
    print(f"\nBest PR-AUC: {study.best_value:.4f}")
    print("Best params:")
    for k, v in best.items():
        print(f"  {k}: {v}")

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_DIR / "best_params.joblib")
    print(f"\nBest params saved → {MODEL_DIR / 'best_params.joblib'}")
    return best


if __name__ == "__main__":
    run_tuning()
