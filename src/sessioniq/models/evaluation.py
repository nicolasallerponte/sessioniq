"""
Model evaluation utilities.
- Optimal threshold search by F1
- Precision-Recall curve
- SHAP feature importance
All outputs are saved to models/eval/
"""

from pathlib import Path

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.metrics import (
    PrecisionRecallDisplay,
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR = Path("models")
EVAL_DIR = Path("models/eval")

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


def find_optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Find the probability threshold that maximises F1 on purchase class."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    f1_scores = [
        f1_score(y_true, (y_prob >= t).astype(int), pos_label=1, zero_division=0)
        for t in thresholds
    ]
    best_threshold = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    print(f"Optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")
    return float(best_threshold)


def plot_precision_recall(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    PrecisionRecallDisplay.from_predictions(
        y_true, y_prob, name="LightGBM (calibrated)", ax=ax
    )
    ax.set_title("Precision-Recall Curve — Purchase Intent")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"PR curve saved → {save_path}")


def plot_shap(
    model: lgb.LGBMClassifier,
    X_sample: np.ndarray,
    save_path: Path,
) -> None:
    print("Computing SHAP values (sample of 5000)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # shap_values is a list [class0, class1] for binary classifiers
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=FEATURE_COLS,
        show=False,
        plot_size=None,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP plot saved → {save_path}")


def full_evaluation(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_lgbm: lgb.LGBMClassifier,
) -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics at default threshold
    print("\nEvaluation (threshold=0.5) ")
    print(
        classification_report(
            y_test,
            (y_prob >= 0.5).astype(int),
            target_names=["no_purchase", "purchase"],
        )
    )

    # Optimal threshold
    threshold = find_optimal_threshold(y_test, y_prob)
    print(f"\nEvaluation (threshold={threshold:.2f}) ")
    print(
        classification_report(
            y_test,
            (y_prob >= threshold).astype(int),
            target_names=["no_purchase", "purchase"],
        )
    )

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"ROC-AUC:       {roc_auc:.4f}")
    print(f"PR-AUC:        {pr_auc:.4f}")
    print("\n")

    # Plots
    plot_precision_recall(y_test, y_prob, EVAL_DIR / "precision_recall.png")

    # SHAP on 5000 sample
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(X_test), size=min(5000, len(X_test)), replace=False)
    plot_shap(base_lgbm, X_test.iloc[sample_idx], EVAL_DIR / "shap_summary.png")

    # Save threshold
    joblib.dump(threshold, MODEL_DIR / "optimal_threshold.joblib")

    return {"roc_auc": roc_auc, "pr_auc": pr_auc, "threshold": threshold}


if __name__ == "__main__":
    from sessioniq.models.intent import FEATURE_COLS, load_model, load_splits

    print("Loading test split...")
    _, _, X_test, y_test = load_splits()

    print("Loading model...")
    model = load_model()

    # Extract base LightGBM from calibrated wrapper for SHAP
    base_lgbm = model.calibrated_classifiers_[0].estimator

    full_evaluation(model, X_test, y_test, base_lgbm)
