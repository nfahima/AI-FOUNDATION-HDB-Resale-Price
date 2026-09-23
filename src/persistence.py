"""
Persists pipeline outputs to disk: the fitted model, run metrics, and test
predictions. Without this, results only ever exist in the console log and
can't be reused or audited later.
"""

import json
import logging
import os
from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def save_model(model: Pipeline, output_dir: str, filename: str = "best_model.joblib") -> str:
    """
    Serialises the fitted pipeline to disk with joblib.

    Args:
        model: The fitted sklearn Pipeline to save.
        output_dir: Directory to save into (created if it doesn't exist).
        filename: Name of the output file.

    Returns:
        The full path the model was saved to.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    joblib.dump(model, path)
    logger.info("Saved fitted model to %s", path)
    return path


def save_metrics(all_metrics: Dict[str, Any], output_path: str) -> None:
    """
    Writes all models' metrics (baseline, tuned, and final test metrics) to a
    JSON file for later comparison.

    Args:
        all_metrics: Dictionary mapping model/stage name to its metrics dict.
        output_path: Path to write the JSON file to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=float)
    logger.info("Saved metrics to %s", output_path)


def save_predictions(
    X_test: pd.DataFrame, y_test: pd.Series, y_pred, output_path: str
) -> None:
    """
    Writes test-set predictions alongside ground truth to a CSV file.

    Args:
        X_test: Test features (used only for the row index).
        y_test: Ground-truth target values.
        y_pred: Predicted target values.
        output_path: Path to write the CSV file to.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    result_df = pd.DataFrame(
        {"actual": y_test.values, "predicted": y_pred}, index=X_test.index
    )
    result_df["residual"] = result_df["actual"] - result_df["predicted"]
    result_df.to_csv(output_path)
    logger.info("Saved predictions to %s", output_path)


def get_feature_importance(model: Pipeline, top_n: int = 15) -> pd.DataFrame:
    """
    Extracts a ranked feature-importance table from a fitted pipeline.
    Uses `coef_` for linear models and `feature_importances_` for
    tree-based models, whichever the final estimator provides.

    Args:
        model: A fitted Pipeline with steps named 'preprocessor' and 'regressor'.
        top_n: Number of top features to return.

    Returns:
        A DataFrame with columns ['feature', 'importance'], sorted by
        absolute importance, descending. Empty if the estimator exposes
        neither attribute.
    """
    regressor = model.named_steps["regressor"]
    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    if hasattr(regressor, "coef_"):
        importances = regressor.coef_
    elif hasattr(regressor, "feature_importances_"):
        importances = regressor.feature_importances_
    else:
        logger.warning("Model has neither coef_ nor feature_importances_; skipping.")
        return pd.DataFrame(columns=["feature", "importance"])

    importance_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_df["abs_importance"] = importance_df["importance"].abs()
    importance_df = importance_df.sort_values("abs_importance", ascending=False).drop(
        columns="abs_importance"
    )
    return importance_df.head(top_n).reset_index(drop=True)