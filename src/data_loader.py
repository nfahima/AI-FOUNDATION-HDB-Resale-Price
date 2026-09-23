"""
Dedicated dataset loading and validation, split out of main.py so that
file errors and data-quality issues are caught early and logged clearly,
rather than surfacing later as confusing model-training errors.
"""

import logging
import os
from typing import Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataValidationError(ValueError):
    """Raised when the raw dataset fails a basic sanity check."""


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Loads the raw dataset from disk with basic error handling and logging.

    Args:
        file_path: Path to the CSV file.

    Returns:
        The loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        DataValidationError: If the file exists but is empty or unreadable.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found at {file_path}")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset at {file_path} is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"Dataset at {file_path} could not be parsed: {exc}") from exc

    logger.info("Loaded dataset with shape %s from %s", df.shape, file_path)
    logger.info("Dtypes:\n%s", df.dtypes)
    return df


def validate_raw_data(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Runs basic data-quality checks on the raw dataset before cleaning, logging
    a summary of any issues found. This does not mutate or drop data — it is
    a reporting/early-warning step.

    Args:
        df: The raw, unprocessed DataFrame.
        config: The pipeline configuration (used to locate the target column).

    Raises:
        DataValidationError: If the target column is missing, or if the
            target column contains non-positive values (which would be
            impossible for a resale price).
    """
    target_column = config["target_column"]

    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' not found in dataset.")

    n_duplicates = df.duplicated().sum()
    if n_duplicates:
        logger.warning("Found %d duplicate rows in raw data.", n_duplicates)

    missing_summary = df.isna().sum()
    missing_summary = missing_summary[missing_summary > 0]
    if not missing_summary.empty:
        logger.warning("Missing values by column:\n%s", missing_summary)

    if (df[target_column] <= 0).any():
        n_bad = (df[target_column] <= 0).sum()
        raise DataValidationError(
            f"Found {n_bad} rows where '{target_column}' is zero or negative."
        )

    high_cardinality = {
        col: df[col].nunique()
        for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 100
    }
    if high_cardinality:
        logger.info("High-cardinality categorical columns: %s", high_cardinality)