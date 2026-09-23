"""
Unit and smoke tests for the resale-price pipeline.

Run with: pytest tests/ -v
"""

import sys
import os

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config_validator import ConfigError, validate_config
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining


@pytest.fixture
def sample_config():
    return {
        "file_path": __file__,  # any real file, just needs to exist
        "numerical_features": ["floor_area_sqm", "remaining_lease_months", "lease_commence_date", "year"],
        "nominal_features": ["month", "town_name", "flatm_name"],
        "ordinal_features": ["flat_type"],
        "flat_type_categories": ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "MULTI-GENERATION", "EXECUTIVE"],
        "passthrough_features": ["storey_range"],
        "target_column": "resale_price",
        "test_size": 0.2,
        "random_state": 42,
        "val_size": 0.5,
        "param_grid": {"regressor__alpha": [0.1, 1]},
        "cv": 2,
        "scoring": "r2",
        "n_jobs": 1,
    }


@pytest.fixture
def raw_dataframe():
    return pd.DataFrame(
        {
            "id": range(20),
            "town_id": [1] * 20,
            "town_name": ["Bedok"] * 20,
            "flatm_id": [1] * 20,
            "flatm_name": ["Model A"] * 20,
            "flat_type": ["4 ROOM"] * 10 + ["FOUR ROOM"] * 10,
            "storey_range": ["07 TO 09"] * 20,
            "floor_area_sqm": np.random.uniform(60, 120, 20),
            "lease_commence_date": [-1990] * 10 + [1990] * 10,
            "month": ["2020-01"] * 20,
            "remaining_lease": ["70 years 3 months"] * 20,
            "block": ["123"] * 20,
            "street_name": ["Main St"] * 20,
            "resale_price": np.random.uniform(300000, 600000, 20),
        }
    )


class TestConfigValidation:
    def test_valid_config_passes(self, sample_config):
        validate_config(sample_config)  # should not raise

    def test_missing_key_raises(self, sample_config):
        del sample_config["target_column"]
        with pytest.raises(ConfigError):
            validate_config(sample_config)

    def test_test_size_as_list_raises(self, sample_config):
        sample_config["test_size"] = [0.2]
        with pytest.raises(ConfigError):
            validate_config(sample_config)

    def test_out_of_range_test_size_raises(self, sample_config):
        sample_config["test_size"] = 1.5
        with pytest.raises(ConfigError):
            validate_config(sample_config)


class TestDataPreparation:
    def test_convert_storey_range(self, sample_config):
        prep = DataPreparation(sample_config)
        assert prep._convert_storey_range("07 TO 09") == 8.0
        assert prep._convert_storey_range("01 TO 03") == 2.0

    def test_extract_lease_info_years_and_months(self, sample_config):
        prep = DataPreparation(sample_config)
        assert prep._extract_lease_info("70 years 3 months") == 843
        assert prep._extract_lease_info("85 years") == 1020
        assert prep._extract_lease_info("67") == 804
        assert prep._extract_lease_info(np.nan) is None

    def test_fill_missing_names(self, sample_config):
        prep = DataPreparation(sample_config)
        df = pd.DataFrame({"id": [1, 2, 3], "name": ["A", None, "A"]})
        filled = prep._fill_missing_names(df, id_column="id", name_column="name")
        # id=2 has no other row mapping it to a name, so it stays missing
        assert filled.loc[filled["id"] == 1, "name"].iloc[0] == "A"

    def test_clean_data_end_to_end(self, sample_config, raw_dataframe):
        prep = DataPreparation(sample_config)
        cleaned = prep.clean_data(raw_dataframe.copy())
        # flat_type standardised
        assert set(cleaned["flat_type"].unique()) == {"4 ROOM"}
        # negative lease dates fixed
        assert (cleaned["lease_commence_date"] > 0).all()
        # id columns dropped
        for col in ["id", "town_id", "flatm_id", "remaining_lease", "block", "street_name"]:
            assert col not in cleaned.columns
        # engineered columns present
        assert "remaining_lease_months" in cleaned.columns
        assert "year" in cleaned.columns

    def test_preprocessor_is_column_transformer(self, sample_config):
        prep = DataPreparation(sample_config)
        assert hasattr(prep.preprocessor, "fit_transform")


class TestModelTraining:
    def test_split_data_return_order(self, sample_config, raw_dataframe):
        prep = DataPreparation(sample_config)
        cleaned = prep.clean_data(raw_dataframe.copy())
        trainer = ModelTraining(sample_config, prep.preprocessor)

        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned)

        # The features (X_*) must be DataFrames; the targets (y_*) must be Series.
        # This is exactly the check that would have caught the original
        # X/y swap bug immediately.
        for X in (X_train, X_val, X_test):
            assert isinstance(X, pd.DataFrame)
            assert "resale_price" not in X.columns
        for y in (y_train, y_val, y_test):
            assert isinstance(y, pd.Series)
            assert y.name == "resale_price"

        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(cleaned)

    def test_smoke_end_to_end_training(self, sample_config, raw_dataframe):
        """A tiny end-to-end run: clean -> split -> fit -> predict, on a small sample."""
        # Need enough rows and variety for OneHotEncoder/train_test_split to behave;
        # duplicate the sample with slight noise.
        df = pd.concat([raw_dataframe] * 5, ignore_index=True)
        df["floor_area_sqm"] += np.random.normal(0, 1, len(df))
        df["resale_price"] += np.random.normal(0, 1000, len(df))

        prep = DataPreparation(sample_config)
        cleaned = prep.clean_data(df.copy())
        trainer = ModelTraining(sample_config, prep.preprocessor)
        X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data(cleaned)

        pipelines, metrics = trainer.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
        assert "linear_regression" in pipelines
        assert "R2" in metrics["linear_regression"]

        preds = pipelines["linear_regression"].predict(X_test)
        assert len(preds) == len(X_test)