# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ModelTraining:
    """
    Class used to train and evaluate machine learning models on HDB resale price

     Attributes:
     - config : Dict[str, Any]
       Configuration dictionary containing parameters for model training and evaluation.
     - preprocessor : sklearn.compose.ColumnTransformer
       A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    
    """

    def __init__(self, config: Dict[str, Any], preprocessor: ColumnTransformer):
            """
            initilises the DataPreparation class with configuiration dictionary
    
            Args:
            config (Dict[str, Any]): COnfiguratiuon dic containing paramters
            """
            self.config = config
            self.preprocessor = preprocessor


    def split_data(
        self, df: pd.DataFrame) -> Tuple[
                                    pd.DataFrame,
                                    pd.DataFrame, 
                                    pd.DataFrame,
                                    pd.Series, 
                                    pd.Series, 
                                    pd.Series]:
            """
            split the data into traning, validation and test sets 
            
            Args:
            df(pd.DataFrame): the input DataFrame containing cleaned dataset
            
            Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: 
            A tuple containing the training, validation, and test features and target variables.
            """

            logging.info("Starting data splitting")

            # Separate the data into features and target
            X = df.drop(columns= self.config["target_column"])
            y = df[self.config["target_column"]]

            # Split the data into training (80%) and test-validation (20%) sets
            X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size= self.config["test_size"], random_state=self.config["random_state"])

            # Split the test-validation set (20%) into validation (10%) and test (10%) sets
            X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=self.config["val_size"], random_state=self.config["random_state"])

            logger.info(
                "Split sizes -> train: %s, val: %s, test: %s",
                X_train.shape,
                X_val.shape,
                X_test.shape,
            )
            return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate_baseline_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Create, train, and evaluate baseline models.

        Args:
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the trained pipelines and their evaluation metrics.
        """
        logging.info("Training and evaluating baseline models.")
        models = {
            "dummy_mean": DummyRegressor(strategy="mean"),
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
            "random_forest": RandomForestRegressor(
                 random_state=self.config["random_state"], n_jobs=self.config["n_jobs"]),
            "gradient_boosting": GradientBoostingRegressor(
                 random_state=self.config["random_state"]
            ),
        }
        pipelines = {}
        metrics = {}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )
            pipeline.fit(X_train, y_train)
            pipelines[model_name] = pipeline
            metrics[model_name] = self._evaluate_model(
                pipeline, X_val, y_val, model_name
            )

        return pipelines, metrics
            
    def train_and_evaluate_tuned_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]:
        """
        Perform hyperparameter tuning for Ridge and Lasso models and evaluate them.

        Args:
        -----
        X_train (pd.DataFrame): The training features.
        y_train (pd.Series): The training target variable.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.

        Returns:
        --------
        Tuple[Dict[str, Pipeline], Dict[str, Dict[str, float]]]: A tuple containing the tuned pipelines and their evaluation metrics.
        """
        logging.info("Starting hyperparameter tuning.")
        tuned_models = {}
        tuned_metrics = {}
        param_grid = self.config["param_grid"]
        cv = self.config["cv"]
        scoring = self.config["scoring"]
        n_jobs = self.config["n_jobs"]

        models = {"ridge_tuned": Ridge(), "lasso_tuned": Lasso()}

        for model_name, model in models.items():
            pipeline = Pipeline(
                steps=[("preprocessor", self.preprocessor), ("regressor", model)]
            )
            grid_search = GridSearchCV(
                pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs
            )
            grid_search.fit(X_train, y_train)
            tuned_models[model_name] = grid_search.best_estimator_
            tuned_metrics[model_name] = self._evaluate_model(
                tuned_models[model_name], X_val, y_val, model_name + " (tuned)"
            )

        logging.info("Hyperparameter tuning completed.")
        return tuned_models, tuned_metrics

    def retrain_on_train_and_val(self, model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
        ) ->Pipeline:
        """
        Refits the chosen model's pipeline on the combined train+validation
            data before final test evaluation, so the final model benefits from
            all data not held out for testing.
    
            Args:
                model: The selected fitted pipeline (its hyperparameters are reused).
                X_train: Training features.
                y_train: Training target.
                X_val: Validation features.
                y_val: Validation target.
    
            Returns:
                A new pipeline, refit on train+val, with the same hyperparameters.
        """
        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)
 
        refit_model = Pipeline(steps=[step for step in model.steps])
        refit_model.fit(X_combined, y_combined)
        logger.info("Retrained final model on combined train+val (%d rows).", len(X_combined))
        return refit_model

    
    def evaluate_final_model(
        self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate the final model on the test set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_test (pd.DataFrame): The test features.
        y_test (pd.Series): The test target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """
        y_test_pred = model.predict(X_test)
        metrics = self._compute_metrics(y_test, y_test_pred)
        logging.info(f"Final Test Metrics for {model_name}:")
        for metric_name, metric_value in metrics.items():
            logging.info(f"{metric_name}: {metric_value}")
        return metrics

    def _evaluate_model(
        self, model: Pipeline, X_val: pd.DataFrame, y_val: pd.Series, model_name: str
    ) -> Dict[str, float]:
        """
        Evaluate a model on the validation set and log the metrics.

        Args:
        -----
        model (Pipeline): The trained model pipeline.
        X_val (pd.DataFrame): The validation features.
        y_val (pd.Series): The validation target variable.
        model_name (str): The name of the model being evaluated.

        Returns:
        --------
        Dict[str, float]: A dictionary containing the evaluation metrics.
        """

        y_val_pred = model.predict(X_val)
        metrics = self._compute_metrics(y_val, y_val_pred)
        logging.info(f"{model_name} Validation Metrics: ")
        for metrics_name, metrics_value in metrics.items():
             logging.info(f"{metrics_name}: {metrics_value}")
        return metrics

    @staticmethod
    def _compute_metrics(y_true: pd.Series, y_pred) -> Dict[str, float]:
         """
         Computes the standard set of regression metrics.
 
        NOTE: the metric key is consistently "R2" (ASCII) everywhere in this
        module and in main.py's best-model selection — the original code
        mixed "R2" here with a "R²" lookup elsewhere, which would have
        raised a KeyError.
 
        Args:
            y_true: Ground-truth target values.
            y_pred: Predicted target values.
 
        Returns:
            A dictionary with keys MAE, MSE, RMSE, R2
         """

         return{
            "MAE": mean_absolute_error(y_true, y_pred),
            "MSE": mean_squared_error(y_true, y_pred),
            "RMSE": root_mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred),
        }







