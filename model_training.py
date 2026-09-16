# Standard library imports
import logging
from typing import Any, Dict, Tuple

# Related third-party imports
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline




class ModelTraning:
    """
    Class used to train and evaluate machine learning models on HDB resale price

     Attributes:
     - config : Dict[str, Any]
       Configuration dictionary containing parameters for model training and evaluation.
     - preprocessor : sklearn.compose.ColumnTransformer
       A preprocessor pipeline for transforming numerical, nominal, and ordinal features.
    
    """

    def ___init___(self, config: Dict[str, Any]):
            """
            initilises the DataPreparation class with configuiration dictionary
    
            Args:
            config (Dict[str, Any]): COnfiguratiuon dic containing paramters
            """
            self.config = config
            self.preprocessor = self._create_preprocessor()


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

            logging.info("Data splitting completed")
            return  X_train, y_train,  X_val, X_test, y_val, y_test

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
            "linear_regression": LinearRegression(),
            "ridge": Ridge(),
            "lasso": Lasso(),
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
        metrics = {
            "MAE": mean_absolute_error(y_test, y_test_pred),
            "MSE": mean_squared_error(y_test, y_test_pred),
            "RMSE": root_mean_squared_error(y_test, y_test_pred),
            "R²": r2_score(y_test, y_test_pred),
        }
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
        metrics = {
            "MAE": mean_absolute_error(y_val, y_val_pred),
            "MSE": mean_squared_error(y_val, y_val_pred),
            "RMSE": root_mean_squared_error(y_val, y_val_pred),  # RMSE is the square root of MSE
            "R2":  r2_score(y_val, y_val_pred),
        }
        logging.info(f"{model_name} Validation Metrics: ")
        for metrics_name, metrics_value in metrics.items():
             logging.info(f"{metrics_name}: {metrics_value}")
        return metrics








