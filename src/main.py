# Standard library imports
import logging

# Third-party imports
# import pandas as pd
import yaml
from sklearn.utils._testing import ignore_warnings

# Local application/library specific imports
from src.config_validator import validate_config
from src.data_loader import load_dataset, validate_raw_data
from src.data_preparation import DataPreparation
from src.model_training import ModelTraining
from src.persistence import get_feature_importance, save_metrics, save_model, save_predictions

logging.basicConfig(
    level=logging.INFO,
    format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@ignore_warnings(category=Warning)
def main():

    # Configuration file path
    config_path = "./src/config.yaml"

    # add try function so if the file cannot be opened the program does not crash
    try:

        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error("Config file not found at %s", config_path)
        raise
    except yaml.YAMLError as exc:
        logger.error("Could not parse config file: %s, exc")
        raise

    validate_config(config)

    # Load CSV file into a DataFrame
    df = load_dataset(config["file_path"])
    validate_raw_data(df, config)

    # Initialize and run data preparation
    data_prep = DataPreparation(config)
    cleaned_df = data_prep.clean_data(df)

    # Initialize model training with the created preprocessor
    model_training = ModelTraining(config, data_prep.preprocessor)

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = model_training.split_data(
        cleaned_df
    )

    # Train and evaluate baseline models with default hyperparameters
    baseline_models, baseline_metrics = (
        model_training.train_and_evaluate_baseline_models(
            X_train, y_train, X_val, y_val
        )
    )

    # Train and evaluate tuned models with hyperparameter tuning
    tuned_models, tuned_metrics = model_training.train_and_evaluate_tuned_models(
        X_train, y_train, X_val, y_val
    )

    # Combine all models and their metrics into dictionaries
    all_models = {**baseline_models, **tuned_models}
    all_metrics = {**baseline_metrics, **tuned_metrics}

    # Find the best model based on R2 score
    best_model_name = max(all_metrics, key=lambda k: all_metrics[k]["R2"])
    best_model = all_models[best_model_name]
    logging.info(f"Best Model Found: {best_model_name}")

    ## retrain the wining model before final 
    # test set evaluation 
    final_model = model_training.retrain_on_train_and_val(
        best_model, X_train, y_train, X_val, y_val
    )

    # Evaluate the best model on the test set
    final_metrics = model_training.evaluate_final_model(
        best_model, X_test, y_test, best_model_name
    )

    all_metrics[f"{best_model_name}_final_test"] = final_metrics

    #persist outouts 
    save_model(final_model, config.get("output_dir", "./models"))
    save_metrics(all_metrics, config.get("metrics_path", "./model/metrics.json"))
    y_test_pred = final_model.predict(X_test)
    save_predictions(
        X_test, y_test, y_test_pred, config.get("predictions_path", "./models/predictions.csv")
    )

    importance_df = get_feature_importance(final_model)
    if not importance_df.empty:
        logger.info("Top features for %s: \n%s", best_model_name, importance_df.to_string())


if __name__ == "__main__":
    main()