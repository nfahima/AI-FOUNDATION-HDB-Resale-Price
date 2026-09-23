# HDB Resale Price Prediction 

This project builds a machine learning pipeline to predict HDB resale prices using information such as floor area, flat type, town, flat model, remaining lease and storey range.

The project started as exploratory Jupyter notebook and was then split into smaller Python files so the workflow is easier to read, update and reuse.

The main models are Linear Regression, Ridge Regression and Lasso Regression. Ridge and Lasso are also tuned using Grid Search Cross-Validation.

The target variable is:

```text
resale_price
```

The main project files are:

```text
main.py
data_preparation.py
model_training.py
config.yaml
requirements.txt
eda.ipynb
data/
└── resale_transactions.csv
```

## Prerequisites and Installation

The project uses Python as its main programming langugaes and the needed packages are listed within the file `requirement.txt`

```text
matplotlib==3.11.2
numpy==2.4.6
pandas==3.0.5
pip==26.2.1
PyYAML==6.0.3
scikit-learn==1.9.1
seaborn==0.13.2
``` 

To install the dependencies:

```bash
pip install -r requirements.txt
```

The HDB dataset should be placed at:

```text
./data/resale_transactions.csv
```

This path can also be changed in `config.yaml`.


---

## Instructions for Executing the Pipeline

The entry point for this project is 
```bash
python main.py
```

The machine learning should use the `main.py` first which loads the `config.yaml`, reads the resale transaction data, cleans the data using the function `DataPreparation` from the `data_preparation.py` file and then trains and evaluat ethe models using class `ModelTraning` from the `model_traning.py` file. Before running the pipeline first check the values within the `config.yaml`. The impportant settings are:

| Setting | Purpose |
|---|---|
| `file_path` | Location of the resale transaction CSV file |
| `numerical_features` | Numerical columns that are standardised |
| `nominal_features` | Categorical columns that are one-hot encoded |
| `ordinal_features` | Ordered categorical columns |
| `flat_type_categories` | Order used when encoding flat types |
| `passthrough_features` | Features kept without scaling or encoding |
| `target_column` | Column the model predicts |
| `test_size` | Portion kept aside from the training set |
| `val_size` | Splits the held-out data into validation and test sets |
| `random_state` | Keeps the split reproducible |
| `param_grid` | Ridge and Lasso hyperparameters tested by Grid Search |
| `cv` | Number of cross-validation folds |
| `scoring` | Metric used during Grid Search |
| `n_jobs` | Number of CPU jobs used by scikit-learn |

For this pipline the intended data split is 80% traning, 10% validation, 10% test split within the config these should represent them:
```yaml
test_size: 0.2
val_size: 0.5
random_state: 42
```

---

## Description of Logical Steps/Flow of the Pipeline
The overall flow is:

```text
resale_transactions.csv
        |
        v
Load configuration and dataset
        |
        v
Data cleaning
        |
        +--> Remove duplicate rows
        +--> Fix "FOUR ROOM" to "4 ROOM"
        +--> Convert negative lease commencement years to positive
        +--> Fill missing town and flat model names using their IDs
        |
        v
Feature engineering
        |
        +--> Convert storey ranges to a numerical midpoint
        +--> Extract year and month from the original month field
        +--> Convert remaining lease to total months
        +--> Remove ID, block, street name and old helper columns
        |
        v
Model preprocessing
        |
        +--> StandardScaler for numerical features
        +--> OneHotEncoder for nominal features
        +--> OrdinalEncoder for flat type
        +--> Pass storey range through as a numerical feature
        |
        v
80% Train / 10% Validation / 10% Test
        |
        v
Baseline models
        |
        +--> Linear Regression
        +--> Ridge Regression
        +--> Lasso Regression
        |
        v
Grid Search CV
        |
        +--> Tune Ridge
        +--> Tune Lasso
        |
        v
Validation comparison
        |
        v
Best Ridge model
        |
        v
Final evaluation on untouched test data
```

Important aspect of the design that is scaling and encoding are placed within scikit-learn `Pipeline`. Meaning the preprocessed is fitted using traning data and then applied consistently to the validation and test data.


---

## Overview of Key Findings from EDA
The full Exploratory Data Analysis can be found in the file  `eda.ipynb`. Main finding which influenced the pipeline are here. 

The dataset contained a few data quality problems that needed to be fixed before modelling. There were duplicate rows, negative values in `lease_commence_date`, missing values in `town_name` and `flatm_name`, and an inconsistent flat type label where both `4 ROOM` and `FOUR ROOM` appeared.

The flat type distribution was also quite uneven. **4 ROOM flats were the most common**, followed by 3 ROOM and 5 ROOM flats, while 1 ROOM and MULTI-GENERATION flats appeared much less often.

Resale prices were right-skewed, with most transactions grouped in the lower-to-middle price range and a smaller number of expensive flats stretching the distribution to the right. The notebook recorded prices from about **$160,000 to $1,200,000**, with a median of around **$408,000**.

The relationship between floor area and resale price was one of the clearest patterns. The Pearson correlation between `floor_area_sqm` and `resale_price` was around **0.64**, showing that larger flats generally sold for higher prices. The scatter plot also showed that flats with similar floor areas could still have very different resale prices, which suggests that location, age, flat type and other features also matter.

The EDA also found a moderate positive relationship between `lease_commence_date` and resale price. Flats with more recent lease commencement dates generally had higher resale prices. `storey_range` showed a weaker positive relationship with resale price, so it was still kept as an ordered numerical feature.

Larger flat types generally had higher median resale prices and wider price ranges. Some unusual 3-room outliers were investigated and found to include HDB terrace houses, showing that not every apparent outlier is necessarily bad data.

These findings guided the cleaning, feature engineering and model choices used in the final pipeline.

## Feature Handling Description

| Feature | Type / Handling | What is done |
|---|---|---|
| `resale_price` | Target | Value the regression models try to predict |
| `floor_area_sqm` | Numerical | Kept and standardised using `StandardScaler` |
| `lease_commence_date` | Numerical | Negative values are converted to positive values, then standardised |
| `remaining_lease` | Feature engineered | Converted from values such as `70 years 03 months` into total months |
| `remaining_lease_months` | Numerical | Newly created feature and standardised |
| `month` | Feature engineered + nominal | Original `YYYY-MM` value is converted to datetime, then the month number is extracted and one-hot encoded |
| `year` | Numerical | Extracted from the original month field and standardised |
| `town_name` | Nominal | Missing values are filled using `town_id`, then the feature is one-hot encoded |
| `flatm_name` | Nominal | Missing values are filled using `flatm_id`, then the feature is one-hot encoded |
| `flat_type` | Ordinal | `FOUR ROOM` is changed to `4 ROOM`, then categories are ordinal encoded |
| `storey_range` | Ordinal / numerical | A range such as `07 TO 09` is converted to its midpoint, for example `8.0` |
| `id` | Removed | Identifier does not help with prediction |
| `town_id` | Removed | Used to fill `town_name`, then removed |
| `flatm_id` | Removed | Used to fill `flatm_name`, then removed |
| `remaining_lease` | Removed | Removed after `remaining_lease_months` is created |
| `block` | Removed | Not used in the final model |
| `street_name` | Removed | Not used in the final model |

The final preprocessing setup uses:

```text
Numerical:
floor_area_sqm
remaining_lease_months
lease_commence_date
year

Nominal:
month
town_name
flatm_name

Ordinal:
flat_type

Passthrough:
storey_range
```

---


### Linear Regression

Linear Regression is used as the main baseline model. It is a useful starting point because the target, `resale_price`, is continuous and the model makes it easy to understand how the different features contribute to the prediction.

It also gives a simple benchmark. Ridge and Lasso should ideally match or improve on this baseline.

### Ridge Regression

Ridge Regression adds L2 regularisation to Linear Regression. This is useful here because one-hot encoding produces many features and some of them may be related to each other.

Ridge reduces very large coefficients without completely removing features. This can make the model more stable and help reduce overfitting.

### Lasso Regression

Lasso Regression uses L1 regularisation. Unlike Ridge, Lasso can reduce some coefficients all the way to zero, which also gives it a feature-selection effect.

It was included so its performance and coefficient behaviour could be compared with Linear Regression and Ridge Regression.

### Hyperparameter Tuning

Grid Search Cross-Validation is used to tune Ridge and Lasso.

The current search tests:

```yaml
regressor__alpha: [0.1, 1, 10, 100, 1000]
regressor__fit_intercept: [true, false]
```

with:

```yaml
cv: 5
scoring: r2
```

This makes it possible to compare several regularisation strengths instead of relying only on the default settings.

---

## Evaluation of Models

The models are evaluated using four regression metrics:

| Metric | Meaning |
|---|---|
| MAE | Average absolute difference between the predicted and actual resale price |
| MSE | Average squared prediction error, so larger mistakes are penalised more |
| RMSE | Square root of MSE and is easier to relate back to resale price values |
| R² | Measures how much of the variation in resale prices is explained by the model |

The validation results recorded in the EDA notebook were:

| Model | MAE | MSE | RMSE | R² |
|---|---:|---:|---:|---:|
| Tuned Ridge (`alpha=0.1`) | 42,374.50 | 3,027,107,697.36 | 55,019.16 | 0.8585 |
| Ridge (`alpha=1`) | 42,376.01 | 3,027,468,676.67 | 55,022.44 | 0.8585 |
| Tuned Lasso (`alpha=0.1`) | 42,377.25 | 3,028,324,908.90 | 55,030.22 | 0.8585 |
| Lasso (`alpha=1`) | 42,376.35 | 3,028,079,185.18 | 55,027.99 | 0.8585 |
| Linear Regression | 42,376.66 | 3,028,318,921.18 | 55,030.16 | 0.8585 |

The differences between the models were small, but tuned Ridge produced the lowest validation errors in the notebook. Grid Search selected:

```text
alpha = 0.1
fit_intercept = True
```

for both Ridge and Lasso.

The tuned Ridge model was therefore taken forward for the final test evaluation.

Final Ridge test results:

| Metric | Final Test Result |
|---|---:|
| MAE | 41,272.73 |
| MSE | 2,896,767,079.25 |
| RMSE | 53,821.62 |
| R² | 0.8656 |

The test results were slightly better than the validation results. This suggests that the selected Ridge model generalised reasonably well to the unseen test data in this dataset.

---

## Considerations for Deployment

The current project is a good starting point for turning the notebook work into a repeatable pipeline, but there are still a few things to think about before using it as a real deployed prediction service.

The model should be saved together with its preprocessing pipeline. This is important because new data has to go through the exact same scaling and encoding steps that were used during training.

New input data also needs to follow the same column names and formats as the training dataset. For example, `storey_range` needs to follow the expected range format and the lease information needs to be in a format that the conversion function can understand.

The one-hot encoder is already set to ignore unseen nominal categories, which helps with new towns, months or flat models. The ordinal encoder for `flat_type`, however, currently expects known categories, so a completely new flat type would need to be handled before production use.

Performance should also be monitored over time because the HDB resale market can change. Prices, market conditions and the importance of different locations may shift, so the model may eventually need to be retrained using newer transactions.

For a larger deployment, it would also be useful to add input validation, model versioning, logging, automated testing and a clear way to save and reload the final trained pipeline.

