# Amex Default Prediction: Baseline Model

I created a baseline Logistic Regression model to predict credit card defaults using the dataset from the "American Express - Default Prediction" Kaggle challenge (https://www.kaggle.com/competitions/amex-default-prediction).

## Overview

- **Definition of the tasks / challenge:** The task, as defined by the Kaggle challenge, is to use a massive industrial-scale dataset containing anonymized customer profile information and time-series behavioral data to predict the probability that a customer will default on their credit card balance.
- **Your approach:** The approach in this repository formulates the problem as a binary classification task. Due to the massive size of the dataset and significant outliers, I focused on preprocessing (handling missing values, IQR-based scaling, one-hot encoding) and utilized a Logistic Regression model as a baseline.
- **Summary of the performance achieved:** The baseline model successfully processes the data and generates predictions using Kaggle's evaluation metric. 

## Summary of Workdone

### Data

- **Data:**
  - **Type:** CSV files (`train_data.csv`, `train_labels.csv`, `test_data.csv`) containing anonymized features (D=Delinquency, S=Spend, P=Payment, B=Balance, R=Risk).
  - **Size:** Extremely large (>50GB uncompressed), containing ~5.5 million rows and 190 predictive features.
  - **Instances (Train, Test, Validation Split):** For local prototyping to avoid Out-Of-Memory (OOM) errors, a 10,000-row sample was used. The data was split into 90% for training/validation (further split 80/20) and a 10% holdout test set.

#### Preprocessing / Clean up

- **Missing Data:** Dropped any feature missing more than 75% of its data. Imputed remaining numerical missing values with the median (to resist outliers) and categorical missing values with an `'Unknown'` string class.
- **Scaling:** Applied `RobustScaler` from scikit-learn to all numerical features to normalize the data while mitigating the effect of extreme outliers.
- **Encoding:** Converted categorical variables into binary columns using One-Hot Encoding (`get_dummies` with `drop_first=True` to prevent multicollinearity).

#### Data Visualization

- Generated proportional cross-tabulation tables for categorical variables to identify categories with high default rates.
- Created normalized density histograms (`stat='density'`) for numerical features, layering the Class 0 (Paid) and Class 1 (Default) distributions to visually identify features with high class separation and differing skews.

### Problem Formulation

- **Define:**
  - **Input / Output:**
    - Input is a heavily preprocessed array of numerical and one-hot encoded categorical features.
    - Output is a probability float representing the likelihood of default (Class 1).
  - **Models:** Logistic Regression was chosen as the primary baseline model because it is computationally efficient, handles scaled numerical data well, and provides interpretable probabilities required for the custom metric.
  - **Loss, Optimizer, other Hyperparameters:** Trained using the default 'lbfgs' solver, with `max_iter=1000` to ensure mathematical convergence, and `class_weight='balanced'` to account for the heavy class imbalance (74% Paid vs 26% Default).

### Training

- **Describe the training:**
  - **How you trained:** Model was trained locally using Python, scikit-learn, and Jupyter Notebooks.
  - **Difficulties & Resolutions:** The primary difficulty was Out-Of-Memory (OOM) crashes due to the 50GB dataset size. This was resolved by implementing a local sampling strategy (`nrows=10000` in Pandas) to build the pipeline, ensuring the code worked before scaling up. Furthermore, API authentication issues with `kagglehub` were bypassed by locally downloading the data or strictly defining `os.environ` keys.

### Performance Comparison

- **Metrics:** Evaluated using standard Accuracy, ROC-AUC, and the official custom Amex Kaggle Metric (0.5 * (Gini + Top 4% Recall)).
- Results were evaluated on the validation split, outputting the standard classification metrics alongside the custom competition score (where 1.0 is perfect and 0.0 is worst).

### Conclusions

- Logistic Regression provides a solid, non-trivial baseline for this dataset, provided the data is rigorously scaled and cleaned.
- The dataset contains heavy class imbalance and significant outliers, making `RobustScaler` and median imputation much more effective than standard mean-based scaling.

### Future Work

- **Algorithm Upgrades:** Transition from Logistic Regression to gradient boosting frameworks like XGBoost or LightGBM, which natively handle missing `NaN` values and do not require extensive rescaling.
- **Data Engineering:** Transition the data loading process from CSV to `.parquet` format or use libraries like Dask/Polars to process the full 50GB dataset locally without hitting RAM limits.
- **Time-Series Utilization:** Aggregate the 13 months of behavioral data per customer using summary statistics (mean, max, last) or sequence models (LSTMs) rather than treating every row independently.

## How to reproduce results

- To reproduce these results, you must have an active Kaggle account, agree to the competition rules, and download the data locally. You can then run the provided Jupyter notebook pipeline top-to-bottom.

### Overview of files in repository

- **`Amex_Baseline_Pipeline.ipynb`**: The master Jupyter Notebook containing the entire end-to-end process. It includes data loading, the feature summary generator, outlier detection, data visualization, preprocessing, model training, and submission file generation.
- **`submission.csv`**: The output file generated by the notebook, formatted correctly for submission to the Kaggle leaderboard.

### Software Setup

- **Required Packages:** `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `kagglehub`.
- **Installation:** Install requirements via pip:
  `pip install pandas numpy matplotlib seaborn scikit-learn kagglehub`

### Data

- **Download Location:** Data can be downloaded directly from the [Amex Default Prediction Data Page](https://www.kaggle.com/competitions/amex-default-prediction/data).
- Ensure you click **"I Understand and Accept"** on the competition rules page prior to attempting any API downloads. Unzip the downloaded files into the same directory as your Jupyter notebook.

### Training

- Open `Amex_Baseline_Pipeline.ipynb` and run the cells sequentially. The notebook handles data cleaning, model fitting, and metric evaluation automatically. 


