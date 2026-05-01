###Amex Default Prediction Challenge###
One Sentence Summary
This repository holds an attempt to predict customer credit default probabilities using a Random Forest model on massive, chunked Parquet datasets from the Amex Default Prediction Kaggle challenge.

Overview
Definition of the tasks / challenge
The task, as defined by the Kaggle challenge, is to predict the probability that a customer will default on their credit card balance. The dataset contains a time series of 188 anonymized customer profile features, sampled at monthly intervals, to predict the binary default target (0 or 1).

Your approach
The approach in this repository formulates the problem as a binary classification task, using a Random Forest Classifier. Because the original dataset is massive (~11GB CSV), our pipeline uses PyArrow to stream heavily compressed Parquet files in batches of 250,000 rows. Inside the stream, we use fast in-place vectorized imputation, one-hot encoding, and float32 downcasting to run Random Forest predictions at blazing speeds without maxing out RAM.

Summary of the performance achieved
The final model efficiently outputs mean grouped predictions per customer ID, handling hundreds of thousands of rows in mere seconds, resulting in a robust submission file for Kaggle.

Summary of Workdone
Data
Type: Tabular data (originally CSV, converted to Parquet for speed and memory efficiency).

Input: Customer profile features (188 features: numerical and categorical), customer_ID, and date (S_2). Output is a binary target.

Size: ~11GB of raw CSV data (optimized to much smaller footprints using integer-dtype Parquet format).

Instances: 458,913 unique customer data points in the training set labels.

Preprocessing / Clean up
To make the data manageable and ready for modeling, the following manipulations were performed:

Feature Dropping: Dropped non-predictive columns (customer_ID, S_2).

In-Place Imputation: Filled missing numerical features with 0 and missing categorical features with the statistical mode (calculated once prior to chunking to save time).

Data Downcasting: Casted all numerical columns to float32 to drastically speed up scaling and model inference.

Encoding: Applied fast one-hot encoding for categorical variables (cat_cols) and reindexed the columns to perfectly match the training schema.

Scaling: Applied StandardScaler to all numerical columns.

Data Visualization
Data visualization was carried out on a 10,000-row sample to avoid Out-Of-Memory (OOM) errors:

Analyzed target variables to find a heavy class imbalance: ~74.1% Non-Default (0) vs ~25.9% Default (1).

Generated seaborn countplots for categorical features and density histograms for numerical features. These visualizations helped identify features where the density peaks for Class 0 and Class 1 were far apart, indicating high predictive power.

Problem Formulation
Input / Output: Input is a customer's monthly profile; Output is a continuous probability [0, 1] indicating the likelihood of a future default.

Models: A Random Forest model (rf_model) was used because of its robustness to outliers and ability to capture non-linear relationships among 188 features.

Hyperparameters: Default probability is extracted specifically from the positive class tree votes (predict_proba()[:, 1]). Predictions are then grouped by customer_ID and averaged.

Training
How you trained: The pipeline was developed and trained in Google Colab, utilizing the kagglehub library to pull datasets securely and Colab's RAM to cache batches.

Difficulties & Resolutions: The original .csv reading approach combined with pd.get_dummies in a loop took 20+ minutes and choked the RAM. This was resolved by swapping to an open-source Amex Parquet dataset (raddar/amex-data-integer-dtypes-parquet-format), reading the file iteratively via pyarrow.parquet.ParquetFile.iter_batches(), and strictly utilizing in-place operations with copy=False.

Conclusions
Working with 11GB+ tabular datasets requires moving away from traditional pandas.read_csv text parsing. Using Parquet formats and PyArrow batches combined with pre-computed dictionaries and float32 downcasting bridges the gap between memory safety (avoiding OOM) and maximum execution speed.

Future Work
Algorithm Swapping: The next step is to substitute the Random Forest for a gradient-boosting algorithm like XGBoost, LightGBM, or CatBoost to potentially capture better metrics.

Feature Engineering: Calculate moving averages, lags, or standard deviations over the monthly time series data for each customer_ID before aggregating.

How to reproduce results
To reproduce these results or use this package to perform your own study:

Open Google Colab (Recommended hardware: Standard CPU/RAM or High-RAM if doing massive feature engineering).

Grab your Kaggle API key (kaggle.json).

Set your Colab Secrets to include KAGGLE_USERNAME and KAGGLE_KEY.

Run the data fetching and chunked inference scripts provided in the repository.

Overview of files in repository
Kaggle.ipynb: The main Jupyter Notebook containing exploratory data analysis (EDA), Kaggle API authentication, feature distribution plotting, and the baseline data sampling.

inference.py / inference.ipynb: Contains the optimized PyArrow streaming loop. Loads the Parquet test data in 250,000-row chunks, imputes, encodes, scales, runs Random Forest predictions, and generates submission.csv.

Software Setup
Required Packages:

pandas, numpy, matplotlib, seaborn

scikit-learn

pyarrow (Crucial for fast Parquet batching)

kagglehub

Installation:
Standard libraries are pre-installed on Google Colab. If running locally, you can install the required packages via:

Bash
pip install pandas numpy scikit-learn pyarrow matplotlib seaborn kagglehub
Data Access:
Data is pulled automatically via the kagglehub API in the script:

Official: kagglehub.competition_download('amex-default-prediction')

Parquet (Optimized): kagglehub.dataset_download("raddar/amex-data-integer-dtypes-parquet-format")
