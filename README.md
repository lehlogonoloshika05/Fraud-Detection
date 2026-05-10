Credit Card Fraud Detection: Exploratory Data Analysis
A Python-based EDA on a real-world credit card fraud dataset. This project analyses 284,807 transactions to understand data structure, distributions, class imbalance, outliers, and feature correlations. Every finding connects directly to a preprocessing decision for machine learning.

Dataset

Source: Kaggle Credit Card Fraud Detection
Rows: 284,807 transactions
Columns: 31 (Time, V1-V28 PCA features, Amount, Class)
Target: Class (0 = not fraud, 1 = fraud)


Tools

Python 3
Pandas
NumPy
Matplotlib
Seaborn


Project Structure
credit-card-fraud-eda/
├── notebook.ipynb       # Full EDA notebook with code, outputs, and findings
├── README.md
└── requirements.txt

Analysis Steps
Step 1: Load and Inspect

284,807 rows, 31 columns
All columns numeric. No type conversion needed.
V1-V28 are PCA-anonymized features from the original dataset.

Step 2: Missing Values and Duplicates

Zero missing values across all 31 columns.
Found and removed 1,081 duplicate rows.
Amount ranges from 0 to 25,691. Heavily right-skewed.

Step 3: Distributions

Raw Amount: extreme right skew. Unusable without transformation.
Log-transformed Amount: far more balanced. Confirms log1p as a preprocessing step.
Time: clear two-day pattern. Two distinct transaction peaks visible.

Step 4: Class Imbalance

Fraud: 492 transactions (0.17%)
Not Fraud: 284,315 transactions (99.83%)
Ratio: 578:1
A model predicting "not fraud" every time scores 99.83% accuracy. That metric is useless here. AUC-ROC and F1-score are the correct evaluation metrics.
Severe imbalance justifies applying SMOTE before model training.

Step 5: Outlier Detection (IQR Method)

IQR upper bound for Amount: 185.38
Outliers found: 31,685 rows (11.17% of the dataset)
Of those: 87 are fraud cases.
Decision: do not remove outliers. Removing them deletes fraud signal. Apply log transformation to Amount instead.

Step 6: Feature Correlations

Strongest negative correlation with fraud: V17 (-0.31), V14 (-0.29), V12 (-0.25)
Strongest positive correlation with fraud: V11 (+0.15), V4 (+0.13)
Amount: near-zero correlation (0.006). Raw Amount alone is not a strong predictor.
V1-V28 show near-zero inter-feature correlation. No redundant features. No columns need to be dropped.


Key Preprocessing Decisions Identified
FeatureActionReasonAmountApply log1p transformationExtreme right skew, max 25,691AmountDo not remove outliers87 fraud cases sit in outlier zoneTimeScale before modellingRange 0-172,792V1-V28Leave as-isAlready PCA-scaledDuplicatesRemove 1,081 rowsExact copies skew trainingClass imbalanceApply SMOTE578:1 ratio, only 492 fraud cases

How to Run

Clone the repo

bashgit clone https://github.com/your-username/credit-card-fraud-eda.git
cd credit-card-fraud-eda

Install dependencies

bashpip install -r requirements.txt

Download the dataset from Kaggle and place creditcard.csv in the project folder.
Open the notebook

bashjupyter notebook notebook.ipynb

Requirements
pandas
numpy
matplotlib
seaborn
jupyter

About
Built as a portfolio project demonstrating practical EDA skills on a real imbalanced fraud detection dataset. Covers data quality checks, statistical analysis, visualisation, and preprocessing decision-making.
