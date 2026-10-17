# PhishGuard ML

A supervised machine-learning system that classifies websites as **phishing** or **legitimate** using 30 pre-extracted URL and page characteristics.

Built as part of a college ML Lab project (Semester 5).

## Models Used

| Algorithm | Scaling | Notes |
|---|---|---|
| Decision Tree | No | Gini vs Entropy, depth tuning |
| K-Nearest Neighbours | Yes (StandardScaler) | k = 3, 5, 7, 9, 11 |
| Support Vector Machine | Yes (StandardScaler) | Linear and RBF kernels |
| Categorical Naive Bayes | No (features shifted to 0,1,2) | Suited for discrete features |

## Dataset

**UCI Phishing Websites Dataset** — 11,055 records, 30 integer-valued features, binary target.

## Project Structure

```
PhishGuard-ML/
├── 01_eda.ipynb              # Data loading, inspection, EDA plots
├── 02_preprocessing.ipynb    # Train/test split, feature selection
├── 03_model_training.ipynb   # Train 4 models with GridSearchCV
├── 04_evaluation.ipynb       # Metrics, confusion matrices, ROC curves
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── phishing_websites.csv
│   ├── X_train.csv, X_test.csv
│   ├── y_train.csv, y_test.csv
│   └── feature_importance.csv
└── models/
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── svm.pkl
    └── categorical_naive_bayes.pkl
```

## Setup

```bash
pip install -r requirements.txt
```

Run notebooks in order: `01` → `02` → `03` → `04`

## Streamlit Dashboard

```bash
streamlit run app.py
```

Pages: Overview, Model Comparison, Predict, Risk Analysis.

## License

Academic project — UCI Phishing Websites Dataset.
