# Feature Discretization with KBinsDiscretizer

Binning continuous features (Age, Fare) from the Titanic dataset using sklearn's KBinsDiscretizer and measuring the impact on DecisionTree accuracy.

## What This Project Does

- Loads Titanic data (Age, Fare, Survived columns only)
- Splits into train/test (80/20)
- Baseline DecisionTree accuracy: **61.5%** (test), **62.6%** (10-fold CV)
- Applies KBinsDiscretizer (15 bins, quantile strategy) via ColumnTransformer
- After discretization accuracy: **63.6%** (test), **69.5%** (10-fold CV)
- Compares kmeans vs quantile strategies across 10 bins
- Visualizes before/after distributions for Age and Fare

## Key Results

| Setup | Test Accuracy | CV Accuracy (10-fold) |
|---|---|---|
| Raw features | 61.5% | 62.6% |
| KBins (15, quantile) | 63.6% | 69.5% |
| KBins (10, kmeans) | — | 63.3% |
| KBins (10, quantile) | — | 63.3% |

## Stack

Python · pandas · NumPy · scikit-learn · matplotlib

## How to Run

```bash
pip install pandas numpy scikit-learn matplotlib
jupyter notebook feature_discretization.ipynb
```

Place `train.csv` (Titanic dataset from Kaggle) in the same directory.

## Key Concepts

- **KBinsDiscretizer** — converts continuous features into ordinal bins
- **ColumnTransformer** — applies different transformers to different columns
- **Quantile strategy** — equal-frequency bins, handles skewed data better
- **KMeans strategy** — bins based on cluster centers
- Transformers fit on train split only — zero data leakage
