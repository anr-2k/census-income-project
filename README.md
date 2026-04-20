# Census Income Analysis Project

A machine learning project built on 1994-1995 U.S. Census Bureau data to support income-based marketing decisions for a retail business client. The project delivers two models:

1. **Classification Model** — predicts whether an individual earns above or below $50,000 annually
2. **Segmentation Model** — identifies natural customer groups within the population for targeted marketing

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Results Summary](#results-summary)
- [Key Findings](#key-findings)

---

## Project Overview

### Task 1: Classification

Predicts whether a person earns above or below $50,000 per year using 40 demographic and employment variables. Multiple models were trained and evaluated. The best results:

- **Tuned LightGBM** — ROC-AUC 0.9374, Recall 0.86 (best for maximum reach)
- **Ensemble (LightGBM + MLP) with threshold tuning** — ROC-AUC 0.9359, Precision 0.61, F1 0.62 (best for balanced marketing efficiency)

### Task 2: Segmentation

Identifies natural customer segments within the population. Two segmentations were produced:

- **Full adult population** — 5 segments for broad campaign planning
- **High earners only** — 2 segments for premium product targeting

---

## Dataset

The dataset is based on the 1994-1995 U.S. Census Bureau Current Population Surveys.

| File | Description |
|---|---|
| `census-bureau.data` | Main dataset — 199,523 records, 42 columns |
| `census-bureau.columns` | Column names file |

> **Note:** The data files are not included in this repository due to size constraints. The dataset is available on Kaggle: [census-income-project](https://www.kaggle.com/datasets/nikhilram9/census-income-project). Download and place the files in the same directory as the notebooks before running, or add directly as a Kaggle dataset input.

---

## Project Structure

```
census-income-project/
│
├── census-income-classification.ipynb    # EDA, preprocessing, and classification model
├── census-income-segmentation.ipynb      # Segmentation model
├── project_report.docx                   # Full project report with charts
└── README.md                             # This file
```

---

## Setup & Installation

### Requirements

Python 3.8 or higher. Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn category_encoders xgboost lightgbm catboost
```

### Platform Notes

| Platform | Notes |
|---|---|
| **Kaggle** | All libraries pre-installed. Upload data files as a dataset and add via **Add Input**. |
| **Google Colab** | Run `!pip install category_encoders catboost` before starting. |
| **Local / Jupyter** | Run the pip install command above before launching the notebooks. |

---

## How to Run

### Step 1 — Get the data

Download `census-bureau.data` and `census-bureau.columns` from the [Kaggle dataset](https://www.kaggle.com/datasets/nikhilram9/census-income-project).

If running on **Kaggle**: upload both files as a dataset and add to the notebook via the **Add Input** button on the right sidebar. Update the file paths in the first cell accordingly.

### Step 2 — Run the Classification Notebook

Open `census-income-classification.ipynb` and run all cells top to bottom.

**Sections covered:**
1. EDA — data shape, distributions, class imbalance, visualizations
2. Data Cleaning — remove minors, drop irrelevant columns, handle missing values
3. Train/Test Split — 80/20 stratified
4. Preprocessing — target encoding, standard scaling, column type corrections
5. Model Building — Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost, Neural Network
6. Hyperparameter Tuning — RandomizedSearchCV on XGBoost, LightGBM, and MLP
7. Feature Selection — SelectFromModel on MLP pipeline
8. Ensemble — soft voting: tuned LightGBM + tuned MLP
9. Threshold Tuning — optimal threshold from precision-recall curve
10. Final Model Comparison — all models compared on Recall, Precision, F1, ROC-AUC

**Expected runtime:** 20-40 minutes (hyperparameter tuning is the slowest step)

### Step 3 — Run the Segmentation Notebook

Open `census-income-segmentation.ipynb` and run all cells top to bottom.

**Sections covered:**
1. Data Cleaning — same steps as classification
2. Dataset Preparation — full population and high earners split
3. Encoding & Scaling — target encoding + StandardScaler
4. PCA — 18 components for full population, 19 for high earners (both at 85% variance)
5. Cluster Selection — elbow method + silhouette score for k=2 to k=10
6. Clustering — K-Means: k=5 for full population, k=2 for high earners
7. Cluster Profiling — mean/mode statistics per cluster
8. Visualization — PCA scatter plots and feature comparison charts

**Expected runtime:** 10-20 minutes

### File Path Note

Default paths in both notebooks point to a Kaggle dataset location. If running on a different platform, update the paths in the first code cell to point to where your data files are stored.

---

## Results Summary

### Classification

| Model | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| **LightGBM (tuned) ★** | **0.86** | **0.37** | **0.52** | **0.9374** |
| XGBoost (tuned) | 0.83 | 0.39 | 0.53 | 0.9372 |
| CatBoost | 0.86 | 0.36 | 0.50 | 0.9336 |
| **Ensemble + Threshold ★** | **0.64** | **0.61** | **0.62** | **0.9359** |
| Logistic Regression | 0.86 | 0.33 | 0.47 | 0.9239 |
| Neural Network (tuned) | 0.41 | 0.73 | 0.53 | 0.9269 |
| Random Forest | 0.40 | 0.74 | 0.52 | 0.9233 |

★ Recommended. All metrics for the high-income class (label = 1).

### Segmentation — Full Population (5 Clusters)

| Segment | Size | Avg Age | Avg Weeks Worked | Avg Wage/Hr |
|---|---|---|---|---|
| Working Professionals | 76,899 (53.6%) | 40 | 48 | $123 |
| Retired/Inactive Homemakers | 44,335 (30.9%) | 56 | 5 | $2 |
| Part-time Workers | 10,867 (7.6%) | 38 | 28 | $44 |
| Mid-level Workers | 9,839 (6.9%) | 35 | 34 | $72 |
| Wealthy Inactive/Investors | 1,591 (1.1%) | 59 | 20 | $48 |

### Segmentation — High Earners (2 Clusters)

| Segment | Size | Avg Age | Avg Weeks Worked | Avg Dividends |
|---|---|---|---|---|
| High Earning Professionals | 11,474 (92.7%) | 45 | 51 | $1,151 |
| Wealthy Investors & Retirees | 906 (7.3%) | 63 | 14 | $6,657 |

---

## Key Findings

- **Weeks worked in year** is the single strongest income predictor — above education and occupation. Employment consistency matters more than job title or credentials.
- **Occupation type** predicts income more reliably than industry sector. Executive, managerial, and professional roles are strongly associated with high income regardless of sector.
- **Education** is a near-prerequisite for high income but not a guarantee. Almost all high earners hold at least a Bachelor's degree.
- **Investment income** (dividends, capital gains) is a strong signal of existing wealth and likely high income.
- **Two genuinely different high earner profiles** exist: active working professionals (93%) and wealthy investors/retirees living off passive income (7%). These groups require completely different marketing approaches despite being in the same income bracket.
- The **class imbalance** (91.4% vs 8.6%) required explicit handling in every model — without it, models default to the majority class and become useless for the client's purpose.
