# Book Crossing ML Pipeline

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-interactive-blueviolet?logo=plotly)](https://plotly.com/)
[![Jupyter](https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

An end-to-end machine learning pipeline built on the **Book Crossing dataset** — 278,859 users, 271,378 books, and 1,149,753 ratings. Three complete ML tasks unified under a single shared data pipeline:

- **Task 1 — Recommender System:** Collaborative filtering with cosine similarity
- **Task 2 — User Clustering:** KMeans, Hierarchical, and DBSCAN clustering
- **Task 3 — Age Estimation:** Regression models to predict missing user ages

---

## Pipeline Architecture

```
Raw CSVs (Users, Books, Ratings)
         │
         ▼
┌─────────────────────┐
│   Module 0          │
│   Shared Data       │  BookCrossingLoader + BookCrossingPreprocessor
│   Pipeline          │  → Cleans, validates, exports libsvm + CSVs
└────────┬────────────┘
         │
    ┌────┴─────────────────────────┐
    │              │               │
    ▼              ▼               ▼
┌────────┐   ┌─────────┐   ┌────────────┐
│Task 1  │   │Task 2   │   │Task 3      │
│Recomm- │   │Cluster- │   │Age         │
│ender   │   │ing      │   │Estimation  │
└────────┘   └─────────┘   └────────────┘
    │              │               │
    ▼              ▼               ▼
Book_recom-  clustering_   Task3_Estimated
mendation    results.txt   _User_Ages.csv
.csv
```

---

## Dataset

**Book Crossing Dataset** — [Kaggle](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset)

| File | Description | Size |
|---|---|---|
| `Users.csv` | 278,859 users with optional age | 2.4 MB |
| `Books.csv` | 271,379 books with title, author, year | 23 MB |
| `Ratings.csv` | 1,149,780 ratings (scale 0–10) | 22 MB |

> **Note:** Raw data files are not included in this repository due to size. Download from Kaggle and place in `data/raw/`.

---

## Exploratory Data Analysis

| Metric | Value |
|---|---|
| Total users | 278,859 |
| Users with known age | 165,917 (59.5%) |
| Users with missing age | 112,942 (40.5%) |
| Total books | 271,378 |
| Total ratings | 1,149,753 |
| Explicit ratings (1–10) | 433,662 (37.7%) |
| Implicit ratings (0) | 716,091 (62.3%) |
| Matrix sparsity | 99.9988% |
| Mean user age | 34.7 years |
| Median user age | 32.0 years |
| Age range | 5 – 100 years |
| Median ratings per user | 1.0 |
| Max ratings by one user | 13,599 |
| Users with only 1 rating | 59,166 |
| Peak publication year | 2002 (17,628 books) |

---

## Results

### Task 1 — Recommender System

Collaborative filtering using cosine similarity on a **105,283 × 340,054** sparse user-book matrix.

| Metric | Value |
|---|---|
| Similarity metric | Cosine similarity |
| Neighbors (K) | 10 |
| Users with similar users found | 64,053 |
| Users with recommendations | 61,020 |
| Total recommendation rows | 275,057 |
| Recommendations per user | Top 5 |

- Output: `outputs/Book_recommendation.csv`

### Task 2 — User Clustering

Dimensionality reduction: **TruncatedSVD (50 components)** — explains **11.85%** of variance.

**KMeans Results**

| k | Inertia | Silhouette Score |
|---|---|---|
| 2 | 67,180.42 | 0.1825 |
| 4 | 61,476.77 | 0.1946 |
| 6 | 58,588.90 | 0.2109 |
| 8 | 56,752.60 | 0.2108 |
| 10 | 54,972.10 | 0.2366 |
| 20 | 48,647.90 | 0.2534 |
| **30** | **43,477.96** | **0.2913 ✅ Best** |

**Hierarchical Clustering (Ward linkage) Results**

| k | Silhouette Score |
|---|---|
| 2 | 0.1849 |
| 4 | 0.1765 |
| 6 | 0.1926 |
| 8 | 0.2008 |
| 10 | 0.2092 |
| 20 | 0.2449 |
| **30** | **0.2705 ✅ Best** |

**DBSCAN Results**

| eps | min_samples | Clusters | Noise Points | Silhouette |
|---|---|---|---|---|
| **0.3** | **5** | **99** | **5,611** | **0.9258 ✅ Best** |
| 0.5 | 5 | 150 | 4,304 | 0.7621 |
| 0.5 | 10 | 64 | 5,023 | 0.8205 |

**KMeans k=10 — Cluster Profiles**

| Cluster | Users | Top Book |
|---|---|---|
| 0 | 44,882 | The Lovely Bones: A Novel |
| 1 | 2,007 | The Da Vinci Code |
| 2 | 9,313 | The Catcher in the Rye |
| 3 | 7,482 | A Painted House |
| 4 | 8,337 | The Red Tent (Bestselling Backlist) |
| 5 | 6,128 | Harry Potter and the Sorcerer's Stone |
| 6 | 6,129 | Free |
| 7 | 3,238 | The Nanny Diaries: A Novel |
| 8 | 6,202 | Wild Animus |
| 9 | 11,565 | Divine Secrets of the Ya-Ya Sisterhood |

- Output: `outputs/clustering_results.txt`

### Task 3 — Age Estimation

Dimensionality reduction: **TruncatedSVD (50 components)** — explains **11.85%** of variance.  
Training set: **61,354 users** with known age · Prediction set: **43,929 users** with missing age.

**Model Comparison (5-Fold Cross Validation)**

| Model | RMSE |
|---|---|
| Linear Regression | 19.2207 |
| Logistic Regression (binned) | 16.8767 |
| **Decision Tree Regressor** | **14.3196 ✅ Best** |

**Decision Tree — Residuals Analysis**

| Metric | Value |
|---|---|
| Mean residual | -0.04 years |
| Std of residuals | 14.32 years |
| Predictions within 5 years | 26.39% |
| Predictions within 10 years | 50.42% |

**Predictions**

| Metric | Value |
|---|---|
| Users with predicted ages | 43,929 |
| Predicted age range | 13 – 69 years |
| Mean predicted age | 35.9 years |

- Output: `outputs/Task3_Estimated_User_Ages.csv`

---

## Project Structure

```
bookcrossing-ml-pipeline/
├── src/
│   └── data/
│       ├── __init__.py
│       ├── loader.py              ← loads raw CSVs
│       └── preprocessor.py        ← cleans and exports data
├── notebooks/
│   ├── 00_EDA.ipynb               ← exploratory data analysis
│   ├── 02_recommender.ipynb       ← Task 1: recommender system
│   ├── 03_clustering.ipynb        ← Task 2: user clustering
│   └── 04_age_estimation.ipynb    ← Task 3: age estimation
├── data/
│   ├── raw/                       ← place CSVs here (gitignored)
│   └── processed/                 ← auto-generated outputs
├── outputs/
│   ├── Book_recommendation.csv
│   ├── clustering_results.txt
│   └── Task3_Estimated_User_Ages.csv
├── assets/                        ← saved chart images
├── run_module0.py                 ← runs full data pipeline
└── requirements.txt
```

---

## Quick Start

**1. Clone the repository:**
```bash
git clone https://github.com/sohailalij/bookcrossing-ml-pipeline.git
cd bookcrossing-ml-pipeline
```

**2. Create and activate environment:**
```bash
conda create -n bookcrossing python=3.12 -y
conda activate bookcrossing
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/somnambwl/bookcrossing-dataset) and place the three CSV files in `data/raw/`.

**5. Run the data pipeline:**
```bash
python run_module0.py
```

**6. Open Jupyter and run the notebooks in order:**
```bash
jupyter notebook
```

---

## Tech Stack

| Category | Libraries |
|---|---|
| Data processing | pandas, numpy, scipy |
| Machine learning | scikit-learn |
| Sparse matrices | scipy.sparse |
| Visualization | matplotlib, seaborn, plotly |
| Environment | Anaconda, Jupyter Notebook |

---

## Key Challenges

- **99.9988% matrix sparsity** — handled via sparse CSR matrix and TruncatedSVD
- **112,942 missing user ages** — estimated using Decision Tree regression (RMSE 14.32)
- **ISBN format inconsistencies** — zero-padded to 10 characters in shared preprocessor
- **Scale** — 1.1 million ratings processed through a single unified pipeline
- **62.3% implicit ratings (0)** — separated from explicit ratings throughout analysis

---

## License

This project is licensed under the MIT License.
