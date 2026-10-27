<div align="center">

# ⚙️ Maintena Pro
### Industrial Predictive Maintenance & Diagnostics Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)


<p align="center">
  <b>Real-time equipment failure prediction, physics-informed condition monitoring, and telemetry diagnostics powered by Machine Learning.</b>
</p>

[Key Features](#-key-features) • [Installation](#-quick-start) • [Machine Learning Architecture](#-machine-learning-architecture) • [Dataset](#-dataset-information) • [Dashboard Overview](#-dashboard-walkthrough)

---

</div>

## 📌 Project Overview

**Maintena Pro** is an end-to-end industrial **Predictive Maintenance (PdM)** platform designed for CNC milling equipment. By combining **supervised classification**, **linear regression**, and **unsupervised operating regime clustering** with physical engineering constraints, the platform predicts equipment failure before unplanned downtime occurs.

Built using the benchmark **10,000 CNC milling machine telemetry dataset** (AI4I 2020), this repository packages trained machine learning models, physics-informed failure logic, exploratory data analysis, and batch file processing into a **Streamlit** web application.

---

## 🚀 Key Features

* **🎯 Live Machine Health Predictor**: Interactive condition simulator with sliders in Celsius (°C) and RPM. Real-time failure probability gauge and plain-language root-cause explanations.
* **⚡ 4 Quick Operational Presets**: 1-click testing of *Nominal Steady-State*, *Thermal Dissipation Warning (HDF)*, *Tool Blade Fatigue (TWF/OSF)*, and *Motor Power Surge (PWF)*.
* **📊 Comprehensive Exploratory Data Analysis (EDA)**: 5 interactive views including fleet health KPIs, failure mode breakdowns, sensor distribution histograms, outlier boxplots, physical boundary scatters, and annotated correlation heatmaps.
* **📁 Batch Fleet Auditing**: Upload multi-unit CSV telemetry spreadsheets or load a pre-packaged 20-machine sample fleet to run automated diagnostics and export enriched reports.
* **🤖 Model Benchmarking Leaderboard**: Quantitative comparison across 7 classification algorithms (Random Forest, Bagging, Decision Tree, KNN, Logistic Regression, Naive Bayes) with confusion matrix inspections and feature importance rankings.
* **🌐 Machine Operating Regimes**: Unsupervised K-Means ($k=3$) and Gaussian Mixture Models (GMM) with 2D PCA mapping and live coordinate projections.
* **🛠️ Prescriptive Maintenance Protocol**: Prioritized, actionable checklists for machine operators and maintenance engineers.

---

## ⚙️ Machine Learning Architecture

```
                               ┌────────────────────────────────┐
                               │   Raw Machine Sensor Stream    │
                               │  (Type, Temps, Speed, Torque)  │
                               └───────────────┬────────────────┘
                                               │
                                               ▼
                               ┌────────────────────────────────┐
                               │    Preprocessing Pipeline      │
                               │  • StandardScaler (Numeric)    │
                               │  • OneHotEncoder (Type L,M,H)  │
                               └───────┬───────────────┬────────┘
                                       │               │
                     ┌─────────────────┴─┐           ┌─┴────────────────┐
                     ▼                   ▼           ▼                  ▼
             ┌───────────────┐   ┌───────────────┐ ┌───────────────┐  ┌───────────────────┐
             │ Supervised    │   │ Supervised    │ │ Unsupervised  │  │ Physics Engine    │
             │ Classification│   │ Regression    │ │ Clustering    │  │ Domain Rules      │
             │ Random Forest │   │ Linear Reg.   │ │ KMeans & GMM  │  │ PWF, HDF, OSF, TWF│
             │ (Fail/Healthy)│   │ (Process Temp)│ │ (Regimes 0-2) │  │                   │
             └───────┬───────┘   └───────┬───────┘ └───────┬───────┘  └─────────┬─────────┘
                     │                   │                 │                    │
                     └───────────────────┴────────┬────────┴────────────────────┘
                                                  │
                                                  ▼
                                 ┌─────────────────────────────────┐
                                 │   Streamlit Executive Frontend  │
                                 │ • Failure Probability (0-100%)  │
                                 │ • Physical Root-Cause Alerts    │
                                 │ • Prescriptive Action Protocols │
                                 └─────────────────────────────────┘
```

### 1. Classification (Failure Prediction)
- **Target**: `Machine failure` ($0 = \text{Healthy}$, $1 = \text{Failure}$)
- **Imbalance**: $3.39\%$ failure rate ($9,661$ normal vs $339$ failed)
- **Primary Metric**: **F1-Score** & **Recall** to minimize missed failures

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Tuned Random Forest (GridSearchCV)** | **97.4%** | **60.3%** | **68.6%** | **0.642** |
| **Bagging Classifier (50 Trees)** | **98.2%** | **86.4%** | **55.9%** | **0.679** |
| **Baseline Random Forest (100 Trees)** | **97.7%** | **69.3%** | **59.8%** | **0.642** |
| **Decision Tree (Balanced)** | **97.4%** | **63.0%** | **56.9%** | **0.598** |
| **K-Nearest Neighbors (k=5)** | **97.2%** | **70.7%** | **28.4%** | **0.406** |
| **Logistic Regression** | **83.1%** | **14.3%** | **79.4%** | **0.243** |
| **Gaussian Naive Bayes** | **95.7%** | **27.1%** | **15.7%** | **0.199** |

### 2. Regression (Process Temperature Estimation)
- **Target**: `Process temperature [K]`
- **Features**: `Air temperature`, `Rotational speed`, `Torque`, `Tool wear`
- **Model**: Linear Regression
- **Performance**: $R^2 = 0.758$ | $\text{MAE} = 0.590\text{ K}$ | $\text{RMSE} = 0.721\text{ K}$

### 3. Physics-Informed Anomaly Rules
The platform computes real-time physical bounds directly from machine dynamics:
- **Heat Dissipation Failure (HDF)**: $\Delta T = (T_{\text{proc}} - T_{\text{air}}) < 8.6\text{ K}$ and $\text{Speed} < 1380\text{ RPM}$.
- **Power Failure (PWF)**: Mechanical Shaft Power $P = \tau \cdot \left(\frac{2\pi N}{60}\right)$ exits $[3,500\text{ W} - 9,000\text{ W}]$.
- **Overstrain Failure (OSF)**: $\text{Torque} \times \text{Tool Wear} > \text{Limit}$ ($\text{Type L: } 11,000$, $\text{Type M: } 12,000$, $\text{Type H: } 13,000\text{ min}\cdot\text{Nm}$).
- **Tool Wear Failure (TWF)**: Cumulative blade usage $\ge 200 - 240\text{ minutes}$.

---

## 📊 Dataset Information

This project utilizes the **AI4I 2020 Predictive Maintenance Dataset** from the UCI Machine Learning Repository, reflecting 10,000 data points from real-world milling machines:

| Feature | Type | Range / Values | Description |
| :--- | :---: | :---: | :--- |
| **Type** | Categorical | `L` (50%), `M` (30%), `H` (20%) | Product quality variant |
| **Air temperature** | Continuous | $295.3\text{ K} - 304.5\text{ K}$ ($22.1 - 31.4\text{ }^\circ\text{C}$) | Ambient factory room temperature |
| **Process temperature** | Continuous | $305.7\text{ K} - 313.8\text{ K}$ ($32.5 - 40.6\text{ }^\circ\text{C}$) | Contact zone spindle temperature |
| **Rotational speed** | Integer | $1,168 - 2,886\text{ RPM}$ | Spindle rotational velocity |
| **Torque** | Continuous | $3.8 - 76.6\text{ Nm}$ | Torque resistance load |
| **Tool wear** | Integer | $0 - 253\text{ min}$ | Cumulative cutting time of the blade |
| **Machine failure** | Binary | `0` (Normal), `1` (Failure) | Overall failure target label |

---

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/maintena-pro.git
cd maintena-pro
```

### 2. Create and Activate Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Platform
```bash
streamlit run app.py
```
Or on Windows, double-click **`run_app.bat`**.

Then open **`http://localhost:8501`** in your browser.

---

## 📁 Repository Structure

```
maintena-pro/
├── app.py                     # Main Streamlit web application
├── utils.py                   # Cached model loaders, physics engine & batch processor
├── train_models.py            # Automated training, tuning, & serialization pipeline
├── sample_batch.csv           # 20-machine sample fleet for batch testing
├── requirements.txt           # Project dependencies
├── run_app.bat                # 1-Click Windows execution launcher
├── README.md                  # Comprehensive documentation
├── .gitignore                 # Git ignore rules
│
├── assets/
│   └── styles.css             # Custom glassmorphic dark theme styling
│
├── data/
│   └── ai4i2020.csv           # Cleaned 10,000-record benchmark dataset
│
└── models/                    # Serialized models & precomputed evaluations
    ├── preprocessor.pkl       # ColumnTransformer (StandardScaler + OneHotEncoder)
    ├── failure_model.pkl      # Tuned Random Forest Classifier
    ├── baseline_models.pkl    # Dictionary of all 6 baseline classifiers
    ├── regression_model.pkl   # Linear Regression model (Process Temperature)
    ├── cluster_scaler.pkl     # Scaler for clustering telemetry
    ├── kmeans_model.pkl       # K-Means model (k=3)
    ├── gmm_model.pkl          # Gaussian Mixture Model (k=3)
    ├── pca_model.pkl          # PCA (2 components)
    └── summary_metrics.json   # Precomputed evaluation metrics & cross-validation scores
```

---

## 🖥️ Dashboard Walkthrough

### 1. 🎯 Live Machine Health Predictor
- Realistic sliders with dual Celsius/Kelvin readouts.
- Radial gauge indicating failure risk percentage ($0\% - 100\%$).
- Prescriptive maintenance instructions for factory technicians.

### 2. 📊 Exploratory Data Analysis (EDA)
- **Fleet Health & Quality Breakdown**: Failure rates across Type L (3.92%), Type M (3.00%), and Type H (1.53%).
- **Distributions & Outlier Boxplots**: Histograms and boxplots showing normal vs failing sensor ranges.
- **Physical Boundaries**: Torque vs. Speed power curves, Tool Wear vs. Torque overstrain zones, and temperature differentials.
- **Correlation Heatmap**: Annotated correlation matrix.
- **Raw Data Explorer**: Searchable, filterable table with full statistical summaries.

### 3. 📁 Batch Fleet Diagnostics
- Upload telemetry CSV files to audit multiple machines simultaneously.
- 1-click test button with pre-bundled `sample_batch.csv`.
- Export enriched reports with risk levels, probabilities, and operating clusters.

### 4. 🤖 ML Models Leaderboard & Benchmarks
- Side-by-side Accuracy, Precision, Recall, and F1 comparison charts.
- Interactive confusion matrix selector.
- Feature importance rankings (Top features: Rotational Speed, Torque, Tool Wear).

### 5. 🌐 Machine Operating Regimes
- Unsupervised operating state segmentation (K-Means & GMM).
- 2D PCA coordinate map displaying the entire fleet with the user's active simulator point projected live.

---

## 📦 Requirements

- **Python**: `>= 3.10` (tested on 3.10, 3.11, 3.12, 3.14)
- **Core Libraries**:
  - `streamlit >= 1.30.0`
  - `scikit-learn >= 1.3.0`
  - `pandas >= 2.0.0`
  - `numpy >= 1.24.0`
  - `plotly >= 5.18.0`
  - `joblib >= 1.3.0`

---


---

<div align="center">
  <b>Built for Industrial Reliability & Predictive Maintenance</b>
</div>
