# PflegeOptix — Elderly Care Planning & Resource Allocation Dashboard

**PflegeOptix**

An interactive planning dashboard for forecasting and optimizing elderly care home bed allocation across German districts (Landkreise) for 2025–2040.

🔗 **Live Dashboard**: [pflegeoptix.streamlit.app](https://pflegeoptix.streamlit.app/)

The project addresses the demographic challenge of Germany's aging population by combining socioeconomic predictors, time-series forecasting, and linear programming to ensure that new bed budgets are allocated fairly, preventing undersupply in both dense urban centers and dispersed rural areas.

---

## 🚀 How to Run Locally

### 1. Prerequisites
- Python 3.10 or higher.

### 2. Setup Virtual Environment
In the root directory, create and activate a virtual environment, then install the dependencies:

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Activate (macOS/Linux)
source .venv/bin/activate

# Install all development and model training dependencies
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
Navigate to the app folder and run Streamlit:

```bash
cd PflegeOptix
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```text
Capstone Project/
│
├── .gitignore                   # Excludes venv, pycache, and massive raw data files
├── requirements.txt             # Development and training dependencies (pinned)
├── README.md                    # Project documentation
├── folder_tree.txt              # Historical folder layout
│
├── reports/                     # Output documentation and diagnostic plots
│   └── figures/                 # Pre-computed plots (SHAP, Prophet, XGBoost)
│
└── PflegeOptix/                 # Deployed Web Application
    ├── app.py                   # Main Streamlit dashboard script
    ├── requirements.txt         # Minimal dependency list for cloud deployment
    │
    ├── static/                  # Localized SHAP figures for the dashboard
    │   ├── shap_bar_clean.png
    │   └── shap_beeswarm_clean.png
    │
    ├── dataset/                 # Data directory containing cleaned datasets
    │   └── dashboard/
    │       ├── master_clean.csv            # Cleaned historical data (2015-2023)
    │       ├── forecast_clean.csv          # Prophet population forecasts (2025-2040)
    │       ├── pulp_clean.csv              # XGBoost demand and pre-optimization gap data
    │       ├── landkreise_with_ags.geo.json# GeoJSON boundaries with injected AGS codes
    │       └── ags_geojson_mapping.csv     # AGS-to-GeoJSON name-mapping references
    │
    └── notebooks/                # Jupyter Notebooks detailing the data science pipeline
        ├── 01_cleaning.ipynb               # Sprint 1: Data ETL, merging, and imputation
        ├── 02_prophet.ipynb                # Sprint 2: Time-series population forecasting
        ├── 03_xgboost_shap.ipynb           # Sprint 3: Demand modeling and SHAP explainability
        └── 04_pulp_optimization.ipynb      # Sprint 3: PuLP resource allocation modeling
```

---

## ⚙️ The Data & Machine Learning Pipeline

The project is structured into four sequential development phases (Sprints):

### 1. Data Engineering & ETL (Sprint 1)
- **Sources**: Consolidates 10 separate annual population files (2015-2024) from Destatis, care provider supply metrics from Pflegestatistik, and over 20 socioeconomic indicators from the **INKAR 2025** database (6.7 GB raw).
- **Processing**: Standardizes District Association Codes (AGS) to 5-digit padded strings, resolves spatial name mismatches, and applies linear imputation for missing indicators.

### 2. Time-Series Population Forecasting (Sprint 2)
- **Methodology**: Uses **Facebook Prophet** to forecast the total population for all 399 valid districts up to 2040.
- **Validation**: Evaluated using historical backtesting (rolling windows), outperforming a standard linear baseline by reducing Mean Absolute Percentage Error (MAPE) to under **0.55%**.

### 3. Care Demand & Explainability (Sprint 3)
- **Model**: An **XGBoost Regressor** predicts the benchmark care-bed ratio (Pflegequote) based on local socioeconomic profiles.
- **Explainable AI (XAI)**: Pre-computed **SHAP** values explain feature contribution. Major drivers include *Premature Mortality* (reflecting local healthcare quality), *Population Density* (urbanization constraints), and *Single-Person Household Percentage* (lack of informal home care).
- **Ablation Study**: Compares the full model ($R^2 \approx 0.90$) against a clean model ($R^2 \approx 0.43$). The full model suffered from *proxy leakage* (using current care home capacities to predict demand). Removing the leaking variable yielded a clean, policy-grade predictive model.

### 4. Mathematical Resource Allocation (Sprint 3 & 4)
- **Algorithm**: Implements **PuLP (Linear Programming)** utilizing a **Min-Max Fairness** objective. 
- **Mechanism**: Rather than assigning all beds greedily to the largest cities (leaving rural areas with 100% unmet deficit), the optimizer minimizes the *maximum unmet deficit ratio* across all selected districts, sharing the burden of supply shortages equally.

---

## 💡 Cloud Deployment Notes (Streamlit Community Cloud)

If deploying the app to the cloud, keep the following parameters in mind:

- **Self-Contained Subfolder**: Deploy using the `PflegeOptix/` subfolder. All necessary datasets and SHAP images have been moved inside it. The app has no external path dependencies.
- **GitHub Size Exclusions**: Ensure that `.gitignore` successfully blocks the raw **6.7 GB** `inkar_2025.csv` file from being pushed. The pre-processed CSVs in `dataset/dashboard/` are small (~1.5 MB total) and contain everything the dashboard needs.
- **Runtime Dependencies**: The dashboard itself does not run XGBoost, Prophet, or SHAP training on-the-fly. Therefore, [PflegeOptix/requirements.txt](file:///d:/Capstone%20Project/PflegeOptix/requirements.txt) only includes basic runtime libraries (`streamlit`, `pandas`, `numpy`, `plotly`, `pulp`). Excluding heavy ML compilation libraries ensures fast container deployment and avoids compiler failures.

---

## ⚠️ Known Limitations

- **Berlin Exclusion**: Excluded from the map representation due to a hierarchical mismatch (statistical data is split into 12 districts, whereas the GeoJSON represents Berlin as a single unified polygon).
- **Unmapped Districts**: 16 districts (representing ~4% of the dataset) are calculated in the tables and data downloads but do not render on the choropleth map. This is due to historical regional boundary changes in Germany that are not captured in the open-source GeoJSON map.
