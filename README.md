# SIH26009: Manganese Reserve Prediction & Production Shortfall Prevention

An end-to-end Machine Learning decision-support system and terminal prototype for **Smart India Hackathon 2026** (Problem Statement **SIH26009**), developed for the MOIL Central Indian Manganese Mining Belt (Balaghat, Gumgaon, Kandri, Munsar, Ukwa, Chikla, and Dongri Buzurg).

---

## 📌 Project Overview

Manganese mining operations in central India face significant daily production shortfalls driven by:
1. **Equipment breakdowns** (primary hoists, crushers, shovels).
2. **Internal logistics bottlenecks** (haul truck shortages, dumper dispatch delays).
3. **Monsoon microclimate events** (pit sump flooding, haul road slush slowing tramming cycles).
4. **Geotechnical and blasting delays** (muckpile clearance, secondary rock fragmentation).

This project integrates mine lease geology, NASA POWER satellite daily meteorological records, and historical shift-level shortfall data to:
- Benchmark 4 regression algorithms (Ridge, Random Forest, XGBoost, and Gradient Boosting).
- Select the top-performing model (**Gradient Boosting Regressor** with ^2 = 0.8516$ and Test $\text{MAE} = 19.72\text{ tonnes}$).
- Provide a lightweight, zero-dependency interactive CLI prototype that operators and shift engineers can run directly on pithead terminals to forecast production shortfalls and obtain prescriptive advisory directives.

---

## 🗂️ Repository Structure

`	ext
├── data/                                  # Raw CSV Datasets
│   ├── Manganese Reservation Prediction - Mine Lease Details.csv
│   ├── Production_Shortfall_India_SIH_2026.csv
│   └── nasa_power_daily.csv
├── figures/                               # EDA, Correlation & Feature Importance Plots
│   ├── fig1_correlation_matrix.png       # Correlation heatmap across operational & weather metrics
│   ├── fig2_downtime_vs_shortfall.png    # Equipment stoppage vs production loss
│   ├── fig3_transport_vs_shortfall.png   # Transport availability curves
│   ├── fig4_seasonal_shortfall.png       # Monsoon vs non-monsoon monthly shortfall
│   ├── fig5_rainfall_vs_transport.png    # Precipitation impact on fleet logistics
│   ├── fig6_mine_baselines.png           # Mine-by-mine production quota vs historical shortfalls
│   ├── fig7_shap_summary.png             # SHAP feature attribution summary
│   └── fig8_xgboost_importance.png       # Gini feature importance ranking
├── models/                                # Trained weights & evaluation metrics
│   ├── best_model_gbr.pkl                # Selected best Gradient Boosting model
│   ├── feature_names.pkl                 # Exact feature alignment schema
│   ├── benchmark_summary.json            # MAE / RMSE / R2 comparative metrics
│   ├── xgboost_model.pkl
│   ├── rf_model.pkl
│   └── ridge_model.pkl
├── train_models.py                        # Training & model benchmarking pipeline
├── terminal_app.py                        # Minimalist interactive decision-support prototype
├── requirements.txt                       # Python dependencies
└── README.md
`

---

## 📊 Exploratory Data Analysis & Visualizations

| Figure | Description |
| :--- | :--- |
| **igures/fig1_correlation_matrix.png** | Correlation heatmap showing high coupling between equipment downtime ( \approx 0.65$), transport logistics ( \approx -0.48$), and daily shortfall. |
| **igures/fig2_downtime_vs_shortfall.png** | Scatter and trend line illustrating non-linear failure thresholds when machine downtime exceeds 4 hours. |
| **igures/fig3_transport_vs_shortfall.png** | Impact of haul truck and dumper availability on surface ore transfer. |
| **igures/fig4_seasonal_shortfall.png** | Monthly seasonality curves confirming monsoon surge (July–September). |
| **igures/fig7_shap_summary.png** | Tree SHAP explanation showing how each operational factor drives the final prediction. |

---

## 🧠 Model Benchmark & Selection Architecture

`	ext
TRAINING & BENCHMARKING (train_models.py):
  Ridge Regression (Linear)       ─────┐
  Random Forest (Bagging)         ─────┤
  XGBoost (Gradient Tree Boost)   ─────┤ ───► Model Benchmark (MAE / R2) ───► Best Model (GBR)
  Gradient Boosting (GBR)         ─────┘

INTERACTIVE PROTOTYPE (terminal_app.py):
  User / Shift Incharge Input
             │
             ▼
      Best Model Only (GBR)
             │
             ▼
  Daily Shortfall Forecast (Tonnes) + Prescriptive Action Directives
`

### Performance Comparison on Unseen Test Data

| Algorithm | Test MAE (t) | Test RMSE (t) | Test ^2$ | Why / Why Not Chosen |
| :--- | :---: | :---: | :---: | :--- |
| **Gradient Boosting Regressor (GBR)** | **19.72** | **31.28** | **0.8516** | **Selected**: Lowest MAE and highest ^2$; best sequential residual error correction. |
| **Random Forest Regressor** | 20.77 | 33.79 | 0.8270 | Good stability, but bagging cannot sequentially correct prior split errors. |
| **XGBoost Regressor** | 21.05 | 34.20 | 0.8227 | Fast, but slightly higher MAE than GBR on extreme monsoon conditions. |
| **Ridge Regression** | 26.63 | 44.52 | 0.7000 | Inadequate: Linear hypothesis cannot capture sudden breakdown inflection points. |
| **Historical Baseline Heuristic** | 39.90 | 66.82 | 0.2692 | Failed: Static daily mean cannot react to operational stoppages. |

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
`ash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
`

### 2. Set Up Virtual Environment & Dependencies
`ash
python -m venv .venv

# On Linux / WSL2:
source .venv/bin/activate

# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
`

### 3. (Optional) Re-train and Compare Models
`ash
python train_models.py
`

### 4. Run the Prototype
`ash
python terminal_app.py
`

---

## 🖥️ Prototype Walkthrough

When running 	erminal_app.py, the CLI guides the shift supervisor through:
1. **Mine Selection**: 7 supported mines across Maharashtra & Madhya Pradesh.
2. **Scenario Mode**:
   - 1: Normal Dry Day (Optimal Shift)
   - 2: Major Equipment Breakdown (e.g. 7.5 hrs primary hoist trip)
   - 3: Heavy Monsoon Rain (55 mm cloudburst + haul road slush)
   - 4: Blasting Delay & Workforce Shortage
   - 5: Custom Input (specify downtime, rainfall, transport availability, and labour attendance)
3. **Prescriptive Directives**: Immediate guidance such as starting secondary sump dewatering pumps or dispatching auxiliary contractor dumpers.

---

## 👥 Hackathon Team & Acknowledgements
- **Smart India Hackathon 2026**
- **Problem Statement SIH26009**: Manganese Mining Shortfall Prevention & Reserve Prediction
