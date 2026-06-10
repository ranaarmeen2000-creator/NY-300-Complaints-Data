# NYC 311 Service Requests — Predicting Complaint Resolution Time

A machine learning project that predicts how long a NYC 311 service complaint will take to resolve, using classification models trained on over 300,000 real service requests.

---

## Problem Statement

New York City receives millions of 311 non-emergency service requests every year — from noise complaints and pothole reports to heat outages and unsanitary conditions. This project builds a **multi-class classification model** to predict which resolution time category a complaint will fall into at the time it is filed.

**Target Classes:**
| Class | Label | Meaning |
|-------|-------|---------|
| 0 | Fast | Resolved within 2 days |
| 1 | Moderate | Resolved in 3–7 days |
| 2 | Slow | Resolved in more than 7 days |

---

## Dataset

**Source:** [NYC Open Data — 311 Service Requests from 2010 to Present](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)

- 16+ million rows, 53 features
- Publicly available, no licensing restrictions for research use
- Features include complaint type, agency, borough, location type, timestamps, and channel type

> The notebook loads 300,000 rows by default for performance. Remove `nrows=300_000` in the data loading cell to use the full dataset.

---

## Project Structure

```
├── ML_PROJECT_FINAL.ipynb   # Main notebook — all code, EDA, models, results
└── README.md
```

---

## Machine Learning Pipeline

### 1. Data Preprocessing
- Calculated resolution time in days from `Created Date` and `Closed Date`
- Removed invalid/null resolution times and outliers (> 365 days)
- Binned resolution time into 3 target classes
- Extracted date features: hour, day of week, day of month, month, year
- Handled missing values (drop key nulls, fill others with `'Unknown'`)
- Label-encoded categorical variables: Agency, Complaint Type, Descriptor, Location Type, Borough, Channel Type

### 2. Exploratory Data Analysis
- Most/least frequent complaint types
- Complaint distribution across NYC boroughs
- Volume by month and hour of day
- Average resolution time by agency
- Target class distribution
- Correlation heatmap

### 3. Models Trained

| Model | Justification |
|-------|--------------|
| Logistic Regression | Simple interpretable baseline |
| Decision Tree | Captures non-linear patterns; interpretable rules |
| Random Forest | Ensemble method; handles imbalance well |
| XGBoost | Gradient boosting; top performer on tabular data |

### 4. Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score (weighted)
- Confusion matrices for all models

### 5. Hyperparameter Tuning
- Applied **RandomizedSearchCV** on the best-performing model (Random Forest)
- Parameters tuned: `n_estimators`, `max_depth`, `min_samples_split`, `max_features`
- Before vs after comparison included

---

## How to Run

### Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
```

Install all dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
```

### Steps
1. Download the dataset from [NYC Open Data](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9)
2. Clone this repository
3. Open `ML_PROJECT_FINAL.ipynb` in Jupyter Notebook
4. Update the `DATA_PATH` variable in Cell 3 to your local CSV path:
```python
DATA_PATH = r"C:\Your\Path\To\311-service-requests-from-2010-to-present.csv"
```
5. Run all cells (Kernel → Restart & Run All)

---

## Key Findings

- **Agency** is the strongest predictor of resolution time — NYPD and DHS resolve complaints within hours, while DPR (Parks) takes weeks on average
- **Brooklyn and the Bronx** generate the highest complaint volumes with longer resolution times, suggesting resource allocation gaps
- **Noise complaints** are resolved fastest (often same day); structural and environmental complaints take significantly longer
- **Summer months** see peak complaint volumes, particularly for noise and heat-related issues
- **Random Forest (tuned)** achieved the best overall F1 score across all three classes

---

## Business Recommendations

| Recommendation | Evidence |
|----------------|----------|
| Set agency-specific SLA targets | Agency is the top predictive feature |
| Show citizens real-time resolution ETAs | Model predicts class at submission time |
| Increase winter staffing for heat/hot water | Clear seasonal spike in EDA |
| Audit Brooklyn & Bronx resource allocation | Highest volume + longer resolution times |
| Retrain model quarterly | Data drift risk as policies and staffing change |

---

## Limitations

- Class imbalance: most complaints fall into the "Fast" class, making "Slow" harder to predict
- Historical data may not reflect post-2020 changes in city staffing or complaint patterns
- External factors (weather, events, storms) are not included as features
- Fairness risk: model should be audited across boroughs before any deployment

---

## Author

This project was completed as a final machine learning assignment applying an end-to-end ML pipeline to a real-world business analytics problem.
