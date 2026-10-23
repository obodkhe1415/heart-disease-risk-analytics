# ❤️ Executive Heart Disease & Cardiovascular Risk Analytics Platform

## 📌 Project Overview
An end-to-end Machine Learning and Business Intelligence web application built for the **IBM SkillsBuild & AICTE Data Analytics Internship Program 2026**.

The application evaluates clinical biomarkers (resting blood pressure, serum cholesterol, fasting blood sugar, max heart rate, exercise angina, and ST depression) to predict cardiovascular disease risk using Machine Learning, generating clinical risk stratifications and delivering a 5-level decision dashboard designed for clinical directors and healthcare executives.

---

## 📊 5-Level Business Intelligence Structure
1. **Level 1 (KPIs)**: Evaluated Patient Count, High-Risk Patient Rate %, Avg Resting BP (mmHg), Model R² Score.
2. **Level 2 (Trends)**: Resting Blood Pressure vs Cholesterol scatter plots with OLS trendlines by Chest Pain Type.
3. **Level 3 (Drivers)**: Correlation matrix quantifying the clinical drivers of cardiovascular risk score.
4. **Level 4 (AI Valuation Engine)**: Real-time patient risk estimator & clinical risk stratification engine.
5. **Level 5 (Executive Action)**: Preventive cardiology referral guidelines, automated EHR clinical alerts, and community wellness programs.

---

## 🔗 Dataset Source
* **Dataset**: Heart Disease UCI Clinical Dataset
* **Source Link**: [Kaggle Heart Disease Dataset](https://www.kaggle.com) *(Update with your exact Kaggle dataset link)*

---

## 🛠️ Technology Stack
| Layer | Technology |
|---|---|
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Random Forest Regressor) |
| Web Dashboard | Streamlit |
| Data Visualization | Plotly Express |
| Documentation | Python-Docx, Markdown |

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/<your-github-username>/heart-disease-risk-analytics.git
cd heart-disease-risk-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

---

## 📁 Repository Structure
```
heart-disease-risk-analytics/
├── app.py                      # Streamlit dashboard + ML model
├── generate_report.py          # Builds the Word report with real model metrics
├── data/heart_disease_data.csv # Clinical dataset (25 patients)
├── reports/                    # Generated project report (.docx)
├── requirements.txt
└── README.md
```

## ⚠️ Limitations
The dataset is small (25 records), so model metrics vary a lot between splits (test R² ≈ 0.38 on a 5-row hold-out; ≈ 0.63 mean 5-fold CV). This is an educational project, not a medical device.
