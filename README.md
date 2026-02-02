# 🎓 Student Retention & Dropout Prediction Engine

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/cfdcc906-fa3f-4aac-97b8-34966aac45a6" />

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![TensorFlow](https://img.shields.io/badge/Model-Neural%20Network-yellow?logo=tensorflow)](https://www.tensorflow.org/)
[![Status](https://img.shields.io/badge/Status-Completed-green)]()

A comparative supervised learning study utilizing **XGBoost** and **Neural Networks** to predict student dropout risk across 25,000+ learners. The system is designed to enable early intervention strategies for and international education company.

## 🎯 Business Problem

**The Challenge:**
Student attrition significantly impacts educational outcomes and institutional revenue. Identifying at-risk students *before* they drop out allows for targeted support interventions.

**The Objective:**
Develop a predictive engine to flag high-risk students by analyzing:
1.  **Demographics** (Static data)
2.  **Engagement** (Attendance patterns)
3.  **Performance** (Academic results)

**Impact:**
Optimizing the trade-off between **Recall** (catching all dropouts) and **Precision** (minimizing wasted intervention costs on false alarms).

---

## 🏗 Solution Architecture

The analysis was structured into a **3-Stage Feature Evolution Pipeline** to determine the earliest possible point of accurate prediction.

### Stage 1: The "Cold Start" (Demographics Only)
*   **Input:** Applicant data, course details, nationality, and age.
*   **Finding:** Baseline models showed that nationality was a significant early predictor, but overall predictive power was limited without behavioral data.

### Stage 2: The "Behavioral" Lift (Engagement Metrics)
*   **Input:** Added **Authorized** and **Unauthorized Absence** counts.
*   **Finding:** Significant performance boost. `UnauthorizedAbsenceCount` emerged as the #7 most important feature, validating the hypothesis that disengagement precedes dropout.

### Stage 3: The "Performance" Leap (Academic Results)
*   **Input:** Assessed, Passed, and Failed module counts.
*   **Finding:** Models achieved near-perfect accuracy, but this raised a critical **Data Leakage** concern (discussed below).

---

## 🛠 Methodology & Tech Stack

*   **Models:**
    *   **XGBoost:** Hyperparameter tuned via `GridSearchCV` (Learning Rate, Max Depth, Child Weight).
    *   **Neural Networks:** Deep Learning architecture tuned via `KerasTuner` (Hyperband algorithm).
*   **Handling Imbalance:** Applied **Class Weighting** (6:1 imbalance) to prioritize Recall (catching dropouts) over accuracy.
*   **Evaluation:** AUC-ROC (primary), F1-Score, and Confusion Matrices.

## 📉 Critical Finding

A key finding of this project was the **Post-Mortem Analysis** of Stage 3.

**The Data Leakage Discovery:**
While Stage 3 achieved near-perfect metrics, I identified that features like `FailedModules` likely arrive *simultaneously* with the dropout event, rendering them useless for *early* prediction.

**Recommendation:**
*   **Operational Pivot:** Production deployment should rely on **Stage 2 (Behavioral)** models for early warning systems.
*   **Data Engineering Fix:** Future iterations require strict temporal splitting (e.g., training on Q1 academic data to predict Q2 dropout) to make Stage 3 features viable.

## 📊 Results Summary

| Model | Stage | Insight | Verdict |
| :--- | :--- | :--- | :--- |
| **XGBoost** | Stage 2 | High AUC, Fast Inference | **Recommended for Production** |
| **Neural Network** | Stage 2 | Comparable Performance | Higher Compute Cost |
| **XGBoost** | Stage 3 | Near-Perfect Accuracy | **Rejected (Data Leakage)** |

**Why XGBoost Won:**
Despite Neural Networks showing marginally higher AUC in some tests, **XGBoost** was selected for the final prediction model due to:
1.  **Explainability:** Feature importance plots (SHAP) allowed stakeholders to understand *why* a student was flagged (e.g., "High Absence").
2.  **Efficiency:** Lower training cost and faster inference latency compared to the Deep Learning stack.

## 📂 Project Structure

```text
├── notebooks/
├── requirements.txt        # Dependencies
└── README.md               # Documentation
