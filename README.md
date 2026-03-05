# Asian-PCa-Lymphatic-Leak-Risk

This repository contains the source code and analytical pipeline for the study:

**"The Ethnic Rift in Metabolic Risk: Why Western Models Underestimate Lymphatic Complications in Asian Prostate Cancer Patients—A Deep Phenotyping and Machine Learning Study"**

## 📄 Overview

This project investigates the ethnic-specific risk factors for chyle leak (CL) following radical prostatectomy in Asian populations. It employs a "Deep Phenotyping" approach combined with Machine Learning (XGBoost) to identify a unique metabolic risk threshold significantly lower than Western standards.

Key components of the analysis include:
1.  **Multivariate Logistic Regression**: Identifying independent clinical predictors (Table 3).
2.  **XGBoost Machine Learning Model**: Capturing non-linear risk interactions (Figure 4).
3.  **SHAP (SHapley Additive exPlanations)**: Interpreting model decisions (Figure 3).
4.  **Cost-Benefit Analysis**: Bridging deep phenotyping with surgical strategy (Figure 5a).
5.  **Cross-Ethnic Stress Test**: demonstrating the calibration drift of Western models in Asian cohorts (Figure 5c).

## 🛠️ Requirements

The analysis was performed using Python 3.9+. Key dependencies include:

* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `shap`
* `lifelines` (for survival analysis)
* `matplotlib` & `seaborn` (for visualization)

You can install the dependencies using:
```bash
pip install pandas numpy scikit-learn xgboost shap lifelines matplotlib seaborn
