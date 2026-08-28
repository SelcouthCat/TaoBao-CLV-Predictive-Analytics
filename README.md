# 🛒 E-Commerce Customer Lifetime Value (CLV) Predictive Analytics
> **A Two-Stage Hybrid Framework Combining BG/NBD Probabilistic Model and Gradient Boosted Trees (XGBoost) with SHAP Interpretability**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost%20%7C%20BG--NBD-orange.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Interpretability-TreeSHAP-brightgreen.svg)](https://github.com/slundberg/shap)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📌 Executive Summary

Predicting short-term customer repurchase frequency and Customer Lifetime Value (CLV) during promotional campaigns is challenging due to extreme **zero-inflation** and **long-tail right-skewness** in transaction data. 

This repository implements an end-to-end data science architecture on the **Alibaba Taobao User Behavior Dataset (~100 Million interaction logs, 3.67 GB)**:
1. **Stage 1 (Macro Priors)**: Utilizes the **BG/NBD (Beta-Geometric/Negative Binomial Distribution)** stochastic model to capture latent churn probabilities $P(\text{Alive})$ and baseline transaction rates.
2. **Stage 2 (Micro Signal Integration)**: Feeds empirical Bayesian priors alongside non-linear log-transformed micro-clickstream signals (`pv`, `cart`, `fav`, and interaction-to-purchase ratios) into an **XGBoost Regressor** with $L_1/L_2$ regularization.
3. **Statistical Validation & Interpretability**: Validates performance via **Paired t-tests**, **Wilcoxon signed-rank tests**, **5-Fold Cross-Validation**, and **TreeSHAP** game-theoretic feature attribution.

---

## 🏗️ Project Architecture & Pipeline

```text
TaoBao-CLV-Predictive-Analytics/
├── data/
│   ├── sample_data.csv                   # First 1,000 records for fast pipeline testing
│   └── .gitkeep                          # (Raw 3.67GB UserBehavior.csv git-ignored)
├── figures/                              # High-resolution generated visual plots
│   ├── EDA/                              # Daily trends, conversion funnel, 24h patterns
│   ├── NBD/                              # Recency-frequency matrices, P(Alive) decay, MAE/RMSE
│   └── XGBoost/                          # Model benchmark, SHAP bar, SHAP beeswarm
├── notebooks/                            # Annotated end-to-end research pipelines
│   ├── 01_eda_data_exploration.ipynb     # Chunk-ingestion, outlier cleaning, behavioral EDA
│   ├── 02_bgnbd_probabilistic_model.ipynb # Calibration/Holdout split & BG/NBD MLE estimation
│   └── 03_xgboost_hybrid_modeling_shap.ipynb # Two-stage XGBoost, hypothesis testing, SHAP
├── output/                               # Processed intermediate feature matrices
│   ├── total_behaviour_summary.csv
│   ├── bgnbd_features_for_xgboost.csv
│   └── user_micro_behavior_features.csv
├── requirements.txt                      # Environment dependencies
├── .gitignore
└── README.md
