# Credit Scoring with Imbalanced Learning, Bayesian Optimization, and SHAP

> Benchmarking data balancing (ROS/RUS/SMOTE) vs algorithm-level class weighting (`scale_pos_weight`) across gradient boosting models (LightGBM/XGBoost/CatBoost); optimized with Bayesian (Optuna/TPE) and explained via SHAP.

## 🔎 Overview

This project evaluates how different **imbalanced learning strategies** affect both **predictive performance** and **interpretability** in credit scoring.

- **Models:** LightGBM, XGBoost, CatBoost  
- **Balancing:** Random Over/Under-Sampling (ROS/RUS), **SMOTE**, and **class weighting** (`scale_pos_weight`)  
- **Optimization:** **Bayesian Optimization** (Optuna/TPE)  
- **Explainability:** **SHAP** with **stability** measured via **Spearman rank correlation**  
- **Metrics:** AUC, TPR/TNR, FPR, **G-Mean**, Accuracy

**Key result (reported):**  
**LightGBM + `scale_pos_weight`** achieved **AUC 0.7841**, **G-Mean 0.7009**, and **SHAP stability Spearman 0.98**. In contrast, **SMOTE** degraded minority detection (**G-Mean 0.3626**) and shifted model logic (**Spearman 0.64**).  
> Results may vary depending on preprocessing and random seeds.

---

## 🗄️ Dataset

- **Source:** Home Credit Default Risk (public Kaggle dataset).  
