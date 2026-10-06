# 🧬 IDC vs ILC Classification Using Gene Expression Data

<p align="center">
  <strong>Comparative Machine Learning & Explainable AI for Breast Cancer Molecular Classification</strong>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white" alt="Python">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn">
  </a>
  <a href="https://xgboost.readthedocs.io/">
    <img src="https://img.shields.io/badge/XGBoost-Gradient%20Boosting-1A1A1A" alt="XGBoost">
  </a>
  <a href="https://shap.readthedocs.io/">
    <img src="https://img.shields.io/badge/SHAP-Explainable%20AI-8A2BE2" alt="SHAP">
  </a>
  <img src="https://img.shields.io/badge/Domain-AI%20Healthcare-0F766E" alt="AI Healthcare">
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-2EA44C" alt="MIT License">
  </a>
</p>

<p align="center">
  <strong>AI Healthcare Research · Computational Oncology · Machine Learning · Explainable AI</strong>
</p>

---

## 🔬 Research Overview

Breast cancer exhibits substantial molecular and histological heterogeneity. **Infiltrating Ductal Carcinoma (IDC)** and **Infiltrating Lobular Carcinoma (ILC)** represent distinct histological forms with different biological characteristics.

This research investigates whether **high-dimensional gene-expression and molecular features** can distinguish IDC from ILC using machine learning.

The study compares:

**Logistic Regression · Random Forest · XGBoost**

The analysis extends beyond classification performance by incorporating:

**Feature Selection · Cross-Validation · Statistical Testing · Permutation Importance · SHAP Explainability · Consensus Feature Ranking**

### Research Objective

> **To develop a reproducible and interpretable computational framework for IDC vs ILC classification and to investigate molecular features contributing to model predictions.**

> **Status:** Ongoing computational research. Results and biological interpretations remain subject to further validation.

---

# 🎯 Research Question

> **Can machine learning reliably distinguish IDC from ILC using high-dimensional molecular features, and which features contribute most strongly to the predictions?**

The study examines:

* Comparative model performance
* Effect of dimensionality reduction
* Cross-validation stability
* Statistical differences between models
* Consistency of predictive features across interpretability methods

---

# 🧬 Biological Context

### Infiltrating Ductal Carcinoma — IDC

A common invasive breast carcinoma originating primarily from the breast ducts.

### Infiltrating Lobular Carcinoma — ILC

An invasive breast carcinoma originating from the lobular tissue of the breast and characterized by distinct morphological and molecular features.

In this study, the distinction is formulated as a **binary supervised-learning problem**:

```text
histological.type

        ┌─────────┐
        │   IDC   │
        └────┬────┘
             │
             │  Classification
             │
             ▼
        ┌─────────┐
        │   ILC   │
        └─────────┘
```

---

# 🧪 Data

The study uses a **high-dimensional breast cancer molecular dataset** containing gene-expression and related molecular/clinical variables.

### Target variable

```text
histological.type
```

### Classification task

```text
IDC  ↔  ILC
```

The feature space is substantially larger than the number of observations, making **feature selection, regularization, validation strategy, and leakage control** important methodological considerations.

🔒 **Raw datasets are not included in this public repository.**

See [`data/README.md`](data/README.md).

---

# ⚙️ Research Framework

```text
                    MOLECULAR DATA
                           │
                           ▼
              ┌────────────────────────┐
              │ Quality Control & EDA  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Stratified Data Split  │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ Leakage-Safe           │
              │ Preprocessing          │
              └────────────┬───────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          All Features         Feature Selection
                                      │
                               Top 100 / Top 500
                └──────────┬──────────┘
                           ▼
              ┌────────────────────────┐
              │  Logistic Regression   │
              │  Random Forest         │
              │  XGBoost               │
              └────────────┬───────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ 5-Fold Stratified CV   │
              └────────────┬───────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
          Performance            Statistical
           Evaluation             Analysis
                │                     │
                └──────────┬──────────┘
                           ▼
                   SHAP Explainability
                           │
                           ▼
              Consensus Feature Ranking
                           │
                           ▼
                Biological Interpretation
```

---

# 🤖 Model Architecture

| Model                   | Purpose                                                   |
| ----------------------- | --------------------------------------------------------- |
| **Logistic Regression** | Regularized linear baseline                               |
| **Random Forest**       | Nonlinear ensemble learning                               |
| **XGBoost**             | Gradient-boosted nonlinear modeling + SHAP interpretation |

All models are evaluated within a consistent preprocessing and validation framework.

---

# 🧩 Feature Selection

Three feature configurations are investigated:

| Configuration    | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| **All Features** | Full molecular feature representation                      |
| **Top 100**      | Reduced representation using the top 100 selected features |
| **Top 500**      | Reduced representation using the top 500 selected features |

Feature selection is performed using training data within the corresponding experiment to reduce information leakage.

The objective is to determine whether a smaller feature representation can preserve or improve predictive performance while improving interpretability.

---

# 🔁 Validation Strategy

Model stability is evaluated using:

### **5-Fold Stratified Cross-Validation**

Stratification preserves class proportions across folds.

### Evaluation metrics

**Accuracy · Precision · Recall · F1-score · ROC AUC**

The analysis also considers confidence intervals around cross-validation performance.

---

# 📊 Statistical Analysis

Model comparison incorporates statistical analysis rather than relying only on mean performance.

Pairwise cross-validation performance is evaluated using the:

**Wilcoxon signed-rank test**

This provides an additional assessment of whether observed performance differences are supported by the validation results.

---

# 🧠 Explainable AI

Predictive performance alone does not explain model behavior.

The XGBoost model is therefore investigated using **SHAP (SHapley Additive exPlanations)**.

The interpretability analysis examines:

* Global feature importance
* Feature contribution direction
* Distribution of feature effects
* Top predictive features

Feature importance is additionally assessed through:

**XGBoost importance + Permutation importance + SHAP importance**

These complementary rankings are combined into a **consensus feature analysis**.

> Computational feature importance does not establish biological causality or clinical biomarker validity.

---

# 📈 Research Outputs

The repository contains selected analytical outputs generated from the study.

### Feature Selection

[`consensus_biomarkers.csv`](results/feature_selection/consensus_biomarkers.csv)
[`all_feature_importance.csv`](results/feature_selection/all_feature_importance.csv)
[`feature_selection_experiments.csv`](results/feature_selection/feature_selection_experiments.csv)

### SHAP Analysis

[`shap_feature_importance.csv`](results/shap/shap_feature_importance.csv)
[`xgboost_shap_feature_importance.csv`](results/shap/xgboost_shap_feature_importance.csv)
[`xgboost_shap_directional_analysis.csv`](results/shap/xgboost_shap_directional_analysis.csv)

---

# 🖼️ Research Figures

### Consensus Feature Ranking

![Consensus Feature Ranking](results/figures/consensus_biomarkers.png)

### SHAP Feature Importance

![SHAP Feature Importance](results/figures/shap_bar_plot.png)

### SHAP Beeswarm

![SHAP Beeswarm](results/figures/shap_beeswarm_plot.png)

### Top Feature Importance

![Top Feature Importance](results/figures/top20_feature_importance.png)

---

# 📁 Repository Structure

```text
IDC-ILC-Gene-Expression-Classification/
│
├── data/
│   └── README.md
│
├── notebooks/
│   ├── 01_IDC_ILC_Exploratory_Analysis.ipynb
│   └── 02_IDC_ILC_Publication_Grade_Analysis.ipynb
│
├── results/
│   ├── feature_selection/
│   ├── shap/
│   ├── figures/
│   └── README.md
│
├── .gitignore
├── LICENSE
└── requirements.txt
```

---

# ♻️ Reproducibility

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the notebooks sequentially:

```text
01_IDC_ILC_Exploratory_Analysis.ipynb
                    ↓
02_IDC_ILC_Publication_Grade_Analysis.ipynb
```

The notebooks contain the analysis workflow covering preprocessing, feature selection, model development, validation, statistical analysis, and explainability.

---

# ⚠️ Limitations

* High-dimensional feature space relative to sample size
* No independent external-cohort validation in the current repository
* Computational feature importance does not establish biological causality
* Generalization requires evaluation on additional cohorts
* Models are research prototypes and **not clinical diagnostic systems**

---

# 🔭 Future Research

Future extensions include:

**External validation · Multi-cohort analysis · Nested validation · Multi-omics integration · Pathway-level analysis · Feature-selection stability · Explainability stability · Independent biological validation**

The broader research direction is to investigate how **interpretable AI can support computational analysis of clinically relevant molecular patterns**.

---

# 👨‍🔬 Researcher

### Vrund Mirani

**AI Healthcare Researcher · BTech CSE (Data Science)**

Research focus:

**Artificial Intelligence · Machine Learning · Computational Biology · Biomedical Data Analysis · Explainable AI**

---

## ⚕️ Disclaimer

This repository is intended for **research and educational purposes**.

The models, predictions, feature rankings, and explainability analyses presented here are not clinical diagnostic systems and should not be used for medical diagnosis or treatment decisions.

---

<p align="center">
  <strong>🧬 AI × Healthcare × Computational Biology</strong>
  <br>
  <sub>Interpretable machine learning for biomedical research.</sub>
</p>
