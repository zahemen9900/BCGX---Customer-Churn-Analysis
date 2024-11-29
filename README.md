# Predicting Customer Churn - BCG _**X**_

This repository showcases the work I completed during my virtual internship at **BCG X** _(British Consulting Group)_ as a Data Science Intern. The goal of this project was to design a data-driven solution to predict customer churn, empowering businesses to proactively address customer retention challenges.

---

## 🚀 Project Overview

**Customer churn** prediction is a critical component for industries, especially utilities, seeking to retain customers in competitive markets. By identifying customers likely to churn, organizations can implement targeted strategies to boost retention and reduce operational costs.

---

### 🔑 Key Features

#### 1. **Data Analysis**
- Conducted extensive **exploratory data analysis (EDA)** to uncover trends and patterns.
- Leveraged visualizations to highlight customer behaviors and pinpoint features influencing churn.

#### 2. **Feature Engineering**
- Created innovative features based on domain insights, including:
  - **Pricing trends** (e.g., off-peak energy differences).
  - **Rolling averages** for capturing seasonality effects.
  - Variability measures for detecting consumption shifts.
- Engineered predictive features such as **tenure calculations** and **contract proximity buckets**.

#### 3. **Model Development**
- Implemented a **Random Forest Classifier** to predict churn with high accuracy.
- Optimized model performance using **Optuna** for hyperparameter tuning.
- Assessed model reliability through **precision**, **recall**, **F1-score**, and **ROC-AUC** metrics.

#### 4. **Model Interpretation**
- Analyzed feature importance to identify the most influential predictors.
- Planned for enhanced interpretability with **SHAP** values (next steps).

---

## 🗂 Project Structure

```plaintext
├── data/                       # Raw and processed datasets
│   ├── raw/                    # Original data files
│   └── processed/              # Cleaned and feature-enriched datasets
├── notebooks/                  # Jupyter notebooks for EDA and feature engineering
├── src/                        # Python scripts for data processing and modeling
│   ├── preprocessing.py        # Data preprocessing functions
│   ├── feat_engineering.py     # Feature engineering functions
│   ├── utils.py                # Utility functions for project automation
├── visuals/                    # Visualizations and plots
├── install_dependencies.py     # Script to set up the project environment
├── README.md                   # Project documentation
└── tree.txt                    # File structure of the repository
```

---

## ⚙️ Installation

To set up the environment and install required libraries, clone this repository and run:

```bash
py install_dependencies.py
```

---

## 💻 Usage

### 1. **Prepare the Data**
- Place raw datasets in the `data/raw/` directory.
- Run `src/preprocessing.py` to clean and transform the data.

### 2. **Feature Engineering**
- Execute `src/feat_engineering.py` to generate additional predictive features.

### 3. **Model Training**
- Train and evaluate models using the scripts in `src/` or Jupyter notebooks in `notebooks/`.
- Use **Optuna** for hyperparameter tuning.

### 4. **Visualizations**
- Explore insightful plots in the `visuals/` directory.

---

## 🛠 Next Steps

1. **Model Comparisons**
   - Experiment with advanced models like **Gradient Boosting**, **XGBoost**, and **LightGBM**.

2. **Deployment**
   - Develop a REST API for real-time churn prediction.

3. **Interpretability**
   - Use **SHAP** values to improve model explainability and enhance business insights.

---

## 🤝 Contribution

Contributions are welcome! Feel free to:
- Open an **issue** for bug reports or feature suggestions.
- Submit a **pull request** to improve the project.

---

## 🌟 Acknowledgments

A special thanks to the **BCG X** team for providing this opportunity and invaluable guidance throughout the project.

