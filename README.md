# Predicting Customer Churn - BCG-_**X**_

This project is part of my virtual internship at **BCG-X** as a Data Science Intern, where I developed a data-driven solution to predict customer churn. The project focuses on analyzing customer data, engineering meaningful features, and training a machine learning model to identify customers at risk of churn.

## Project Overview

Churn prediction is a critical task in many industries, particularly for utility companies looking to retain customers. By identifying customers likely to churn, proactive strategies can be implemented to improve retention rates.

### Key Components

1. **Data Analysis**
   - Performed exploratory data analysis (EDA) to uncover trends and patterns.
   - Used visualizations to highlight customer behaviors and features impacting churn.

2. **Feature Engineering**
   - Created new features based on domain insights, such as pricing trends and seasonality effects.
   - Engineered innovative features like off-peak energy differences between December and January, rolling averages, and variability measures.

3. **Model Training**
   - Trained a **Random Forest Classifier** for churn prediction.
   - Used **Optuna** for hyperparameter tuning to optimize model performance.

4. **Evaluation**
   - Evaluated the model using **precision**, **recall**,**F1-score**, and **ROC-AUC** metrics.
   - Conducted feature importance analysis to interpret the model.

## Project Structure

```text
├── data/                       # Raw and processed datasets
├── notebooks/                  # Jupyter notebooks for EDA and feature engineering
├── src/                        # Python scripts for data processing, feature engineering, and modeling
│   ├── data_processing.py      # Data preprocessing pipeline
│   ├── feature_engineering.py  # Feature engineering functions
│   ├── modeling.py             # Model training and evaluation
├── dependencies.py             # Script to install project dependencies
├── README.md                   # Project documentation
└── visuals/                    # Visualizations and plots
```

## Installation

To set up the project environment, clone the repository and run the following:

```bash
py dependencies.py
```

This will install all the required libraries.

## Usage

1. **Prepare the Data**
   - Place the dataset in the `data/` directory.
   - Run the preprocessing script to clean and transform the data.

2. **Feature Engineering**
   - Use the feature engineering module to generate new features.

3. **Model Training**
   - Train the model using the scripts in `src/modeling.py`.
   - Use Optuna for hyperparameter tuning.

4. **Visualizations**
   - Check the `visuals/` directory for key plots illustrating insights and results.

## Next Steps

- Adding additional ML models (e.g., Gradient Boosting, XGBoost) for comparison.
- Deploying the model using a REST API for real-time churn predictions.
- Exploring SHAP values for enhanced model interpretability.

## Contribution

Feel free to open an issue or submit a pull request if you have suggestions or improvements.
