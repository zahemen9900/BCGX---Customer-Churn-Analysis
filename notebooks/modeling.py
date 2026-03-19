#!/usr/bin/env python3
"""Counterpart script for modeling.ipynb."""

# %% Cell 1
import warnings
warnings.filterwarnings("ignore")

# %% Cell 2
import os, sys
import optuna
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

sys.path.append('../src')
from utils import save_plot
from models import ChurnPredictor, LightGBMChurnPredictor
# %matplotlib inline

# Set plot style
sns.set_color_codes('deep')
sns.set_theme(style='whitegrid', context='notebook')

#plot presets
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)
plot_params = dict(
    color="0.75",
    style=".-",
    markeredgecolor="0.25",
    markerfacecolor="0.25",
    legend=False,
)
# %config InlineBackend.figure_format = 'retina'
# %matplotlib inline

# %% Cell 3
script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, ".."))
data_dir = os.path.join(repo_root, "data")
df = pd.read_csv(os.path.join(data_dir, "data_for_predictions.csv"))
df.drop(columns=["Unnamed: 0"], inplace=True)
df.head()

# %% Cell 4
df.info(memory_usage = 'deep')

# %% Cell 5
# Make a copy of our data
train_df = df.copy(deep=True)

# Separate target variable from independent variables
y = df['churn']
X = df.drop(columns=['id', 'churn'])
print(X.shape)
print(y.shape)

# %% Cell 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# %% Cell 7
# Add model training in here!
model_params = dict(
    n_estimators = 100, max_depth = 4, n_jobs = -1, random_state = 1234
)
baseline_model = RandomForestClassifier(**model_params) # Add parameters to the model!
baseline_model.fit(X_train, y_train) # Complete this method call!

# %% Cell 8
y_pred = baseline_model.predict(X_test)
report = classification_report(y_test, y_pred)
print("Classification Report:\n")

print(report)

# %% Cell 9
predictor_0 = ChurnPredictor(n_estimators=100, test_size=0.25)

# Train model (automatically handles data splitting and preprocessing)
cv_results = predictor_0.train(df)

# %% Cell 10
cv_results

# %% Cell 11

# Evaluate model on test set
test_results = predictor_0.evaluate()

print("\nClassification Report:")
print(pd.DataFrame(test_results['classification_report']).transpose())

print("\nConfusion Matrix:")
print(np.array(test_results['confusion_matrix']))

print(f"\nROC-AUC Score: {test_results['roc_auc']:.3f}")

print("\nClass Distribution:")
print("Original Training Set:", test_results['class_distribution']['train_original'])
print("Resampled Training Set:", test_results['class_distribution']['train_resampled'])
print("Test Set:", test_results['class_distribution']['test'])

print("\nTop 5 Most Important Features:")
for feature in test_results['feature_importance'][:5]:
    print(f"{feature['feature']}: {feature['importance']:.4f}")

# %% Cell 12
predictor_0.plt_feature_importance(top_n=15)


# %% Cell 13
# Initialize and train model
predictor_1 = LightGBMChurnPredictor(
    sampling_method='adasyn',
    use_gpu = True
)
predictor_1

# %% Cell 14
# pipeline.train_final_model()    # Train the final model with the best parameters
# pipeline.evaluate_model()       # Evaluate the model on the test set
# pipeline.plot_confusion_matrix()  # Plot the confusion matrix
# pipeline.visualize_study()      # Visualize the study and model feature importances

# %% Cell 15
predictor_1.train(df, n_trials=30)  # Run the optimization with 50 trials

# %% Cell 16
test_results = predictor_1.evaluate()

# %% Cell 17
print("Classification Report:")
print(pd.DataFrame(test_results['classification_report']).transpose())

print("\nConfusion Matrix:")
print(np.array(test_results['confusion_matrix']))

print(f"\nROC-AUC Score: {test_results['roc_auc']:.3f}")

print("\nClass Distribution:")
print("Original Training Set:", test_results['class_distribution']['train'])
print("Test Set:", test_results['class_distribution']['test'])

# %% Cell 18
predictor_1.plot_optimization_history()  # Train the final model with the best parameters

# %% Cell 19
predictor_1.plot_param_importances()

# %% Cell 20
predictor_1.plot_feature_importance(n = 10,save=True)

# %% Cell 21
# Plot ROC curve
roc_fig = predictor_1.plot_roc_curve(figsize=(10, 6), save=True)
# roc_fig.savefig('roc_curve.png')  # optional: save the plot

# Plot Precision-Recall curve
pr_fig = predictor_1.plot_precision_recall_curve(figsize=(10, 6), save=True)
# pr_fig.savefig('pr_curve.png')  # optional: save the plot

# %% Cell 22
predictor_1.save_best_model()
