import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier, Pool



class ChurnPredictionPipelineGBM:
    def __init__(self, df, target_col, test_size=0.2, random_state=42):
        """
        Initialize the pipeline with the dataset and target column.
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.study = None
        self.final_model = None

    def _prepare_data(self):
        """
        Prepare the training and test sets.
        """
        X = self.df.drop(columns=[self.target_col, 'id']) if 'id' in self.df.columns.tolist() else self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        return train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def objective(self, trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """

        n_estimators = trial.suggest_int("n_estimators", 150, 400)
        max_depth = trial.suggest_int("max_depth", 5, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        num_leaves = trial.suggest_int("num_leaves", 31, 128)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)  
        subsample = trial.suggest_float("subsample", 0.7, 1.0) 
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0) 
        random_state = 321

        # Instantiate LightGBM model with suggested hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )

        # Perform cross-validation and return the mean accuracy score
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy").mean()
        return score

    def optimize(self, n_trials=50):
        """
        Run the Optuna optimization.
        """
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", self.study.best_params)

    def train_final_model(self):
        """
        Train the final model using the best parameters found by Optuna.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        best_params = self.study.best_params
        self.final_model = lgb.LGBMClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=self.random_state
        )
        self.final_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print the accuracy.
        """
        y_pred = self.final_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test set accuracy:", accuracy)

        print(classification_report(self.y_test, y_pred))

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the model's performance on the test set.
        """
        sns.set_theme(context = 'notebook', style = 'white')
        if self.final_model is None:
            raise ValueError("The final model has not been trained yet. Run the `train_final_model` method first.")
        
        y_pred = self.final_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.final_model.classes_)
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_study(self):
        """
        Visualize the optimization study to understand hyperparameter distributions and results.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
                
        # Plot the optimization history (value of the objective function over trials)
        optuna.visualization.plot_optimization_history(self.study).show()
        # Plot the parameter importance
        optuna.visualization.plot_param_importances(self.study).show()

        # Show feature importances if the model is trained
        if self.final_model is not None:
            importances = self.final_model.feature_importances_
            feature_importances_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            plt.figure(figsize=(14, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df)
            plt.title('Feature Importances from Trained Model')



class ChurnPredictionPipelineGBM_SMOTE:
    def __init__(self, df, target_col, test_size=0.2, random_state=42):
        """
        Initialize the pipeline with the dataset and target column.
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.study = None
        self.final_model = None

    def _prepare_data(self):
        """
        Prepare the training and test sets and apply SMOTE for oversampling the minority class.
        """
        X = self.df.drop(columns=[self.target_col, 'id']) if 'id' in self.df.columns.tolist() else self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Apply SMOTE to the training set to balance the classes
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Return the resampled training set and the original test set
        return X_train_resampled, X_test, y_train_resampled, y_test

    def objective(self, trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """
        n_estimators = trial.suggest_int("n_estimators", 150, 400)
        max_depth = trial.suggest_int("max_depth", 5, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        num_leaves = trial.suggest_int("num_leaves", 31, 128)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)  
        subsample = trial.suggest_float("subsample", 0.7, 1.0) 
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.7, 1.0) 
        random_state = 321

        # Instantiate LightGBM model with suggested hyperparameters
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state
        )

        # Perform cross-validation and return the mean accuracy score
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy").mean()
        return score

    def optimize(self, n_trials=50):
        """
        Run the Optuna optimization.
        """
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", self.study.best_params)

    def train_final_model(self):
        """
        Train the final model using the best parameters found by Optuna.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        best_params = self.study.best_params
        self.final_model = lgb.LGBMClassifier(
            n_estimators=best_params["n_estimators"],
            max_depth=best_params["max_depth"],
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            min_child_samples=best_params["min_child_samples"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            random_state=self.random_state
        )
        self.final_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print the accuracy.
        """
        y_pred = self.final_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test set accuracy:", accuracy)

        print(classification_report(self.y_test, y_pred))

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the model's performance on the test set.
        """
        sns.set_theme(context='notebook', style='white')
        if self.final_model is None:
            raise ValueError("The final model has not been trained yet. Run the `train_final_model` method first.")
        
        y_pred = self.final_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.final_model.classes_)
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_study(self):
        """
        Visualize the optimization study to understand hyperparameter distributions and results.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
                
        # Plot the optimization history (value of the objective function over trials)
        optuna.visualization.plot_optimization_history(self.study).show()
        # Plot the parameter importance
        optuna.visualization.plot_param_importances(self.study).show()

        # Show feature importances if the model is trained
        if self.final_model is not None:
            importances = self.final_model.feature_importances_
            feature_importances_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            plt.figure(figsize=(14, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df)
            plt.title('Feature Importances from Trained Model')




























class ChurnPredictionPipelineCatBoost_SMOTE:
    def __init__(self, df, target_col, test_size=0.2, random_state=42):
        """
        Initialize the pipeline with the dataset and target column.
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.study = None
        self.final_model = None

    def _prepare_data(self):
        """
        Prepare the training and test sets and apply SMOTE for oversampling the minority class.
        """
        X = self.df.drop(columns=[self.target_col, 'id']) if 'id' in self.df.columns.tolist() else self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Apply SMOTE to the training set to balance the classes
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Return the resampled training set and the original test set
        return X_train_resampled, X_test, y_train_resampled, y_test

    def objective(self, trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """
        n_estimators = trial.suggest_int("n_estimators", 150, 400)
        max_depth = trial.suggest_int("max_depth", 5, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        depth = trial.suggest_int("depth", 5, 15)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.7, 1.0)

        # Instantiate CatBoost model with suggested hyperparameters
        model = CatBoostClassifier(
            iterations=n_estimators,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            random_state=self.random_state,
            verbose=0,  # Suppress output during training
            class_weights=[1, 5]  # Adjust class weights to handle imbalance
        )

        # Perform cross-validation and return the mean accuracy score
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy").mean()
        return score

    def optimize(self, n_trials=50):
        """
        Run the Optuna optimization.
        """
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", self.study.best_params)

    def train_final_model(self):
        """
        Train the final model using the best parameters found by Optuna.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        best_params = self.study.best_params
        self.final_model = CatBoostClassifier(
            iterations=best_params["n_estimators"],
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            subsample=best_params["subsample"],
            colsample_bylevel=best_params["colsample_bylevel"],
            random_state=self.random_state,
            verbose=0,
            class_weights=[1, 5]  # Handle class imbalance
        )
        self.final_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print the accuracy and additional metrics.
        """
        y_pred = self.final_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test set accuracy:", accuracy)

        # Display classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Display precision-recall and ROC curves
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
        import matplotlib.pyplot as plt
        import numpy as np

        # Precision-recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        roc_auc = roc_auc_score(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the model's performance on the test set.
        """
        sns.set_theme(context='notebook', style='white')
        if self.final_model is None:
            raise ValueError("The final model has not been trained yet. Run the `train_final_model` method first.")
        
        y_pred = self.final_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.final_model.classes_)
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_study(self):
        """
        Visualize the optimization study to understand hyperparameter distributions and results.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        # Plot the optimization history (value of the objective function over trials)
        optuna.visualization.plot_optimization_history(self.study).show()
        # Plot the parameter importance
        optuna.visualization.plot_param_importances(self.study).show()

        # Show feature importances if the model is trained
        if self.final_model is not None:
            importances = self.final_model.get_feature_importance()
            feature_importances_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            plt.figure(figsize=(14, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df)
            plt.title('Feature Importances from Trained Model')
            plt.show()






###########################################################################################

import optuna
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from catboost import CatBoostClassifier, Pool

class ChurnPredictionPipelineCatBoost_SMOTE:
    def __init__(self, df, target_col, test_size=0.2, random_state=42):
        """
        Initialize the pipeline with the dataset and target column.
        """
        self.df = df
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        self.study = None
        self.final_model = None

    def _prepare_data(self):
        """
        Prepare the training and test sets and apply SMOTE for oversampling the minority class.
        """
        X = self.df.drop(columns=[self.target_col, 'id']) if 'id' in self.df.columns.tolist() else self.df.drop(columns=[self.target_col])
        y = self.df[self.target_col]
        
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

        # Apply SMOTE to the training set to balance the classes
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Return the resampled training set and the original test set
        return X_train_resampled, X_test, y_train_resampled, y_test

    def objective(self, trial):
        """
        Objective function for Optuna to optimize hyperparameters.
        """
        n_estimators = trial.suggest_int("n_estimators", 150, 400)
        max_depth = trial.suggest_int("max_depth", 5, 15, log=True)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        depth = trial.suggest_int("depth", 5, 15)
        l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
        subsample = trial.suggest_float("subsample", 0.7, 1.0)
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.7, 1.0)

        # Instantiate CatBoost model with suggested hyperparameters
        model = CatBoostClassifier(
            iterations=n_estimators,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            subsample=subsample,
            colsample_bylevel=colsample_bylevel,
            random_state=self.random_state,
            verbose=0,  # Suppress output during training
            class_weights=[1, 5]  # Adjust class weights to handle imbalance
        )

        # Perform cross-validation and return the mean accuracy score
        score = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring="accuracy").mean()
        return score

    def optimize(self, n_trials=50):
        """
        Run the Optuna optimization.
        """
        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.objective, n_trials=n_trials)
        print("Best hyperparameters:", self.study.best_params)

    def train_final_model(self):
        """
        Train the final model using the best parameters found by Optuna.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        best_params = self.study.best_params
        self.final_model = CatBoostClassifier(
            iterations=best_params["n_estimators"],
            depth=best_params["depth"],
            learning_rate=best_params["learning_rate"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            subsample=best_params["subsample"],
            colsample_bylevel=best_params["colsample_bylevel"],
            random_state=self.random_state,
            verbose=0,
            class_weights=[1, 5]  # Handle class imbalance
        )
        self.final_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """
        Evaluate the trained model on the test set and print the accuracy and additional metrics.
        """
        y_pred = self.final_model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print("Test set accuracy:", accuracy)

        # Display classification report
        print("Classification Report:")
        print(classification_report(self.y_test, y_pred))

        # Display precision-recall and ROC curves
        from sklearn.metrics import precision_recall_curve, auc, roc_auc_score, roc_curve
        import matplotlib.pyplot as plt
        import numpy as np

        # Precision-recall curve
        precision, recall, _ = precision_recall_curve(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        pr_auc = auc(recall, precision)
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.show()

        # ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        roc_auc = roc_auc_score(self.y_test, self.final_model.predict_proba(self.X_test)[:, 1])
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self):
        """
        Plot the confusion matrix for the model's performance on the test set.
        """
        sns.set_theme(context='notebook', style='white')
        if self.final_model is None:
            raise ValueError("The final model has not been trained yet. Run the `train_final_model` method first.")
        
        y_pred = self.final_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.final_model.classes_)
        disp.plot(cmap='viridis', values_format='d')
        plt.title('Confusion Matrix')
        plt.show()

    def visualize_study(self):
        """
        Visualize the optimization study to understand hyperparameter distributions and results.
        """
        if self.study is None:
            raise ValueError("The study has not been optimized yet. Run the `optimize` method first.")
        
        # Plot the optimization history (value of the objective function over trials)
        optuna.visualization.plot_optimization_history(self.study).show()
        # Plot the parameter importance
        optuna.visualization.plot_param_importances(self.study).show()

        # Show feature importances if the model is trained
        if self.final_model is not None:
            importances = self.final_model.get_feature_importance()
            feature_importances_df = pd.DataFrame({
                'feature': self.X_train.columns,
                'importance': importances
            }).sort_values(by='importance', ascending=False)

            plt.figure(figsize=(14, 8))
            sns.barplot(x='importance', y='feature', data=feature_importances_df)
            plt.title('Feature Importances from Trained Model')
            plt.show()
