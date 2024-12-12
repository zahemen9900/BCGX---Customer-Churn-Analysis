import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, classification_report, confusion_matrix,
                           ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN
from optuna.visualization import plot_optimization_history, plot_param_importances
from sklearn.base import BaseEstimator, ClassifierMixin
import sys
sys.path.append('../src')
from utils import save_plot

class LightGBMChurnPredictor(ClassifierMixin, BaseEstimator):
    def __init__(self, test_size=0.2, random_state=42, sampling_method='none', use_gpu=True):
        """
        Initialize the advanced churn predictor using LightGBM
        
        Parameters:
        -----------
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random state for reproducibility
        sampling_method : str
            Sampling method to use: 'none', 'smote', 'adasyn', or 'class_weight'
        use_gpu : bool
            Whether to use GPU acceleration for training (will fallback to CPU if GPU is not available)
        """
        self.test_size = test_size
        self.random_state = random_state
        self.sampling_method = sampling_method
        self.use_gpu = use_gpu
        self.feature_names = None
        self.best_params = None
        self.model = None
        self.scaler = StandardScaler()
        self.study = None  # Store Optuna study
        self.is_fitted_ = False
        
        # Check GPU availability
        if self.use_gpu:
            try:
                # Create a larger test dataset
                X = np.random.rand(100, 10)  # 100 samples, 10 features
                y = np.random.randint(0, 2, 100)
                train_data = lgb.Dataset(X, label=y)
                gpu_params = {
                    'dSevice': 'gpu',
                    'gpu_platform_id': 0,
                    'gpu_device_id': 0,
                    'min_data_in_bin': 1,  # Reduce minimum data requirements for test
                    'min_data_in_leaf': 1,
                    'verbose': -1
                }
                lgb.train(gpu_params, train_data, num_boost_round=1)
                print("GPU is available and will be used for training")
            except Exception as e:
                print(f"GPU initialization failed: {str(e)}")
                print("Falling back to CPU training")
                self.use_gpu = False
        
    def _get_base_params(self):
        """Get base LightGBM parameters including GPU configuration if available"""
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss', # use recall for optimization, `binary_logloss` can also be used
            'verbosity': -1,
            'boosting_type': 'gbdt',  # use dart for faster training, use gbdt for better accuracy
            'deterministic': True,  # for reproducibility
        }
        
        if self.use_gpu:
            params.update({
                'device': 'cuda',
                'gpu_platform_id': 0,
                'gpu_device_id': 0,
                'gpu_use_dp': True,  # use double precision for better accuracy
                'force_row_wise': True  # better accuracy with GPU
            })
            
        return params

    def _prepare_data(self, data):
        """Prepare features and target from the input data"""
        y = data['churn']
        X = data.drop(['id', 'churn'], axis=1)
        self.feature_names = X.columns.tolist()
        return X, y
    
    def _objective(self, trial, X, y, n_splits=5):
        """Optuna objective function for hyperparameter optimization"""
        # Get base parameters including GPU configuration
        params = self._get_base_params()
        
        # Add hyperparameters to be optimized
        params_to_tune = {
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
            'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
            'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
            'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
            'scoring': 'recall',
            'eval_metric': 'f1',
        }

        params.update(params_to_tune)
        
        # Add GPU-specific parameters if GPU is available
        if self.use_gpu:
            params.update({
                'max_bin': trial.suggest_int('max_bin', 255, 510),
                'min_data_in_bin': trial.suggest_int('min_data_in_bin', 3, 100)
            })
        
        # Add class weights if using class_weight sampling method
        if self.sampling_method == 'class_weight':
            weight_0 = trial.suggest_uniform('weight_0', 0.1, 1.0)
            weight_1 = trial.suggest_uniform('weight_1', 1.0, 10.0)
            params['class_weight'] = {0: weight_0, 1: weight_1}
        
        # Initialize cross-validation
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = []
        
        # Perform cross-validation
        for train_idx, val_idx in skf.split(X, y):
            X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
            y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_fold_train_scaled = self.scaler.fit_transform(X_fold_train)
            X_fold_val_scaled = self.scaler.transform(X_fold_val)
            
            # Apply sampling method if specified
            if self.sampling_method == 'smote':
                sampler = SMOTE(random_state=self.random_state)
                X_resampled, y_resampled = sampler.fit_resample(X_fold_train_scaled, y_fold_train)
            elif self.sampling_method == 'adasyn':
                sampler = ADASYN(random_state=self.random_state)
                X_resampled, y_resampled = sampler.fit_resample(X_fold_train_scaled, y_fold_train)
            else:
                X_resampled, y_resampled = X_fold_train_scaled, y_fold_train
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_resampled, label=y_resampled)
            val_data = lgb.Dataset(X_fold_val_scaled, label=y_fold_val, reference=train_data)
            
            try:
                # Train model with error handling
                callbacks = [lgb.early_stopping(stopping_rounds=10), lgb.log_evaluation(period=0)]
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=callbacks,
                    num_boost_round=1000
                )
                
                # Make predictions
                y_pred = model.predict(X_fold_val_scaled) > 0.5
                scores.append(f1_score(y_fold_val, y_pred))
            except Exception as e:
                print(f"Warning: Training failed with error: {str(e)}")
                print("Retrying with CPU...")
                # Fallback to CPU if GPU fails
                self.use_gpu = False
                params = self._get_base_params()  # Get updated params without GPU
                params.update(params_to_tune)
                # Retry training with CPU
                model = lgb.train(
                    params,
                    train_data,
                    valid_sets=[val_data],
                    callbacks=callbacks,
                    num_boost_round=1000
                )
                
                y_pred = model.predict(X_fold_val_scaled) > 0.5
                scores.append(f1_score(y_fold_val, y_pred))
        
        return np.mean(scores)
    
    def train(self, data, n_trials=100, n_splits=5):
        """
        Train the model using Optuna for hyperparameter optimization
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe containing features, 'id', and 'churn' columns
        n_trials : int
            Number of optimization trials
        n_splits : int
            Number of cross-validation splits
            
        Returns:
        --------
        dict
            Best parameters and optimization results
        """
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        
        # Create and run optimization study
        self.study = optuna.create_study(direction='maximize')
        self.study.optimize(
            lambda trial: self._objective(trial, self.X_train, self.y_train, n_splits),
            n_trials=n_trials
        )
        
        # Store best parameters and add GPU configuration
        self.best_params = self._get_base_params()
        self.best_params.update(self.study.best_params)
        
        # Extract num_boost_round from best params
        num_boost_round = self.best_params.pop('num_boost_round', 1000)
        
        # Train final model with best parameters
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        
        if self.sampling_method in ['smote', 'adasyn']:
            sampler = SMOTE(random_state=self.random_state) if self.sampling_method == 'smote' else ADASYN(random_state=self.random_state)
            X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, self.y_train)
        else:
            X_resampled, y_resampled = X_train_scaled, self.y_train
        
        train_data = lgb.Dataset(X_resampled, label=y_resampled)
        
        # Train final model
        callbacks = [lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)]
        self.model = lgb.train(
            self.best_params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data],
            callbacks=callbacks
        )
        
        # Set the fitted flag
        self.is_fitted_ = True
        
        return {
            'best_params': {**self.best_params, 'num_boost_round': num_boost_round},
            'best_score': self.study.best_value,
            'optimization_history': self.study.trials_dataframe()
        }
    
    def plot_optimization_history(self, figsize=(10, 6)):
        """Plot Optuna's optimization history"""
        if self.study is None:
            raise ValueError("No optimization study available. Call train() first.")
        
        fig = plot_optimization_history(self.study)
        fig.update_layout(
            width=figsize[0]*100,
            height=figsize[1]*100,
            title="Optimization History",
            xaxis_title="Trial",
            yaxis_title="F1 Score"
        )
        return fig
    
    def plot_param_importances(self, figsize=(10, 6)):
        """Plot Optuna's parameter importances"""
        if self.study is None:
            raise ValueError("No optimization study available. Call train() first.")
        
        fig = plot_param_importances(self.study)
        fig.update_layout(
            width=figsize[0]*100,
            height=figsize[1]*100,
            title="Hyperparameter Importances",
            xaxis_title="Importance",
            yaxis_title="Hyperparameter"
        )
        return fig
    
    def evaluate(self):
        """Generate comprehensive evaluation report"""
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get predictions
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled) > 0.5
        y_pred_proba = self.model.predict(X_test_scaled)
        
        # Generate evaluation report
        evaluation = {
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'feature_importance': self.get_feature_importance().to_dict('records')
        }
        
        # Add class distribution information
        evaluation['class_distribution'] = {
            'train': pd.Series(self.y_train).value_counts().to_dict(),
            'test': pd.Series(self.y_test).value_counts().to_dict()
        }
        
        return evaluation
    
    def plot_confusion_matrix(self, figsize=(10, 8), save=False):
        """Plot confusion matrix"""
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred = self.model.predict(X_test_scaled) > 0.5
        
        fig, ax = plt.subplots(figsize=figsize)
        cm_display = ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            y_pred,
            display_labels=['Not Churned', 'Churned'],
            cmap='Blues',
            ax=ax
        )
        
        plt.title('Confusion Matrix', pad=20)
        plt.grid(False)
        
        if save:
            save_plot(filename='confusion_matrix.png')
        return fig
    
    def get_feature_importance(self):
        """Get and visualize feature importance scores"""
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get feature importance scores
        importance = self.model.feature_importance(importance_type='gain')
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def predict(self, data):
        """Make predictions on new data"""
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
            
        X = data.drop('id', axis=1) if 'id' in data.columns else data
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing features in input data: {missing_cols}")
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled) > 0.5
    
    def predict_proba(self, data):
        """Get probability predictions"""
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
            
        X = data.drop('id', axis=1) if 'id' in data.columns else data
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing features in input data: {missing_cols}")
        
        X = X[self.feature_names]
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def plot_feature_importance(self, figsize=(12, 8), n = 10, save=False):
        """Plot feature importance"""
        if self.model is None:
            raise ValueError("Model hasn't been trained yet")

        importance_df = self.get_feature_importance()[:n]
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=importance_df, palette='viridis', ax=ax)
        ax.set_title('Feature Importance')
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=8, fontweight='bold')

        if save:
            save_plot(filename='feature_importance.png')
        plt.show()


    def plot_roc_curve(self, figsize=(10, 6), save=False):
        """
        Plot the ROC curve with AUC score
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the ROC curve
        """
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get predictions
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred_proba = self.model.predict(X_test_scaled)
        
        # Calculate ROC curve points
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            save_plot(filename='roc_curve.png')
        return fig
    
    def plot_precision_recall_curve(self, figsize=(10, 6), save=False):
        """
        Plot the Precision-Recall curve with Average Precision score
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            The figure containing the Precision-Recall curve
        """
        if not self.is_fitted_:
            raise ValueError("Model hasn't been trained yet. Call train() first.")
        
        # Get predictions
        X_test_scaled = self.scaler.transform(self.X_test)
        y_pred_proba = self.model.predict(X_test_scaled)
        
        # Calculate Precision-Recall curve points
        from sklearn.metrics import precision_recall_curve, average_precision_score
        precision, recall, _ = precision_recall_curve(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot Precision-Recall curve
        ax.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        
        # Plot baseline
        no_skill = len(self.y_test[self.y_test == 1]) / len(self.y_test)
        ax.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--',
                label=f'Baseline ({no_skill:.3f})')
        
        # Customize plot
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curve')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            save_plot(filename='precision_recall_curve.png')
        return fig

    def save_best_model(self, filepath='src/best_model.pkl'):
        """
        Save the best model from the Optuna study.
        
        Args:
            filepath (str): Path where to save the model. Defaults to 'best_model.pkl'
        
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            if not hasattr(self, 'study') or self.study is None:
                print("No study found. Please train the model first.")
                return False
            
            # Get best trial and parameters
            best_trial = self.study.best_trial
            best_params = best_trial.params
            
            # Create model with best parameters
            best_model = lgb.LGBMClassifier(**best_params)
            best_model.fit(self.X_train, self.y_train)
            
            # Save model using joblib
            import joblib
            joblib.dump(best_model, filepath)
            
            print(f"Best model saved successfully to {filepath}")
            print(f"Best parameters: {best_params}")
            print(f"Best score: {best_trial.value}")
            
            return True
        
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False

    def load_model(self, filepath='src/best_model.pkl'):
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model file
        
        Returns:
            The loaded model or None if loading fails
        """
        try:
            import joblib
            model = joblib.load(filepath)
            print(f"Model loaded successfully from {filepath}")
            return model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None









class ChurnPredictor:
    def __init__(self, n_estimators=100, test_size=0.2, random_state=42):
        """
        Initialize the churn predictor
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the random forest
        test_size : float
            Proportion of the dataset to include in the test split
        random_state : int
            Random state for reproducibility
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1  # Use all available cores
        )
        self.scaler = StandardScaler()
        self.test_size = test_size
        self.random_state = random_state
        self.feature_names = None
        self.importance_df = None
        
    def _prepare_data(self, data):
        """
        Prepare features and target from the input data
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe containing 'churn' as target and 'id' column
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        # Separate target
        y = data['churn']
        
        # Remove id and target columns to get features
        X = data.drop(['id', 'churn'], axis=1)
        
        self.feature_names = X.columns.tolist()
        return X, y
        
    def preprocess_data(self, X):
        """Scale the features"""
        return self.scaler.transform(X)
    
    def train(self, data, n_splits=5):
        """
        Train the model using StratifiedKFold and SMOTE for handling class imbalance.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input dataframe containing features, 'id', and 'churn' columns
        n_splits : int
            Number of folds for cross-validation
            
        Returns:
        --------
        dict
            Cross-validation metrics
        """
        # Prepare data
        X, y = self._prepare_data(data)
        
        # Split data into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y
        )
        
        # Initialize StratifiedKFold and SMOTE
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        smote = SMOTE(random_state=self.random_state)
        
        # Initialize metrics storage
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'roc_auc': []
        }
        
        # Store best model info
        best_f1 = -1
        self.best_model = None
        self.best_scaler = None
        
        # Perform cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(self.X_train, self.y_train), 1):
            # Split data
            X_fold_train = self.X_train.iloc[train_idx]
            X_fold_val = self.X_train.iloc[val_idx]
            y_fold_train = self.y_train.iloc[train_idx]
            y_fold_val = self.y_train.iloc[val_idx]
            
            # Scale features
            self.scaler.fit(X_fold_train)
            X_fold_train_scaled = self.preprocess_data(X_fold_train)
            X_fold_val_scaled = self.preprocess_data(X_fold_val)
            
            # Apply SMOTE to handle class imbalance
            X_fold_train_resampled, y_fold_train_resampled = smote.fit_resample(X_fold_train_scaled, y_fold_train)
            
            # Train model
            self.model.fit(X_fold_train_resampled, y_fold_train_resampled)
            
            # Make predictions
            y_pred = self.model.predict(X_fold_val_scaled)
            y_pred_proba = self.model.predict_proba(X_fold_val_scaled)[:, 1]
            
            # Calculate metrics
            f1 = f1_score(y_fold_val, y_pred)
            metrics['accuracy'].append(accuracy_score(y_fold_val, y_pred))
            metrics['precision'].append(precision_score(y_fold_val, y_pred))
            metrics['recall'].append(recall_score(y_fold_val, y_pred))
            metrics['f1'].append(f1)
            metrics['roc_auc'].append(roc_auc_score(y_fold_val, y_pred_proba))
            
            # Save best model
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = self.model
                self.best_scaler = self.scaler
        
        # Calculate mean and std for each metric
        results = {}
        for metric in metrics:
            results[f'{metric}_mean'] = np.mean(metrics[metric])
            results[f'{metric}_std'] = np.std(metrics[metric])
            
        return results
    
    def evaluate(self):
        """
        Evaluate the model on the test set and generate a comprehensive evaluation report.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics and reports
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Model hasn't been trained yet. Call train() first.")
            
        # Scale the data
        self.scaler.fit(self.X_train)
        X_train_scaled = self.preprocess_data(self.X_train)
        X_test_scaled = self.preprocess_data(self.X_test)
        
        # Apply SMOTE to training data
        smote = SMOTE(random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, self.y_train)
        
        # Train the model
        self.model.fit(X_train_resampled, y_train_resampled)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Generate evaluation report
        evaluation = {
            'classification_report': classification_report(self.y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(self.y_test, y_pred_proba),
            'feature_importance': self.get_feature_importance().to_dict('records')
        }
        
        # Add class distribution information
        evaluation['class_distribution'] = {
            'train_original': pd.Series(self.y_train).value_counts().to_dict(),
            'train_resampled': pd.Series(y_train_resampled).value_counts().to_dict(),
            'test': pd.Series(self.y_test).value_counts().to_dict()
        }
        
        return evaluation
    
    def plot_confusion_matrix(self, figsize=(10, 8)):
        """
        Plot confusion matrix using sklearn's ConfusionMatrixDisplay
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
        """
        if not hasattr(self, 'X_test'):
            raise ValueError("Model hasn't been trained yet. Call train() first.")
            
        # Get predictions
        X_test_scaled = self.preprocess_data(self.X_test)
        y_pred = self.model.predict(X_test_scaled)
        
        # Create confusion matrix display
        fig, ax = plt.subplots(figsize=figsize)
        cm_display = ConfusionMatrixDisplay.from_predictions(
            self.y_test,
            y_pred,
            display_labels=['Not Churned', 'Churned'],
            cmap='Blues',
            ax=ax
        )
        
        # Customize plot
        plt.title('Confusion Matrix', pad=20)
        plt.grid(False)
        
        return fig
    
        
    def get_feature_importance(self):
        """Get feature importance scores"""
        if self.feature_names is None:
            raise ValueError("Model hasn't been trained yet")
            
        importance = self.model.feature_importances_
        self.importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        return self.importance_df

    def plt_feature_importance(self, top_n=20, figsize=(12, 8)):
        """
        Get and visualize feature importance scores using seaborn
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        tuple
            Figure and DataFrame containing feature importance scores
        """
        if self.importance_df is None:
            self.importance_df = self.get_feature_importance()

        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot top N features
        plot_df = self.importance_df.head(top_n)
        sns.barplot(
            data=plot_df,
            y='feature',
            x='importance',
            ax=ax,
            palette='cividis'
        )
        
        # Customize plot
        ax.set_title(f'Top {top_n} Most Important Features', pad=20)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f')
        # Adjust layout
        plt.tight_layout()
        
    
    def predict(self, data):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        data : pd.DataFrame
            New data to make predictions on (should have same columns as training data)
        """
        # Remove id if present
        X = data.drop('id', axis=1) if 'id' in data.columns else data
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing features in input data: {missing_cols}")
            
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, data):
        """
        Get probability predictions on new data
        
        Parameters:
        -----------
        data : pd.DataFrame
            New data to make predictions on (should have same columns as training data)
        """
        # Remove id if present
        X = data.drop('id', axis=1) if 'id' in data.columns else data
        
        # Ensure all feature columns are present
        missing_cols = set(self.feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing features in input data: {missing_cols}")
            
        # Reorder columns to match training data
        X = X[self.feature_names]
        
        X_scaled = self.preprocess_data(X)
        return self.model.predict_proba(X_scaled)
