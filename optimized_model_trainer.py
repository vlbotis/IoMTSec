import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
import joblib
import gc
import multiprocessing
from pathlib import Path
import os

# Import regular scikit-learn components
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel

# Import faster ML libraries
import lightgbm as lgb
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

# Imbalanced learning imports
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class OptimizedModelTrainer:
    """
    Optimized class for training and evaluating models with faster algorithms
    and more efficient processing for large datasets.
    """
    
    def __init__(self, n_jobs=None, batch_size=100000, max_runtime_hours=4, use_gpu=False):
        """
        Initialize the OptimizedModelTrainer.
        
        Args:
            n_jobs (int): Number of parallel jobs (defaults to CPU count - 1)
            batch_size (int): Size of batches for processing large datasets
            max_runtime_hours (float): Maximum runtime for training in hours
            use_gpu (bool): Whether to use GPU acceleration if available
        """
        # Auto-detect optimal number of jobs
        if n_jobs is None:
            self.n_jobs = min(multiprocessing.cpu_count() - 1, 8)
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
            
        self.batch_size = batch_size
        self.max_runtime_hours = max_runtime_hours
        self.use_gpu = use_gpu
        self.start_time = time.time()
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize classifiers with optimized parameters
        self.classifiers = self._initialize_classifiers()
        
        # Set up hyperparameter distributions for randomized search
        self.param_distributions = self._define_param_distributions()
        
        if self.max_runtime_hours:
            self.logger.info(f"Maximum runtime set to {self.max_runtime_hours} hours")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_filename = f'optimized_trainer_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("OptimizedModelTrainer initialized")
    
    def _initialize_classifiers(self):
        """
        Initialize classifiers with optimized choices for speed.
        
        Returns:
            dict: Dictionary of initialized classifiers
        """
        device_type = 'gpu' if self.use_gpu else 'cpu'
        
        return {
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=15,
                learning_rate=0.1,
                num_leaves=31,
                device=device_type,
                n_jobs=self.n_jobs,
                random_state=42,
                verbose=1
            ),
            'sgd_classifier': SGDClassifier(
                loss='log_loss',  # For logistic regression equivalent
                max_iter=100,
                tol=1e-3,
                alpha=0.0001,
                l1_ratio=0.15,
                random_state=42,
                n_jobs=self.n_jobs,
                verbose=1
            ),
            'hist_gradient_boosting': HistGradientBoostingClassifier(
                max_iter=100,
                max_depth=15,
                learning_rate=0.1,
                l2_regularization=0.0,
                random_state=42,
                verbose=1
            )
        }
    
    def _define_param_distributions(self):
        """
        Define hyperparameter distributions for randomized search.
        
        Returns:
            dict: Dictionary of parameter distributions for each classifier
        """
        return {
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 15, 20, -1],  # -1 means no limit
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'num_leaves': [31, 63, 127],
                'min_child_samples': [5, 10, 20, 50]
            },
            'sgd_classifier': {
                'alpha': [0.0001, 0.001, 0.01],
                'l1_ratio': [0.1, 0.15, 0.3, 0.5, 0.7],
                'penalty': ['l2', 'l1', 'elasticnet'],
                'max_iter': [50, 100, 200]
            },
            'hist_gradient_boosting': {
                'max_iter': [50, 100, 200],
                'max_depth': [5, 10, 15, 20, None],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'l2_regularization': [0.0, 0.1, 1.0]
            }
        }
    
    def prepare_balanced_data(self, X_train, y_train, class_names, max_per_class=50000):
        """
        Efficiently balance the dataset using a combination of undersampling 
        and oversampling techniques.
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            class_names (list): Class names corresponding to label indices
            max_per_class (int): Maximum samples per class
            
        Returns:
            tuple: (X_balanced, y_balanced) - Balanced training data
        """
        self.logger.info(f"Balancing dataset (max {max_per_class} samples per class)")
        
        # Get class distribution
        class_counts = np.bincount(y_train)
        self.logger.info("Original class distribution:")
        for i, count in enumerate(class_counts):
            self.logger.info(f"  {class_names[i]}: {count}")
        
        # Calculate target sizes for each class (more balanced approach)
        # For majority classes, undersample to max_per_class
        # For minority classes, keep original or oversample to min_per_class
        min_per_class = min(10000, max_per_class // 5)  # Minimum target size
        
        majority_classes = {}
        minority_classes = {}
        
        for i, count in enumerate(class_counts):
            if count > max_per_class:
                majority_classes[i] = max_per_class
            elif count < min_per_class:
                minority_classes[i] = min_per_class
        
        self.logger.info(f"Classes to undersample: {list(majority_classes.keys())}")
        self.logger.info(f"Classes to oversample: {list(minority_classes.keys())}")
        
        # If we have majority classes to undersample
        if majority_classes:
            self.logger.info("Undersampling majority classes...")
            undersampler = RandomUnderSampler(
                sampling_strategy=majority_classes,
                random_state=42
            )
            X_temp, y_temp = undersampler.fit_resample(X_train, y_train)
        else:
            X_temp, y_temp = X_train, y_train
        
        # If we have minority classes to oversample
        if minority_classes:
            self.logger.info("Oversampling minority classes...")
            # Ensure k_neighbors doesn't exceed possible values
            min_class_samples = min([np.sum(y_temp == cls) for cls in minority_classes.keys()])
            k_neighbors = min(5, min_class_samples - 1) if min_class_samples > 1 else 1
            
            oversampler = SMOTE(
                sampling_strategy=minority_classes,
                k_neighbors=k_neighbors,
                random_state=42,
               # n_jobs=self.n_jobs
            )
            X_balanced, y_balanced = oversampler.fit_resample(X_temp, y_temp)
        else:
            X_balanced, y_balanced = X_temp, y_temp
        
        # Log final class distribution
        balanced_counts = np.bincount(y_balanced)
        self.logger.info("Balanced class distribution:")
        for i, count in enumerate(balanced_counts):
            if i < len(class_names):
                self.logger.info(f"  {class_names[i]}: {count}")
        
        # Force garbage collection to free memory
        del X_temp, y_temp
        gc.collect()
        
        return X_balanced, y_balanced
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, label_encoder, 
                          perform_tuning=True, max_per_class=50000):
        """
        Train and evaluate models with efficient hyperparameter tuning.
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
            label_encoder (LabelEncoder): Encoder used for transforming labels
            perform_tuning (bool): Whether to perform hyperparameter tuning
            max_per_class (int): Maximum samples per class after balancing
            
        Returns:
            dict: Dictionary of evaluation results
        """
        self.logger.info(f"Training and evaluating models with hyperparameter tuning={perform_tuning}")
        self.results = {}
        
        # Store label encoder for reporting
        self.label_encoder = label_encoder
        
        # Convert y_test to 1D if needed
        if len(y_test.shape) > 1:
            y_test = y_test.ravel()
        
        # Balance the dataset efficiently
        X_balanced, y_balanced = self.prepare_balanced_data(
            X_train, y_train, label_encoder.classes_, max_per_class)
        
        # Free original training data to save memory
        del X_train, y_train
        gc.collect()
        
        # Train each model in order
        for name, clf in self.classifiers.items():
            # Skip if runtime limit exceeded
            if self.max_runtime_hours and self._check_runtime() == False:
                self.logger.warning(f"Skipping {name} due to runtime constraints")
                continue
                
            self.logger.info(f"\n{'='*20}\nTraining {name}...")
            try:
                start_time = time.time()
                
                if perform_tuning:
                    best_model = self._tune_hyperparameters(
                        name, clf, X_balanced, y_balanced)
                else:
                    # Direct training without tuning
                    self.logger.info(f"Training {name} without hyperparameter tuning")
                    model = clone(clf)
                    best_model = model.fit(X_balanced, y_balanced)
                
                # Make predictions in batches for large datasets
                y_pred, y_pred_proba = self._predict_in_batches(best_model, X_test)
                
                # Calculate metrics
                training_time = time.time() - start_time
                report = self._generate_classification_report(y_test, y_pred)
                
                # Store results
                self.results[name] = {
                    'model': best_model,
                    'training_time': training_time,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'report': report,
                    'accuracy': report['accuracy'],
                    'macro_f1': report['macro avg']['f1-score']
                }
                
                # Log results
                self.logger.info(f"\nTraining time: {training_time:.2f} seconds")
                self.logger.info("\nClassification Report:")
                self.logger.info(classification_report(
                    y_test,
                    y_pred,
                    target_names=self.label_encoder.classes_,
                    labels=np.unique(y_pred),
                    zero_division=0
                ))
                
                # Save model checkpoint immediately
                self._save_model_checkpoint(name, best_model, report)
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                self.logger.exception("Exception details:")
                continue
        
        return self.results
    
    def _tune_hyperparameters(self, name, clf, X_train, y_train, n_iter=10):
        """
        Tune hyperparameters using RandomizedSearchCV instead of GridSearchCV.
        
        Args:
            name (str): Name of the classifier
            clf (object): Classifier instance
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            n_iter (int): Number of parameter settings to try
            
        Returns:
            object: Trained model with best parameters
        """
        # Use a subsample for tuning if dataset is very large
        if len(X_train) > 200000:
            self.logger.info(f"Sampling data for hyperparameter tuning")
            indices = np.random.choice(X_train.shape[0], 200000, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('classifier', clf)
        ])
        
        # Setup randomized search with parameter distributions
        param_distributions = {f'classifier__{k}': v 
                              for k, v in self.param_distributions[name].items()}
        
        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=3,
            n_jobs=self.n_jobs,
            scoring='balanced_accuracy',
            verbose=1
        )
        
        # Fit search
        self.logger.info(f"Starting randomized search for {name} with {n_iter} iterations")
        search.fit(X_sample, y_sample)
        
        # Log best parameters
        self.logger.info(f"Best parameters for {name}: {search.best_params_}")
        self.logger.info(f"Best score: {search.best_score_:.4f}")
        
        # If using sample for tuning, retrain on full dataset with best params
        if len(X_train) > 200000:
            self.logger.info(f"Retraining {name} on full dataset with best parameters")
            
            # Clone the best estimator for retraining
            best_model = clone(search.best_estimator_.named_steps['classifier'])
            best_model.fit(X_train, y_train)
            return best_model
        else:
            return search.best_estimator_.named_steps['classifier']
    
    def _predict_in_batches(self, model, X_test):
        """
        Make predictions in batches for large datasets.
        
        Args:
            model (object): Trained model
            X_test (ndarray): Test features
            
        Returns:
            tuple: (y_pred, y_pred_proba) - Predictions and probabilities
        """
        # For small datasets, predict directly
        if len(X_test) <= self.batch_size:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)
            else:
                y_pred_proba = None
            return model.predict(X_test), y_pred_proba
        
        # For large datasets, process in batches
        y_pred = []
        y_pred_proba = []
        
        total_batches = (len(X_test) + self.batch_size - 1) // self.batch_size
        
        for i in range(0, len(X_test), self.batch_size):
            batch_num = i // self.batch_size + 1
            end = min(i + self.batch_size, len(X_test))
            X_batch = X_test[i:end]
            
            self.logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Get predictions
            batch_pred = model.predict(X_batch)
            y_pred.extend(batch_pred)
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    batch_proba = model.predict_proba(X_batch)
                    y_pred_proba.extend(batch_proba)
                except Exception as e:
                    self.logger.warning(f"Could not get probabilities: {str(e)}")
                    y_pred_proba = None
            else:
                y_pred_proba = None
        
        # Convert to numpy arrays
        y_pred = np.array(y_pred)
        if y_pred_proba and len(y_pred_proba) > 0:
            y_pred_proba = np.array(y_pred_proba)
        
        return y_pred, y_pred_proba
    
    def _generate_classification_report(self, y_true, y_pred):
        """
        Generate a classification report with proper label handling.
        
        Args:
            y_true (ndarray): True labels
            y_pred (ndarray): Predicted labels
            
        Returns:
            dict: Classification report as dictionary
        """
        # Ensure all classes are represented
        present_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        
        # Create report
        return classification_report(
            y_true, 
            y_pred,
            target_names=[self.label_encoder.classes_[i] for i in present_classes],
            labels=present_classes,
            zero_division=0,
            output_dict=True
        )
    
    def _check_runtime(self):
        """
        Check if runtime limits are exceeded.
        
        Returns:
            bool: True if limits are not exceeded, False otherwise
        """
        if self.max_runtime_hours:
            current_runtime_hours = (time.time() - self.start_time) / 3600
            if current_runtime_hours > self.max_runtime_hours:
                self.logger.warning(f"Maximum runtime of {self.max_runtime_hours} hours exceeded. Current runtime: {current_runtime_hours:.2f} hours")
                return False
            else:
                self.logger.info(f"Current runtime: {current_runtime_hours:.2f} hours (max: {self.max_runtime_hours} hours)")
        
        return True
    
    def _save_model_checkpoint(self, name, model, report):
        """
        Save model checkpoint immediately after training.
        
        Args:
            name (str): Model name
            model (object): Trained model
            report (dict): Classification report
        """
        # Create directory if it doesn't exist
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Save model with metadata
        checkpoint_path = checkpoint_dir / f"{name}_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        model_data = {
            "model": model,
            "label_encoder": self.label_encoder,
            "report": report,
            "timestamp": datetime.now().isoformat(),
            "accuracy": report['accuracy'],
            "macro_f1": report['macro avg']['f1-score']
        }
        
        joblib.dump(model_data, checkpoint_path)
        self.logger.info(f"Model checkpoint saved: {checkpoint_path}")
    
    def export_to_onnx(self, model_name, features_dim):
        """
        Export the trained model to ONNX format for faster inference.
        
        Args:
            model_name (str): Name of the model to export
            features_dim (int): Number of input features
            
        Returns:
            str: Path to the exported ONNX model
        """
        if model_name not in self.results:
            self.logger.error(f"Model {model_name} not found in results")
            return None
            
        try:
            import onnxmltools
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            model = self.results[model_name]['model']
            
            # Define input type
            initial_type = [('float_input', FloatTensorType([None, features_dim]))]
            
            # Convert model to ONNX
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            # Save model
            onnx_dir = Path("onnx_models")
            onnx_dir.mkdir(exist_ok=True)
            
            onnx_path = onnx_dir / f"{model_name}_model.onnx"
            onnxmltools.utils.save_model(onnx_model, onnx_path)
            
            self.logger.info(f"Model exported to ONNX: {onnx_path}")
            return str(onnx_path)
            
        except ImportError:
            self.logger.error("ONNX conversion requires onnxmltools and skl2onnx packages")
            return None
        except Exception as e:
            self.logger.error(f"Error exporting to ONNX: {str(e)}")
            return None