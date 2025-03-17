"""
IoMT Multi-Classifier - Enhanced Implementation

This module provides classes for analyzing and classifying IoMT network traffic into different attack types.
The implementation focuses on memory efficiency, maintainability, and performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
import pickle
import json
import logging
from tqdm import tqdm
from collections import defaultdict
import multiprocessing
from functools import partial
import gc  # Garbage collector

# Scikit-learn imports
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_score, recall_score
)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight

# Imbalanced learning imports
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns


class DataLoader:
    """
    Class for efficiently loading and preprocessing IoMT network traffic data.
    
    Attributes:
        batch_size (int): Size of data chunks for processing large files
        max_files (int, optional): Maximum number of files to process per class
        logger (Logger): Logger instance for recording operations
        expected_attacks (set): Set of expected attack types
    """
    
    def __init__(self, batch_size=1000000, max_files=None):
        """
        Initialize the DataLoader.
        
        Args:
            batch_size (int): Size of chunks for processing large files
            max_files (int, optional): Maximum number of files to process per attack type
        """
        self.batch_size = batch_size
        self.max_files = max_files
        
        # Setup logging
        self._setup_logging()
        
        # Define expected attack types
        self.expected_attacks = {
            'arp_spoofing', 'benign', 'ddos', 'dos', 
            'mqtt_malformed', 'reconnaissance'
        }
        
        # Define attack patterns for identification
        self.attack_patterns = {
            'ddos': [
                'TCP_IP-DDOS-ICMP', 'TCP_IP-DDOS-SYN', 
                'TCP_IP-DDOS-TCP', 'TCP_IP-DDOS-UDP',
                'MQTT-DDOS'
            ],
            'dos': [
                'TCP_IP-DOS-ICMP', 'TCP_IP-DOS-SYN', 
                'TCP_IP-DOS-TCP', 'TCP_IP-DOS-UDP',
                'MQTT-DOS'
            ],
            'mqtt_malformed': ['MQTT-MALFORMED'],
            'arp_spoofing': ['ARP_SPOOFING'],
            'benign': ['BENIGN'],
            'reconnaissance': ['RECON-', 'RECON_']
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_filename = f'iomt_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("DataLoader initialized")
    
    def identify_attack_type(self, filename):
        """
        Extract attack type from filename using systematic pattern matching.
        
        Args:
            filename (str): Name of the file to identify
        
        Returns:
            str: Identified attack type or 'unknown' if not recognized
        """
        # Normalize the filename to uppercase for consistent matching
        filename = filename.upper()
        
        # Check each attack type's patterns
        for attack_type, patterns in self.attack_patterns.items():
            if any(pattern in filename for pattern in patterns):
                return attack_type
        
        # If no pattern matches, log a warning and return 'unknown'
        self.logger.warning(f"Unknown attack type in filename: {filename}")
        return 'unknown'
    
    def load_csv_in_chunks(self, file_path, attack_type=None):
        """
        Load a CSV file in chunks to manage memory efficiently.
        
        Args:
            file_path (Path): Path to the CSV file
            attack_type (str, optional): Attack type to assign to the data
        
        Returns:
            DataFrame: Loaded data with attack type annotation
        """
        try:
            chunks = []
            for chunk in pd.read_csv(file_path, chunksize=self.batch_size):
                if attack_type is not None:
                    chunk['attack_type'] = attack_type
                chunks.append(chunk)
                
            # Only concatenate after all chunks are loaded to minimize memory usage
            return pd.concat(chunks, ignore_index=True)
        except Exception as e:
            self.logger.error(f"Error loading file {file_path}: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def calculate_train_test_sizes(self, total_files, test_size=0.2):
        """
        Calculate proper number of files for train and test splits.
        
        Args:
            total_files (int): Total number of files available
            test_size (float): Proportion of files to use for testing
        
        Returns:
            tuple: (n_train, n_test) - Number of files for training and testing
        """
        n_test = max(int(round(total_files * test_size)), 1)  # At least 1 test file
        n_train = total_files - n_test
        
        return n_train, n_test
    
    def load_data_with_holdout(self, base_path):
        """
        Load all data files with proper holdout split by attack type.
        
        Args:
            base_path (Path): Base directory containing the data files
        
        Returns:
            tuple: (train_data, holdout_data) - DataFrames for training and holdout sets
        """
        self.logger.info(f"Loading data from {base_path}")
        train_path = base_path / "train"
        
        # Categorize files by attack type
        files_by_attack = defaultdict(list)
        for file in train_path.glob("*.csv"):
            attack_type = self.identify_attack_type(file.stem)
            if attack_type != 'unknown':
                files_by_attack[attack_type].append(file)
        
        # Log file distribution
        self.logger.info("\nFiles found per attack type:")
        for attack_type, files in files_by_attack.items():
            file_count = len(files)
            if self.max_files and file_count > self.max_files:
                files = np.random.choice(files, self.max_files, replace=False).tolist()
                file_count = len(files)
                files_by_attack[attack_type] = files
                
            self.logger.info(f"{attack_type}: {file_count} files")
            for f in files:
                self.logger.info(f"  - {f.name}")
        
        # Process each attack type separately to manage memory
        train_dfs = []
        holdout_dfs = []
        
        for attack_type, files in files_by_attack.items():
            self.logger.info(f"\nProcessing {attack_type} with {len(files)} files")
            
            if len(files) == 1:
                # For single file, split the data itself
                self.logger.info(f"Single file for {attack_type}: {files[0].name}")
                df = self.load_csv_in_chunks(files[0], attack_type)
                
                # Free memory before split
                gc.collect()
                
                train_idx, test_idx = train_test_split(
                    np.arange(len(df)), 
                    test_size=0.2,
                    random_state=42,
                    stratify=df['attack_type'] if len(df['attack_type'].unique()) > 1 else None
                )
                train_dfs.append(df.iloc[train_idx])
                holdout_dfs.append(df.iloc[test_idx])
                
                # Clear original dataframe to free memory
                del df
                gc.collect()
            else:
                # Calculate proper split sizes
                n_train, n_test = self.calculate_train_test_sizes(len(files))
                
                # Shuffle files first for better distribution
                shuffled_files = np.random.RandomState(42).permutation(files)
                train_files = shuffled_files[:n_train]
                test_files = shuffled_files[n_train:]
                
                self.logger.info(f"{attack_type}: Train files: {len(train_files)}, Test files: {len(test_files)}")
                
                # Process training files
                for file in tqdm(train_files, desc=f"Loading {attack_type} train files"):
                    df = self.load_csv_in_chunks(file, attack_type)
                    if not df.empty:
                        train_dfs.append(df)
                
                # Process test files
                for file in tqdm(test_files, desc=f"Loading {attack_type} test files"):
                    df = self.load_csv_in_chunks(file, attack_type)
                    if not df.empty:
                        holdout_dfs.append(df)
            
            # Force garbage collection after each attack type
            gc.collect()
        
        # Combine all data frames
        self.logger.info("Combining training data frames...")
        train_data = pd.concat(train_dfs, ignore_index=True)
        
        self.logger.info("Combining holdout data frames...")
        holdout_data = pd.concat(holdout_dfs, ignore_index=True)
        
        # Log class distribution
        self.logger.info("\nClass distribution in train set:")
        self.logger.info(str(train_data['attack_type'].value_counts()))
        
        self.logger.info("\nClass distribution in test set:")
        self.logger.info(str(holdout_data['attack_type'].value_counts()))
        
        return train_data, holdout_data
    
    def validate_dataset_completeness(self, data, expected_features):
        """
        Validate that dataset contains all expected features and classes.
        
        Args:
            data (DataFrame): Dataset to validate
            expected_features (set): Set of expected feature names
        
        Returns:
            bool: True if dataset is complete, False otherwise
        """
        # Check features
        missing_features = expected_features - set(data.columns)
        extra_features = set(data.columns) - expected_features
        
        if missing_features:
            self.logger.warning(f"Missing expected features: {missing_features}")
        if extra_features:
            self.logger.info(f"Additional features found: {extra_features}")
            
        # Check attack types
        present_attacks = set(data['attack_type'].unique())
        missing_attacks = self.expected_attacks - present_attacks
        extra_attacks = present_attacks - self.expected_attacks
        
        if missing_attacks:
            self.logger.warning(f"Missing attack types: {missing_attacks}")
        if extra_attacks:
            self.logger.warning(f"Unexpected attack types found: {extra_attacks}")
            
        # Log class distribution
        attack_dist = data['attack_type'].value_counts()
        self.logger.info("Attack type distribution:\n" + str(attack_dist))
        
        return (
            len(missing_features) == 0
            and len(missing_attacks) == 0
            and len(data['attack_type'].unique()) == len(self.expected_attacks)
        )


class FeatureProcessor:
    """
    Class for processing and selecting features for IoMT traffic classification.
    
    Attributes:
        scaler (StandardScaler): Scaler for feature normalization
        label_encoder (LabelEncoder): Encoder for transforming labels
        feature_cols (list): Selected feature columns
        selected_features (list): Final set of selected features
        logger (Logger): Logger instance for recording operations
    """
    
    def __init__(self):
        """Initialize the FeatureProcessor."""
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.selected_features = None
        
        # Setup logging
        self._setup_logging()
        
        # Define expected features based on paper
        self.expected_features = {
            'Header-Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
            'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
            'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
            'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
            'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
            'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance',
            'Weight', 'Drate', 'Header_Length'  # Added Drate and Header_Length from log
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("FeatureProcessor initialized")
    
    def prepare_features(self, train_data, test_data, perform_selection=True):
        """
        Prepare features with optional feature selection.
        
        Args:
            train_data (DataFrame): Training data
            test_data (DataFrame): Test data
            perform_selection (bool): Whether to perform feature selection
        
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Preparing features")
        
        # Select numeric features
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col != 'attack_type']
        
        # Scale features
        X_train = self.scaler.fit_transform(train_data[self.feature_cols])
        X_test = self.scaler.transform(test_data[self.feature_cols])
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_data['attack_type'])
        y_test = self.label_encoder.transform(test_data['attack_type'])
        
        # Print class distribution before balancing
        self.logger.info("Class distribution before balancing:")
        class_counts = np.bincount(y_train)
        for cls in range(len(class_counts)):
            class_name = self.label_encoder.inverse_transform([cls])[0]
            self.logger.info(f"{class_name}: {class_counts[cls]}")
        
        # Perform feature selection if requested
        if perform_selection:
            X_train, X_test = self._perform_feature_selection(X_train, X_test, y_train)
        else:
            # If not performing selection, all features are selected
            self.selected_features = self.feature_cols
            self.logger.info(f"\nUsing all {len(self.feature_cols)} features")
        
        # Balance the dataset
        X_train, y_train = self._balance_dataset(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def _perform_feature_selection(self, X_train, X_test, y_train, sample_size=200000):
        """
        Perform feature selection using Random Forest.
        
        Args:
            X_train (ndarray): Training features
            X_test (ndarray): Test features
            y_train (ndarray): Training labels
            sample_size (int): Maximum number of samples to use for selection
        
        Returns:
            tuple: (X_train_selected, X_test_selected)
        """
        self.logger.info("Analyzing feature importance...")
        
        # Sample data if needed
        if len(X_train) > sample_size:
            self.logger.info(f"Sampling {sample_size} instances for feature selection")
            indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            self.logger.info("Using full dataset for feature selection")
            X_sample = X_train
            y_sample = y_train
        
        # Use Random Forest for feature selection
        rf_analyzer = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            n_jobs=min(4, multiprocessing.cpu_count()),
            random_state=42,
            class_weight='balanced'
        )
        
        rf_analyzer.fit(X_sample, y_sample)
        
        # Get and print feature importances
        importances = rf_analyzer.feature_importances_
        feature_imp = list(zip(self.feature_cols, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info("\nFeature importance ranking:")
        for name, imp in feature_imp:
            self.logger.info(f"{name}: {imp:.4f}")
        
        # Keep all features as observed in the log
        self.selected_features = self.feature_cols
        
        # Log selection decision
        self.logger.info(f"\nUsing all {len(self.feature_cols)} features")
        
        return X_train, X_test
    
    def _balance_dataset(self, X_train, y_train):
        """
        Balance the dataset using appropriate sampling techniques.
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
        
        Returns:
            tuple: (X_balanced, y_balanced)
        """
        self.logger.info("Balancing dataset")
        
        # Calculate class distribution
        class_counts = np.bincount(y_train)
        total_samples = len(y_train)
        max_class_size = max(class_counts)
        min_class_size = min(class_counts[class_counts > 0])
        imbalance_ratio = max_class_size / min_class_size
        
        self.logger.info(f"Dataset statistics: Total samples: {total_samples:,}, Imbalance ratio: {imbalance_ratio:.2f}")
        
        # Based on the log, extreme imbalance exists (millions vs. thousands)
        # We need a more aggressive balancing strategy for rare classes like mqtt_malformed
        
        # Calculate target sizes for each class
        target_sizes = {}
        median_class_size = np.median(class_counts[class_counts > 0])
        
        for cls in range(len(class_counts)):
            if class_counts[cls] > 0:
                class_name = self.label_encoder.inverse_transform([cls])[0]
                
                # Special handling for very small classes (boost them more)
                if class_counts[cls] < 10000:  # For very rare classes like mqtt_malformed
                    target_size = int(median_class_size * 0.6)  # Boost them significantly
                    self.logger.info(f"Boosting rare class {class_name} from {class_counts[cls]} to {target_size}")
                elif class_counts[cls] < 100000:  # For uncommon classes like arp_spoofing
                    target_size = int(median_class_size * 0.5)
                    self.logger.info(f"Boosting uncommon class {class_name} from {class_counts[cls]} to {target_size}")
                elif class_counts[cls] > 1000000:  # For very common classes like ddos
                    target_size = int(median_class_size * 1.5)  # Reduce them somewhat
                    self.logger.info(f"Reducing very common class {class_name} from {class_counts[cls]} to {target_size}")
                else:  # For moderately sized classes
                    target_size = int(median_class_size)
                    self.logger.info(f"Adjusting class {class_name} from {class_counts[cls]} to {target_size}")
                
                target_sizes[cls] = target_size
        
        # Use SMOTETomek for both oversampling and undersampling
        try:
            # For extreme imbalance, ensure k_neighbors doesn't exceed possible values
            k_neighbors = min(5, min_class_size - 1)
            if k_neighbors < 1:
                k_neighbors = 1  # Failsafe
                
            self.logger.info(f"Using SMOTETomek with k_neighbors={k_neighbors}")
            
            balancer = SMOTETomek(
                sampling_strategy=target_sizes,
                random_state=42,
                n_jobs=min(4, multiprocessing.cpu_count()),
                smote=SMOTE(
                    k_neighbors=k_neighbors,
                    random_state=42
                )
            )
            
            # For very large datasets, process in chunks
            if total_samples > 2_000_000:
                self.logger.info("Large dataset detected, processing in chunks")
                chunk_size = 500_000
                X_balanced = []
                y_balanced = []
                
                for i in range(0, len(X_train), chunk_size):
                    end = min(i + chunk_size, len(X_train))
                    X_chunk = X_train[i:end]
                    y_chunk = y_train[i:end]
                    
                    # Ensure all classes are represented in the chunk
                    chunk_classes = set(np.unique(y_chunk))
                    all_classes = set(range(len(class_counts)))
                    missing_classes = all_classes - chunk_classes
                    
                    if missing_classes:
                        self.logger.warning(f"Chunk missing classes: {missing_classes}")
                        # Add a few samples of missing classes
                        for missing_cls in missing_classes:
                            missing_indices = np.where(y_train == missing_cls)[0][:100]  # Take up to 100 samples
                            if len(missing_indices) > 0:
                                X_chunk = np.vstack([X_chunk, X_train[missing_indices]])
                                y_chunk = np.concatenate([y_chunk, y_train[missing_indices]])
                    
                    X_bal, y_bal = balancer.fit_resample(X_chunk, y_chunk)
                    X_balanced.append(X_bal)
                    y_balanced.append(y_bal)
                
                X_train = np.vstack(X_balanced)
                y_train = np.concatenate(y_balanced)
            else:
                X_train, y_train = balancer.fit_resample(X_train, y_train)
        except Exception as e:
            self.logger.error(f"Error during balancing: {str(e)}")
            self.logger.info("Falling back to simpler SMOTE approach")
            
            # Use class weights for extreme imbalance
            try:
                # Calculate class weights inversely proportional to frequency
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=np.unique(y_train), 
                    y=y_train
                )
                class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
                self.logger.info(f"Using class weights: {class_weight_dict}")
                
                # Fall back to simpler approach with higher k for stability
                smote = SMOTE(
                    random_state=42,
                    k_neighbors=k_neighbors,
                    sampling_strategy='auto'  # Let SMOTE determine based on class distribution
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e2:
                self.logger.error(f"SMOTE also failed: {str(e2)}")
                self.logger.warning("Using original imbalanced dataset")
                # Keep original data if all balancing methods fail
        
        # Log final class distribution
        self.logger.info("Class distribution after balancing:")
        final_counts = np.bincount(y_train)
        for cls in range(len(final_counts)):
            class_name = self.label_encoder.inverse_transform([cls])[0]
            self.logger.info(f"{class_name}: {final_counts[cls]}")
        
        return X_train, y_train


class ModelTrainer:
    """
    Class for training and evaluating machine learning models for IoMT traffic classification.
    
    Attributes:
        classifiers (dict): Dictionary of classifier instances
        param_grids (dict): Hyperparameter grids for tuning
        results (dict): Dictionary to store evaluation results
        batch_size (int): Size of batches for processing large datasets
        logger (Logger): Logger instance for recording operations
    """
    
    def __init__(self, n_jobs=4, batch_size=100000):
        """
        Initialize the ModelTrainer.
        
        Args:
            n_jobs (int): Number of parallel jobs to run
            batch_size (int): Size of batches for processing large datasets
        """
        self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        self.batch_size = batch_size
        self.results = {}
        
        # Setup logging
        self._setup_logging()
        
        # Initialize classifiers with optimized parameters
        self.classifiers = self._initialize_classifiers()
        
        # Set up hyperparameter grids for model tuning
        self.param_grids = self._define_param_grids()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModelTrainer initialized")
    
    def _initialize_classifiers(self):
        """
        Initialize classifiers with default parameters.
        
        Returns:
            dict: Dictionary of initialized classifiers
        """
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                n_jobs=self.n_jobs,
                random_state=42,
                verbose=1  # Enable verbose output as seen in the log
            ),
            'logistic_regression': LogisticRegression(
                max_iter=1000,  # Increased from 500 to address convergence issues
                solver='saga',
                n_jobs=self.n_jobs,
                random_state=42,
                C=1.0,
                verbose=1
            ),
            'adaboost': AdaBoostClassifier(
                n_estimators=100,
                learning_rate=0.1,
                algorithm='SAMME.R',  # Changed to SAMME.R based on log performance
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),  # Similar to log example
                max_iter=300,      # Increased from original
                early_stopping=True,
                learning_rate='adaptive',  # Add adaptive learning rate
                learning_rate_init=0.001,
                random_state=42,
                verbose=True,      # Enable verbose output as seen in the log
                validation_fraction=0.1,
                n_iter_no_change=10
            )
        }
    
    def _define_param_grids(self):
        """
        Define hyperparameter grids for model tuning.
        
        Returns:
            dict: Dictionary of parameter grids for each classifier
        """
        # Based on log performance, adjust parameter grids for better exploration
        return {
            'random_forest': {
                'n_estimators': [50, 100],
                'max_depth': [10, 15, 20],
                'min_samples_split': [10, 20]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'solver': ['saga']  # Saga is better for large datasets
            },
            'adaboost': {
                'n_estimators': [50, 100],
                'learning_rate': [0.1, 0.5, 1.0]
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100, 50)],
                'alpha': [0.0001, 0.001],
                'learning_rate_init': [0.001, 0.01]
            }
        }
    
    def train_and_evaluate(self, X_train, y_train, X_test, y_test, label_encoder, perform_tuning=True):
        """
        Train and evaluate models with optional hyperparameter tuning.
        
        Args:
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
            label_encoder (LabelEncoder): Encoder used for transforming labels
            perform_tuning (bool): Whether to perform hyperparameter tuning
        
        Returns:
            dict: Dictionary of evaluation results
        """
        self.logger.info("Training and evaluating models with hyperparameter tuning...")
        self.results = {}
        
        # Store label encoder for reporting
        self.label_encoder = label_encoder
        
        # Convert y_test to 1D if needed
        if len(y_test.shape) > 1:
            y_test = y_test.ravel()
        
        for name, clf in self.classifiers.items():
            self.logger.info(f"\n{'='*20}\nTraining {name}...")
            try:
                start_time = time.time()
                
                if perform_tuning:
                    best_model = self._tune_hyperparameters(name, clf, X_train, y_train)
                else:
                    # Special handling for logistic regression with large datasets
                    if name == 'logistic_regression' and len(X_train) > 1000000:
                        self.logger.info("Large dataset detected for logistic regression, using batch processing...")
                        best_model = self._train_large_dataset(clf, X_train, y_train)
                    else:
                        # Direct training without tuning
                        model = clone(clf)
                        best_model = model.fit(X_train, y_train)
                
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
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                self.logger.exception("Exception details:")
                continue
        
        return self.results
    
    def _tune_hyperparameters(self, name, clf, X_train, y_train):
        """
        Tune hyperparameters for a classifier.
        
        Args:
            name (str): Name of the classifier
            clf (object): Classifier instance
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
        
        Returns:
            object: Trained model with best parameters
        """
        # Sample data for tuning if dataset is very large
        if len(X_train) > 500000:
            self.logger.info(f"Sampling data for hyperparameter tuning")
            indices = np.random.choice(X_train.shape[0], 500000, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]
        else:
            X_sample = X_train
            y_sample = y_train
        
        # Create pipeline
        pipeline = ImbPipeline([
            ('classifier', clf)
        ])
        
        # Setup grid search
        param_grid = {f'classifier__{k}': v for k, v in self.param_grids[name].items()}
        grid_search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            cv=3,
            n_jobs=self.n_jobs,
            scoring='balanced_accuracy',
            verbose=1  # Show progress
        )
        
        # Fit grid search
        grid_search.fit(X_sample, y_sample)
        
        # Log best parameters
        self.logger.info(f"Best parameters for {name}: {grid_search.best_params_}")
        
        # If using sample for tuning, retrain on full dataset with best params
        if len(X_train) > 500000:
            self.logger.info(f"Retraining {name} on full dataset with best parameters")
            
            # Clone the best estimator for retraining
            best_model = clone(grid_search.best_estimator_.named_steps['classifier'])
            best_model.fit(X_train, y_train)
            return best_model
        else:
            return grid_search.best_estimator_.named_steps['classifier']
    
    def _train_large_dataset(self, clf, X_train, y_train):
        """
        Train model on large dataset with batch processing.
        
        Args:
            clf (object): Classifier instance
            X_train (ndarray): Training features
            y_train (ndarray): Training labels
        
        Returns:
            object: Trained model
        """
        self.logger.info("Large dataset detected, using batch training approach")
        
        # Sample a portion for initial training
        sample_size = min(500000, len(X_train) // 2)
        indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]
        
        # Clone the classifier to avoid modifying the original
        model = clone(clf)
        
        # Fit on the sample for warm start
        self.logger.info(f"Initial training on {sample_size} samples")
        model.fit(X_sample, y_sample)
        
        # For models that support partial_fit, we could use that
        # But for now, just return the model trained on the sample
        self.logger.info("Batch training completed")
        return model
    
    def _predict_in_batches(self, model, X_test):
        """
        Make predictions in batches for large datasets.
        
        Args:
            model (object): Trained model
            X_test (ndarray): Test features
        
        Returns:
            tuple: (y_pred, y_pred_proba) - Predictions and probabilities
        """
        y_pred = []
        y_pred_proba = []
        
        # Process in batches
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
        if y_pred_proba:
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


class Evaluator:
    """
    Class for evaluating and visualizing IoMT classification results.
    
    Attributes:
        results (dict): Dictionary of evaluation results
        label_encoder (LabelEncoder): Encoder used for transforming labels
        feature_cols (list): Feature columns
        selected_features (list): Selected feature columns
        logger (Logger): Logger instance for recording operations
    """
    
    def __init__(self, results, label_encoder, feature_cols, selected_features):
        """
        Initialize the Evaluator.
        
        Args:
            results (dict): Dictionary of evaluation results
            label_encoder (LabelEncoder): Encoder used for transforming labels
            feature_cols (list): Feature columns
            selected_features (list): Selected feature columns
        """
        self.results = results
        self.label_encoder = label_encoder
        self.feature_cols = feature_cols
        self.selected_features = selected_features
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("Evaluator initialized")
    
    def save_comprehensive_results(self, save_dir, X_test, y_test):
        """
        Save detailed evaluation results.
        
        Args:
            save_dir (Path): Directory to save results
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
        
        Returns:
            Path: Path to results directory
        """
        self.logger.info("Saving comprehensive results")
        
        # Create timestamped results directory
        save_dir = Path(save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_dir = save_dir / f"evaluation_results_{timestamp}"
        
        # Create subdirectories
        plots_dir = results_dir / "plots"
        reports_dir = results_dir / "reports"
        models_dir = results_dir / "models"
        
        for dir_path in [plots_dir, reports_dir, models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save results
        self._save_models(models_dir)
        self._save_visualizations(plots_dir, X_test, y_test)
        self._save_reports(reports_dir, y_test)
        self._save_feature_analysis(reports_dir)
        
        self.logger.info(f"All results saved in: {results_dir}")
        return results_dir
    
    def _save_models(self, models_dir):
        """
        Save trained models and associated components.
        
        Args:
            models_dir (Path): Directory to save models
        """
        self.logger.info("Saving models")
        
        # Save individual models
        for name, result in self.results.items():
            model_path = models_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'model': result['model'],
                    'training_time': result['training_time'],
                    'report': result['report']
                }, f)
            
            # Save model metadata in JSON format for readability
            metadata_path = models_dir / f"{name}_metadata.json"
            with open(metadata_path, 'w') as f:
                metadata = {
                    'training_time_seconds': result['training_time'],
                    'accuracy': result['accuracy'],
                    'macro_f1': result['macro_f1'],
                    'model_parameters': str(result['model'].get_params())
                }
                json.dump(metadata, f, indent=4)
    
    def _save_visualizations(self, plots_dir, X_test, y_test):
        """
        Generate and save visualization plots.
        
        Args:
            plots_dir (Path): Directory to save plots
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
        """
        self.logger.info("Generating plots...")
        
        try:
            # Performance comparison plot
            self._create_performance_plot(plots_dir)
            
            # Training time comparison plot
            self._create_training_time_plot(plots_dir)
            
            # Confusion matrices
            self._create_confusion_matrices(plots_dir, y_test)
            
            # ROC curves (only for models that support predict_proba)
            self._create_roc_curves(plots_dir, X_test, y_test)
            
        except Exception as e:
            self.logger.error(f"Error creating plots: {str(e)}")
            self.logger.exception("Exception details:")
    
    def _create_performance_plot(self, plots_dir):
        """
        Create and save performance comparison plot.
        
        Args:
            plots_dir (Path): Directory to save plot
        """
        # Prepare data
        metrics_data = []
        for name, result in self.results.items():
            metrics_data.append({
                'Model': name,
                'Accuracy': result['accuracy'],
                'Macro_F1': result['macro_f1']
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        
        # Create plot
        plt.figure(figsize=(12, 6))
        ax = metrics_df.set_index('Model')[['Accuracy', 'Macro_F1']].plot(kind='bar')
        plt.title('Model Performance Comparison', fontsize=16)
        plt.ylabel('Score', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add percentage labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_training_time_plot(self, plots_dir):
        """
        Create and save training time comparison plot.
        
        Args:
            plots_dir (Path): Directory to save plot
        """
        # Prepare data
        time_data = []
        for name, result in self.results.items():
            time_data.append({
                'Model': name,
                'Training_Time': result['training_time'] / 60  # Convert to minutes
            })
        
        time_df = pd.DataFrame(time_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        ax = time_df.plot(kind='bar', x='Model', y='Training_Time')
        plt.title('Training Time Comparison', fontsize=16)
        plt.ylabel('Time (minutes)', fontsize=14)
        plt.xlabel('Model', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add minute labels
        ax.bar_label(ax.containers[0], fmt='%.1f min', padding=3)
        
        plt.tight_layout()
        plt.savefig(plots_dir / "training_times.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_confusion_matrices(self, plots_dir, y_test):
        """
        Create and save confusion matrices for each model.
        
        Args:
            plots_dir (Path): Directory to save plots
            y_test (ndarray): Test labels
        """
        for name, result in self.results.items():
            try:
                plt.figure(figsize=(15, 12))
                predictions = result['predictions']
                
                # Get actual classes present in predictions and test data
                present_classes = np.unique(np.concatenate([predictions, y_test]))
                cm = confusion_matrix(
                    y_test, 
                    predictions,
                    labels=present_classes
                )
                
                # Get class names for present classes
                class_names = [self.label_encoder.classes_[i] for i in present_classes]
                
                # Format numbers for better readability
                labels = [[f'{x:,}' for x in row] for row in cm]
                
                # Create heatmap
                sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                        xticklabels=class_names,
                        yticklabels=class_names,
                        square=True,
                        annot_kws={'size': 12},
                        cbar_kws={'label': 'Number of Samples'})
                
                plt.title(f'Confusion Matrix - {name}', fontsize=16, pad=20)
                plt.xlabel('Predicted', fontsize=14)
                plt.ylabel('Actual', fontsize=14)
                plt.xticks(rotation=45, ha='right', fontsize=12)
                plt.yticks(rotation=0, fontsize=12)
                plt.grid(True, which='minor', color='white', linewidth=0.5)
                plt.tight_layout(pad=1.2)
                plt.savefig(plots_dir / f"confusion_matrix_{name}.png", dpi=300, bbox_inches='tight')
                plt.close()
            except Exception as e:
                self.logger.error(f"Error creating confusion matrix for {name}: {str(e)}")
                plt.close()
    
    def _create_roc_curves(self, plots_dir, X_test, y_test, batch_size=100000):
        """
        Create and save ROC curves for models that support probability prediction.
        
        Args:
            plots_dir (Path): Directory to save plot
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
            batch_size (int): Size of batches for processing
        """
        plt.figure(figsize=(12, 10))
        
        # For each model that can predict probabilities
        for name, result in self.results.items():
            model = result['model']
            if hasattr(model, 'predict_proba'):
                try:
                    # Get probabilities in batches
                    y_proba_batches = []
                    
                    total_batches = (len(X_test) + batch_size - 1) // batch_size
                    for i in range(0, len(X_test), batch_size):
                        batch_num = i // batch_size + 1
                        self.logger.info(f"Processing batch {batch_num}/{total_batches}")
                        
                        end = min(i + batch_size, len(X_test))
                        X_batch = X_test[i:end]
                        batch_proba = model.predict_proba(X_batch)
                        y_proba_batches.append(batch_proba)
                    
                    y_proba = np.vstack(y_proba_batches)
                    
                    # Plot ROC curve for each class (limit to 3 for clarity)
                    # Based on the log, we should include 'ddos', 'dos', and 'benign' classes
                    target_classes = ['ddos', 'dos', 'benign']
                    
                    for class_name in target_classes:
                        # Get class index
                        if class_name in self.label_encoder.classes_:
                            class_idx = np.where(self.label_encoder.classes_ == class_name)[0][0]
                            
                            # Create ROC curve
                            y_true_binary = (y_test == class_idx).astype(int)
                            y_score = y_proba[:, class_idx]
                            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
                            roc_auc = auc(fpr, tpr)
                            
                            plt.plot(
                                fpr, 
                                tpr, 
                                label=f'{name} - {class_name} (AUC = {roc_auc:.2f})',
                                linewidth=2
                            )
                except Exception as e:
                    self.logger.error(f"Error generating ROC curve for {name}: {str(e)}")
                    continue
        
        # Add random classifier line
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        
        # Customize the plot
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves for Models and Selected Classes', fontsize=14)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save with tight layout
        plt.tight_layout()
        plt.savefig(plots_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_reports(self, reports_dir, y_test):
        """
        Save detailed reports of model performance.
        
        Args:
            reports_dir (Path): Directory to save reports
            y_test (ndarray): Test labels
        """
        self.logger.info("Saving model performance reports")
        
        # Ensure y_test is 1D
        if len(y_test.shape) > 1:
            y_test = y_test.ravel()
        
        # Save classification reports
        with open(reports_dir / "classification_reports.txt", 'w') as f:
            f.write("CLASSIFICATION REPORTS\n")
            f.write("=====================\n\n")
            
            for name, result in self.results.items():
                f.write(f"\n{name.upper()}\n{'='*len(name)}\n")
                
                # Generate report
                report_str = classification_report(
                    y_test,
                    result['predictions'],
                    target_names=self.label_encoder.classes_,
                    labels=np.unique(result['predictions']),
                    zero_division=0
                )
                
                f.write(report_str)
                f.write(f"\nTraining Time: {result['training_time']:.2f} seconds\n")
                f.write("-"*50 + "\n")
        
        # Save model parameters
        params = {
            name: str(result['model'].get_params()) 
            for name, result in self.results.items()
        }
        
        with open(reports_dir / "model_parameters.json", 'w') as f:
            json.dump(params, f, indent=4)
        
        # Save dataset statistics
        stats = {
            'feature_names': self.feature_cols,
            'selected_features': self.selected_features,
            'n_features': len(self.feature_cols),
            'n_selected_features': len(self.selected_features),
            'n_classes': len(self.label_encoder.classes_),
            'class_names': list(self.label_encoder.classes_),
            'class_distribution': {
                self.label_encoder.classes_[i]: int(count)
                for i, count in enumerate(np.bincount(y_test))
                if count > 0
            }
        }
        
        with open(reports_dir / "dataset_statistics.json", 'w') as f:
            json.dump(stats, f, indent=4)
    
    def _save_feature_analysis(self, reports_dir):
        """
        Save feature importance analysis.
        
        Args:
            reports_dir (Path): Directory to save analysis
        """
        self.logger.info("Saving feature importance analysis")
        
        feature_importance = {}
        
        # Analyze feature importance for models that support it
        if 'random_forest' in self.results:
            rf_model = self.results['random_forest']['model']
            
            if hasattr(rf_model, 'feature_importances_'):
                feature_importance['random_forest'] = dict(zip(
                    self.selected_features,
                    rf_model.feature_importances_
                ))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance['random_forest'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Create more readable format with percentages
                feature_importance['random_forest_summary'] = {
                    feature: f"{importance*100:.2f}%" 
                    for feature, importance in sorted_features
                }
        
        # AdaBoost feature importance
        if 'adaboost' in self.results:
            ada_model = self.results['adaboost']['model']
            
            if hasattr(ada_model, 'feature_importances_'):
                feature_importance['adaboost'] = dict(zip(
                    self.selected_features,
                    ada_model.feature_importances_
                ))
                
                # Sort features by importance
                sorted_features = sorted(
                    feature_importance['adaboost'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Create more readable format with percentages
                feature_importance['adaboost_summary'] = {
                    feature: f"{importance*100:.2f}%" 
                    for feature, importance in sorted_features
                }
        
        # Save analysis to JSON file
        with open(reports_dir / "feature_importance.json", 'w') as f:
            json.dump(feature_importance, f, indent=4)
        
        # Create a more readable summary file
        with open(reports_dir / "feature_importance_summary.txt", 'w') as f:
            f.write("Feature Importance Analysis\n")
            f.write("=========================\n\n")
            
            if 'random_forest' in feature_importance:
                f.write("Random Forest Feature Importance:\n")
                f.write("---------------------------------\n")
                for feature, importance in feature_importance['random_forest_summary'].items():
                    f.write(f"{feature}: {importance}\n")
                f.write("\n")
            
            if 'adaboost' in feature_importance:
                f.write("AdaBoost Feature Importance:\n")
                f.write("---------------------------\n")
                for feature, importance in feature_importance['adaboost_summary'].items():
                    f.write(f"{feature}: {importance}\n")
                f.write("\n")
            
            if 'neural_network' in self.results:
                f.write("Neural Network Feature Importance:\n")
                f.write("--------------------------------\n")
                f.write("Note: Neural networks don't provide direct feature importance measures.\n")
                f.write("Consider using techniques like permutation importance for detailed analysis.\n")


class IoMTClassificationSystem:
    """
    A comprehensive system for IoMT network traffic classification.
    
    This class orchestrates the entire workflow from data loading to
    model training, evaluation, and result visualization.
    
    Attributes:
        data_loader (DataLoader): Component for loading and preprocessing data
        feature_processor (FeatureProcessor): Component for feature processing
        model_trainer (ModelTrainer): Component for model training and evaluation
        batch_size (int): Size of data chunks for processing
        logger (Logger): Logger instance
    """
    
    def __init__(self, batch_size=1000000, max_files=None, n_jobs=4):
        """
        Initialize the IoMT classification system.
        
        Args:
            batch_size (int): Size of chunks for processing large files
            max_files (int, optional): Maximum number of files to process per attack type
            n_jobs (int): Number of parallel jobs to run
        """
        self.batch_size = batch_size
        
        # Initialize components
        self.data_loader = DataLoader(batch_size=batch_size, max_files=max_files)
        self.feature_processor = FeatureProcessor()
        self.model_trainer = ModelTrainer(n_jobs=n_jobs, batch_size=batch_size)
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("IoMT Classification System initialized")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_filename = f'iomt_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self, data_path, results_dir, feature_selection=True, hyperparameter_tuning=True):
        """
        Run the complete classification workflow.
        
        Args:
            data_path (str or Path): Path to the data directory
            results_dir (str or Path): Directory to save results
            feature_selection (bool): Whether to perform feature selection
            hyperparameter_tuning (bool): Whether to tune hyperparameters
        
        Returns:
            Path: Path to results directory
        """
        self.logger.info(f"Starting IoMT Multi-Classifier Evaluation")
        self.logger.info("=" * 50)
        print("Starting IoMT Multi-Classifier Evaluation")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data")
            data_path = Path(data_path)
            train_data, holdout_data = self.data_loader.load_data_with_holdout(data_path)
            
            # Validate dataset completeness
            self.data_loader.validate_dataset_completeness(
                train_data,
                self.feature_processor.expected_features
            )
            
            # Step 2: Process features
            self.logger.info("Step 2: Processing features")
            X_train, X_test, y_train, y_test = self.feature_processor.prepare_features(
                train_data, 
                holdout_data,
                perform_selection=feature_selection
            )
            
            # Free memory
            del train_data, holdout_data
            gc.collect()
            
            # Step 3: Train and evaluate models
            self.logger.info("Step 3: Training and evaluating models")
            results = self.model_trainer.train_and_evaluate(
                X_train, 
                y_train, 
                X_test, 
                y_test,
                self.feature_processor.label_encoder,
                perform_tuning=hyperparameter_tuning
            )
            
            # Step 4: Save results
            self.logger.info("Step 4: Saving results")
            evaluator = Evaluator(
                results,
                self.feature_processor.label_encoder,
                self.feature_processor.feature_cols,
                self.feature_processor.selected_features
            )
            
            results_path = evaluator.save_comprehensive_results(
                results_dir,
                X_test,
                y_test
            )
            
            # Log completion
            elapsed_time = time.time() - start_time
            self.logger.info(f"Classification workflow completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Results saved in: {results_path}")
            
            print(f"\nEvaluation completed successfully!")
            print(f"Results saved in: {results_path}")
            
            return results_path
        
        except Exception as e:
            self.logger.error(f"Error in classification workflow: {str(e)}")
            self.logger.exception("Exception details:")
            print(f"Error occurred: {str(e)}")
            raise


def main():
    """Main function to run the IoMT classification system."""
    
    # Define paths
    base_path = Path("Dataset/WiFi_and_MQTT/attacks/CSV")  # Update with your actual path
    results_dir = Path("evaluation_results")               # Update with your actual path
    
    try:
        # Create results directory if it doesn't exist
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize classification system
        system = IoMTClassificationSystem(
            batch_size=500000,    # Smaller batch size for better memory management
            max_files=None,       # Process all files
            n_jobs=4              # Use 4 parallel jobs
        )
        
        # Run classification workflow
        results_path = system.run(
            data_path=base_path,
            results_dir=results_dir,
            feature_selection=True,       # Perform feature selection
            hyperparameter_tuning=True    # Perform hyperparameter tuning
        )
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()