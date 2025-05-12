import numpy as np
import pandas as pd
import logging
from datetime import datetime
import gc
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import multiprocessing
import joblib
from pathlib import Path

class OptimizedFeatureProcessor:
    """
    Optimized class for processing and selecting features for IoMT traffic classification.
    
    This version focuses on memory efficiency and faster processing.
    """
    
    def __init__(self, n_jobs=None):
        """
        Initialize the OptimizedFeatureProcessor.
        
        Args:
            n_jobs (int): Number of parallel jobs (defaults to CPU count - 1)
        """
        # Auto-detect optimal number of jobs
        if n_jobs is None:
            self.n_jobs = min(multiprocessing.cpu_count() - 1, 8)
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
            
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_cols = None
        self.selected_features = None
        
        # Setup logging
        self._setup_logging()
        
        # Expected features based on domain knowledge
        self.expected_features = {
            'Header-Length', 'Protocol Type', 'Duration', 'Rate', 'Srate',
            'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
            'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
            'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count',
            'rst_count', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH',
            'IRC', 'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv',
            'LLC', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size',
            'IAT', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance',
            'Weight', 'Drate', 'Header_Length'
        }
    
    def _setup_logging(self):
        """Set up logging configuration."""
        self.logger = logging.getLogger(__name__)
        self.logger.info("OptimizedFeatureProcessor initialized")
    
    def prepare_features(self, train_data, test_data, perform_selection=True, 
                        importance_threshold=0.01, save_scaler=True):
        """
        Prepare features with optimized feature selection and scaling.
        
        Args:
            train_data (DataFrame): Training data
            test_data (DataFrame): Test data
            perform_selection (bool): Whether to perform feature selection
            importance_threshold (float): Minimum importance for keeping a feature
            save_scaler (bool): Whether to save the fitted scaler
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Preparing features")
        
        # Select numeric features
        numeric_cols = train_data.select_dtypes(include=[np.number]).columns
        self.feature_cols = [col for col in numeric_cols if col != 'attack_type']
        
        # Convert data to float32 to reduce memory usage
        X_train = train_data[self.feature_cols].astype(np.float32).values
        X_test = test_data[self.feature_cols].astype(np.float32).values
        
        # Scale features
        self.logger.info("Scaling features")
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        
        # Encode labels
        self.logger.info("Encoding labels")
        self.label_encoder = LabelEncoder()
        y_train = self.label_encoder.fit_transform(train_data['attack_type'])
        y_test = self.label_encoder.transform(test_data['attack_type'])
        
        # Print class distribution
        self.logger.info("Class distribution before processing:")
        class_counts = np.bincount(y_train)
        for cls in range(len(class_counts)):
            class_name = self.label_encoder.inverse_transform([cls])[0]
            self.logger.info(f"{class_name}: {class_counts[cls]}")
        
        # Perform feature selection if requested
        if perform_selection:
            X_train, X_test = self._perform_advanced_feature_selection(
                X_train, X_test, y_train, importance_threshold)
        else:
            # If not performing selection, all features are selected
            self.selected_features = self.feature_cols
            self.logger.info(f"\nUsing all {len(self.feature_cols)} features")
        
        # Save scaler if requested
        if save_scaler:
            self._save_scaler()
        
        return X_train, X_test, y_train, y_test
    
    def _perform_advanced_feature_selection(self, X_train, X_test, y_train, 
                                          importance_threshold=0.01, sample_size=200000):
        """
        Perform advanced feature selection using Random Forest importance
        and mutual information for better selection quality.
        
        Args:
            X_train (ndarray): Training features
            X_test (ndarray): Test features
            y_train (ndarray): Training labels
            importance_threshold (float): Minimum importance for keeping a feature
            sample_size (int): Maximum number of samples to use for selection
            
        Returns:
            tuple: (X_train_selected, X_test_selected)
        """
        self.logger.info("Performing advanced feature selection")
        
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
            n_jobs=self.n_jobs,
            random_state=42,
            class_weight='balanced'
        )
        
        self.logger.info("Training Random Forest for feature importance")
        rf_analyzer.fit(X_sample, y_sample)
        
        # Get and print feature importances
        importances = rf_analyzer.feature_importances_
        feature_imp = list(zip(self.feature_cols, importances))
        feature_imp.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.info("\nFeature importance ranking:")
        for name, imp in feature_imp:
            self.logger.info(f"{name}: {imp:.4f}")
        
        # Apply mutual information to get another perspective on feature relevance
        self.logger.info("Calculating mutual information")
        mutual_info = mutual_info_classif(
            X_sample, y_sample, 
            discrete_features=False,
            n_neighbors=3,
            random_state=42
        )
        
        # Combine both scores with a weighted average
        combined_scores = 0.7 * importances + 0.3 * (mutual_info / np.max(mutual_info))
        
        # Select features based on combined importance
        selected_indices = np.where(combined_scores > importance_threshold)[0]
        self.selected_features = [self.feature_cols[i] for i in selected_indices]
        
        # Log selection decision
        self.logger.info(f"\nSelected {len(self.selected_features)} out of {len(self.feature_cols)} features")
        self.logger.info("Selected features: " + ", ".join(self.selected_features))
        
        # Extract selected features
        X_train_selected = X_train[:, selected_indices]
        X_test_selected = X_test[:, selected_indices]
        
        # Save mapping of selected features for future use
        self._save_feature_mapping(selected_indices)
        
        return X_train_selected, X_test_selected
    
    def _save_scaler(self):
        """Save the fitted scaler for future use."""
        try:
            # Create models directory if it doesn't exist
            Path("models").mkdir(exist_ok=True)
            
            # Save scaler
            joblib.dump(self.scaler, "models/scaler.pkl")
            self.logger.info("Scaler saved to models/scaler.pkl")
        except Exception as e:
            self.logger.error(f"Error saving scaler: {str(e)}")
    
    def _save_feature_mapping(self, selected_indices):
        """
        Save the mapping of selected features for future use.
        
        Args:
            selected_indices (ndarray): Indices of selected features
        """
        try:
            # Create models directory if it doesn't exist
            Path("models").mkdir(exist_ok=True)
            
            # Save feature mapping
            feature_mapping = {
                'all_features': self.feature_cols,
                'selected_features': self.selected_features,
                'selected_indices': selected_indices.tolist()
            }
            
            joblib.dump(feature_mapping, "models/feature_mapping.pkl")
            self.logger.info("Feature mapping saved to models/feature_mapping.pkl")
        except Exception as e:
            self.logger.error(f"Error saving feature mapping: {str(e)}")
    
    @staticmethod
    def load_processor(scaler_path="models/scaler.pkl", feature_mapping_path="models/feature_mapping.pkl"):
        """
        Load a previously saved feature processor.
        
        Args:
            scaler_path (str): Path to the saved scaler
            feature_mapping_path (str): Path to the saved feature mapping
            
        Returns:
            OptimizedFeatureProcessor: Loaded feature processor
        """
        try:
            processor = OptimizedFeatureProcessor()
            
            # Load scaler
            processor.scaler = joblib.load(scaler_path)
            
            # Load feature mapping
            feature_mapping = joblib.load(feature_mapping_path)
            processor.feature_cols = feature_mapping['all_features']
            processor.selected_features = feature_mapping['selected_features']
            
            return processor
        except Exception as e:
            print(f"Error loading feature processor: {str(e)}")
            return None