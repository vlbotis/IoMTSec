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
import psutil  # For memory monitoring

# Scikit-learn imports
from sklearn.model_selection import train_test_split

class DataLoader:
    """
    Class for efficiently loading and preprocessing IoMT network traffic data.
    
    Attributes:
        batch_size (int): Size of data chunks for processing large files
        max_files (int, optional): Maximum number of files to process per class
        logger (Logger): Logger instance for recording operations
        expected_attacks (set): Set of expected attack types
        attack_patterns (dict): Dictionary of attack patterns for identification
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
        
    def _monitor_resources(self):
        """Monitor system resources."""
        memory_used = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        self.memory_usage.append(memory_used)
        self.logger.info(f"Memory usage: {memory_used:.2f} MB")
        
        if memory_used > 0.8 * psutil.virtual_memory().total / 1024 / 1024:
            self.logger.warning("High memory usage detected!")
            gc.collect()
    
    def _save_checkpoint(self, model_name, model, performance):
        """Save model checkpoint."""
        if self.last_checkpoint is None or time.time() - self.last_checkpoint >= self.checkpoint_interval:
            checkpoint = {
                'model_name': model_name,
                'model_state': model,
                'performance': performance,
                'timestamp': time.time()
            }
            with open(f'checkpoint_{model_name}.pkl', 'wb') as f:
                pickle.dump(checkpoint, f)
            self.last_checkpoint = time.time()
            self.logger.info(f"Checkpoint saved for {model_name}")
    
    def _estimate_remaining_time(self, completed_models):
        """Estimate remaining training time."""
        if not completed_models:
            return "Unknown"
        
        avg_time_per_model = (time.time() - self.start_time) / len(completed_models)
        remaining_models = len(self.model_order) - len(completed_models)
        remaining_seconds = avg_time_per_model * remaining_models
        
        return f"{remaining_seconds/3600:.1f} hours"
    
    def _log_model_performance(self, model_name, report, training_time):
        """Log detailed model performance metrics."""
        self.logger.info(f"\nPerformance Report for {model_name}")
        self.logger.info(f"Training time: {training_time:.2f} seconds")
        self.logger.info(f"Accuracy: {report['accuracy']:.4f}")
        self.logger.info(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        
        if model_name in self.results:
            model = self.results[model_name]['model']
            if hasattr(model, 'n_iter_'):
                self.logger.info(f"Iterations to converge: {model.n_iter_}")
        
        self.logger.info("\nPer-class Performance:")
        for class_name in self.label_encoder.classes_:
            if class_name in report:
                metrics = report[class_name]
                self.logger.info(f"{class_name}:")
                self.logger.info(f"  Precision: {metrics['precision']:.4f}")
                self.logger.info(f"  Recall: {metrics['recall']:.4f}")
                self.logger.info(f"  F1-score: {metrics['f1-score']:.4f}")
        
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