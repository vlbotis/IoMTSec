import numpy as np
import pandas as pd
import time
import logging
from datetime import datetime
import gc
import os
from pathlib import Path
import joblib
import json
import multiprocessing

# Import optimized components
from optimized_feature_processor import OptimizedFeatureProcessor
from optimized_model_trainer import OptimizedModelTrainer

# Import DataLoader (reusing your existing implementation with minor modifications)
# Assume this is in dataloader.py
from dataloader import DataLoader

class OptimizedIoMTClassificationSystem:
    """
    An optimized system for IoMT network traffic classification.
    
    This implementation focuses on speed, memory efficiency, and scalability.
    """
    
    def __init__(self, batch_size=100000, max_files=None, n_jobs=None, 
                max_runtime_hours=4, use_gpu=False, max_per_class=50000):
        """
        Initialize the optimized IoMT classification system.
        
        Args:
            batch_size (int): Size of chunks for processing large files
            max_files (int, optional): Maximum number of files to process per attack type
            n_jobs (int): Number of parallel jobs (defaults to CPU count - 1)
            max_runtime_hours (float): Maximum runtime for training in hours
            use_gpu (bool): Whether to use GPU acceleration if available
            max_per_class (int): Maximum samples per class after balancing
        """
        self.batch_size = batch_size
        self.max_per_class = max_per_class
        
        # Auto-detect optimal number of jobs
        if n_jobs is None:
            self.n_jobs = min(multiprocessing.cpu_count() - 1, 8)
        else:
            self.n_jobs = min(n_jobs, multiprocessing.cpu_count())
        
        # Initialize components
        self.data_loader = DataLoader(batch_size=batch_size, max_files=max_files)
        self.feature_processor = OptimizedFeatureProcessor(n_jobs=self.n_jobs)
        self.model_trainer = OptimizedModelTrainer(
            n_jobs=self.n_jobs, 
            batch_size=batch_size,
            max_runtime_hours=max_runtime_hours,
            use_gpu=use_gpu
        )
        
        # Setup logging
        self._setup_logging()
        
        self.logger.info("OptimizedIoMTClassificationSystem initialized")
        if max_runtime_hours:
            self.logger.info(f"Maximum runtime set to {max_runtime_hours} hours")
    
    def _setup_logging(self):
        """Set up logging configuration."""
        log_filename = f'optimized_system_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            filename=log_filename,
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def run(self, data_path, results_dir, feature_selection=True, hyperparameter_tuning=True):
        """
        Run the complete classification workflow with optimizations.
        
        Args:
            data_path (str or Path): Path to the data directory
            results_dir (str or Path): Directory to save results
            feature_selection (bool): Whether to perform feature selection
            hyperparameter_tuning (bool): Whether to tune hyperparameters
            
        Returns:
            Path: Path to results directory
        """
        self.logger.info(f"Starting Optimized IoMT Multi-Classifier Evaluation")
        self.logger.info("=" * 50)
        print("Starting Optimized IoMT Multi-Classifier Evaluation")
        print("=" * 50)
        
        start_time = time.time()
        
        try:
            # Step 1: Load data
            self.logger.info("Step 1: Loading data")
            data_path = Path(data_path)
            train_data, holdout_data = self.data_loader.load_data_with_holdout(data_path)
            
            # Log memory usage
            self._log_memory_usage("After data loading")
            
            # Step 2: Process features
            self.logger.info("Step 2: Processing features")
            X_train, X_test, y_train, y_test = self.feature_processor.prepare_features(
                train_data, 
                holdout_data,
                perform_selection=feature_selection,
                importance_threshold=0.01,  # Only keep features with >1% importance
                save_scaler=True
            )
            
            # Free memory
            del train_data, holdout_data
            gc.collect()
            self._log_memory_usage("After feature processing")
            
            # Step 3: Train and evaluate models
            self.logger.info("Step 3: Training and evaluating models")
            results = self.model_trainer.train_and_evaluate(
                X_train, 
                y_train, 
                X_test, 
                y_test,
                self.feature_processor.label_encoder,
                perform_tuning=hyperparameter_tuning,
                max_per_class=self.max_per_class
            )
            
            # Free more memory
            del X_train, y_train
            gc.collect()
            self._log_memory_usage("After model training")
            
            # Step 4: Save results
            self.logger.info("Step 4: Saving results")
            results_path = self._save_results(
                results_dir,
                results,
                X_test,
                y_test
            )
            
            # Export best model to ONNX format for faster inference
            if results:
                # Find best model based on macro F1
                best_model_name = max(
                    results.keys(), 
                    key=lambda k: results[k]['macro_f1'] if k in results else 0
                )
                
                self.logger.info(f"Best model: {best_model_name}")
                
                # Export to ONNX if in results
                if best_model_name in results:
                    feature_dim = X_test.shape[1]
                    onnx_path = self.model_trainer.export_to_onnx(best_model_name, feature_dim)
                    if onnx_path:
                        self.logger.info(f"Best model exported to ONNX: {onnx_path}")
            
            # Log completion
            elapsed_time = time.time() - start_time
            self.logger.info(f"Classification workflow completed in {elapsed_time:.2f} seconds")
            self.logger.info(f"Results saved in: {results_path}")
            
            print(f"\nEvaluation completed successfully!")
            print(f"Total time: {elapsed_time/60:.2f} minutes")
            print(f"Results saved in: {results_path}")
            
            return results_path
        
        except Exception as e:
            self.logger.error(f"Error in classification workflow: {str(e)}")
            self.logger.exception("Exception details:")
            print(f"Error occurred: {str(e)}")
            raise
    
    def _log_memory_usage(self, step_name):
        """
        Log current memory usage.
        
        Args:
            step_name (str): Name of the current processing step
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            self.logger.info(f"Memory usage at {step_name}: {memory_mb:.2f} MB")
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
    
    def _save_results(self, results_dir, results, X_test, y_test):
        """
        Save model results and performance metrics.
        
        Args:
            results_dir (str or Path): Directory to save results
            results (dict): Model results
            X_test (ndarray): Test features
            y_test (ndarray): Test labels
            
        Returns:
            Path: Path to results directory
        """
        # Create timestamped results directory
        results_dir = Path(results_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_path = results_dir / f"optimized_results_{timestamp}"
        
        # Create subdirectories
        models_dir = results_path / "models"
        metrics_dir = results_path / "metrics"
        plots_dir = results_path / "plots"
        
        for directory in [models_dir, metrics_dir, plots_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Save each model
        for model_name, model_results in results.items():
            # Save model
            model_file = models_dir / f"{model_name}_model.pkl"
            model_data = {
                "model": model_results["model"],
                "label_encoder": self.feature_processor.label_encoder,
                "feature_columns": self.feature_processor.selected_features,
                "training_time": model_results["training_time"],
                "timestamp": timestamp
            }
            joblib.dump(model_data, model_file)
            
            # Save metrics
            metrics_file = metrics_dir / f"{model_name}_metrics.json"
            with open(metrics_file, 'w') as f:
                metrics = {
                    "accuracy": float(model_results["accuracy"]),
                    "macro_f1": float(model_results["macro_f1"]),
                    "training_time_seconds": float(model_results["training_time"]),
                    "report": self._make_json_serializable(model_results["report"])
                }
                json.dump(metrics, f, indent=4)
            
            # Generate and save plots
            self._generate_plots(
                model_name, 
                model_results, 
                y_test, 
                self.feature_processor.label_encoder,
                plots_dir
            )
        
        # Save system configuration
        config = {
            "timestamp": timestamp,
            "batch_size": self.batch_size,
            "max_per_class": self.max_per_class,
            "n_jobs": self.n_jobs,
            "feature_selection": len(self.feature_processor.selected_features) < len(self.feature_processor.feature_cols),
            "selected_features": self.feature_processor.selected_features,
            "feature_count": len(self.feature_processor.selected_features),
            "models_trained": list(results.keys()),
            "system_info": {
                "cpu_count": os.cpu_count(),
                "platform": os.name
            }
        }
        
        with open(results_path / "system_config.json", 'w') as f:
            json.dump(config, f, indent=4)
        
        return results_path
    
    @staticmethod
    def _make_json_serializable(obj):
        """
        Make an object JSON serializable by converting numpy types.
        
        Args:
            obj: Object to make serializable
            
        Returns:
            object: JSON serializable object
        """
        if isinstance(obj, dict):
            return {k: OptimizedIoMTClassificationSystem._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [OptimizedIoMTClassificationSystem._make_json_serializable(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def _generate_plots(self, model_name, model_results, y_test, label_encoder, plots_dir):
        """
        Generate plots for model visualization.
        
        Args:
            model_name (str): Name of the model
            model_results (dict): Model results
            y_test (ndarray): Test labels
            label_encoder (LabelEncoder): Label encoder
            plots_dir (Path): Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Confusion matrix
            plt.figure(figsize=(12, 10))
            cm = np.zeros((len(label_encoder.classes_), len(label_encoder.classes_)))
            
            # Get unique classes in predictions and test data
            predictions = model_results['predictions']
            
            # Convert predictions and true labels to class indices
            for i in range(len(y_test)):
                cm[y_test[i], predictions[i]] += 1
            
            # Plot heatmap
            plt.figure(figsize=(14, 12))
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='g', 
                cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_
            )
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name}_confusion_matrix.png", dpi=300)
            plt.close()
            
            # Performance metrics comparison
            report = model_results['report']
            metrics = {}
            
            for cls in label_encoder.classes_:
                if cls in report:
                    metrics[cls] = {
                        'precision': report[cls]['precision'],
                        'recall': report[cls]['recall'],
                        'f1-score': report[cls]['f1-score']
                    }
            
            # Create a dataframe for easier plotting
            metrics_df = pd.DataFrame(metrics).T
            
            plt.figure(figsize=(14, 8))
            metrics_df.plot(kind='bar', figsize=(14, 8))
            plt.title(f'Performance Metrics by Class - {model_name}')
            plt.ylabel('Score')
            plt.xlabel('Class')
            plt.xticks(rotation=45)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(plots_dir / f"{model_name}_class_metrics.png", dpi=300)
            plt.close()
            
        except ImportError as e:
            self.logger.warning(f"Could not generate plots: {str(e)}")
    
    def deploy_model(self, model_path, endpoint_port=8000):
        """
        Deploy a trained model as a web service for real-time inference.
        
        Args:
            model_path (str): Path to the saved model
            endpoint_port (int): Port for the model serving endpoint
            
        Returns:
            bool: Whether deployment was successful
        """
        try:
            from fastapi import FastAPI, HTTPException
            from pydantic import BaseModel
            import uvicorn
            from typing import Dict, Any
            
            # Load model data
            model_data = joblib.load(model_path)
            model = model_data["model"]
            label_encoder = model_data["label_encoder"]
            feature_columns = model_data["feature_columns"]
            
            # Load scaler
            scaler_path = "models/scaler.pkl"
            scaler = joblib.load(scaler_path)
            
            # Create FastAPI app
            app = FastAPI(title="IoMT Attack Detection Service")
            
            # Define input and output models
            class PredictionInput(BaseModel):
                features: Dict[str, float]
            
            class PredictionResult(BaseModel):
                prediction: str
                confidence: float
                probabilities: Dict[str, float]
                processing_time_ms: float
            
            # Define prediction endpoint
            @app.post("/predict", response_model=PredictionResult)
            async def predict(input_data: PredictionInput):
                start_time = time.time()
                
                try:
                    # Extract features in the correct order
                    features = np.array([[
                        input_data.features.get(col, 0.0) 
                        for col in feature_columns
                    ]])
                    
                    # Scale features
                    scaled_features = scaler.transform(features)
                    
                    # Make prediction
                    prediction_proba = model.predict_proba(scaled_features)[0]
                    prediction_idx = np.argmax(prediction_proba)
                    prediction = label_encoder.inverse_transform([prediction_idx])[0]
                    
                    # Get all class probabilities
                    probabilities = {
                        label_encoder.inverse_transform([i])[0]: float(prob)
                        for i, prob in enumerate(prediction_proba)
                    }
                    
                    # Calculate processing time
                    process_time = (time.time() - start_time) * 1000  # ms
                    
                    return PredictionResult(
                        prediction=prediction,
                        confidence=float(prediction_proba[prediction_idx]),
                        probabilities=probabilities,
                        processing_time_ms=process_time
                    )
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
            
            # Health check endpoint
            @app.get("/health")
            async def health():
                return {"status": "healthy", "model": Path(model_path).stem}
            
            # Launch the server in a separate process
            import threading
            threading.Thread(
                target=uvicorn.run,
                kwargs={"app": app, "host": "0.0.0.0", "port": endpoint_port},
                daemon=True
            ).start()
            
            self.logger.info(f"Model deployed on http://localhost:{endpoint_port}")
            print(f"Model deployed on http://localhost:{endpoint_port}")
            print(f"Use the /predict endpoint for real-time predictions")
            
            return True
        
        except ImportError as e:
            self.logger.error(f"Deployment requires FastAPI, uvicorn: {str(e)}")
            print(f"Error: {str(e)}")
            print("Please install required packages: pip install fastapi uvicorn pydantic")
            return False
        except Exception as e:
            self.logger.error(f"Deployment error: {str(e)}")
            print(f"Deployment error: {str(e)}")
            return False    
    def stream_prediction_demo(self, model_path, input_data_path, interval_ms=500):
        """
        Demonstrate real-time prediction on a stream of data.
        
        Args:
            model_path (str): Path to the saved model
            input_data_path (str): Path to test data CSV
            interval_ms (int): Interval between predictions in milliseconds
            
        Returns:
            bool: Whether demo completed successfully
        """
        try:
            # Load model data
            model_data = joblib.load(model_path)
            model = model_data["model"]
            label_encoder = model_data["label_encoder"]
            feature_columns = model_data["feature_columns"]
            
            # Load scaler
            scaler_path = "models/scaler.pkl"
            scaler = joblib.load(scaler_path)
            
            # Load some test data
            test_data = pd.read_csv(input_data_path)
            
            # Select first 50 samples for demo
            sample_data = test_data[:50]
            
            print("\nStarting real-time prediction demo")
            print("=================================")
            print(f"Model: {Path(model_path).stem}")
            print(f"Simulating {len(sample_data)} network traffic records\n")
            
            total_time = 0
            
            # Process samples with simulated time intervals
            for i, row in enumerate(sample_data.itertuples()):
                start_time = time.time()
                
                # Extract features
                features = np.array([[
                    getattr(row, col) if hasattr(row, col) else 0.0
                    for col in feature_columns
                ]])
                
                # Scale features
                scaled_features = scaler.transform(features)
                
                # Make prediction
                prediction_proba = model.predict_proba(scaled_features)[0]
                prediction_idx = np.argmax(prediction_proba)
                prediction = label_encoder.inverse_transform([prediction_idx])[0]
                confidence = prediction_proba[prediction_idx]
                
                # Calculate processing time
                process_time = (time.time() - start_time) * 1000  # ms
                total_time += process_time
                
                # Print result
                print(f"Record #{i+1} - Prediction: {prediction}")
                print(f"Confidence: {confidence:.4f}, Processing time: {process_time:.2f} ms")
                
                # Highlight potential attacks
                if prediction != 'benign':
                    print(f"ALERT: Detected {prediction} attack with {confidence:.2%} confidence!")
                
                print("-" * 50)
                
                # Simulate real-time interval
                remaining_interval = (interval_ms / 1000) - process_time/1000
                if remaining_interval > 0:
                    time.sleep(remaining_interval)
            
            avg_time = total_time / len(sample_data)
            max_throughput = 1000 / avg_time
            
            print("\nDemo completed")
            print(f"Average processing time: {avg_time:.2f} ms")
            print(f"Theoretical max throughput: {max_throughput:.0f} predictions/second")
            
            return True
        except Exception as e:
            self.logger.error(f"Stream prediction demo error: {str(e)}")
            print(f"Error: {str(e)}")
            return False


def main():
    """Main function to run the optimized IoMT classification system."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized IoMT Classification System")
    
    # Add arguments
    parser.add_argument("--data_path", type=str, 
                       default="Dataset/WiFi_and_MQTT/attacks/CSV",
                       help="Path to the data directory")
    
    parser.add_argument("--results_dir", type=str,
                       default="optimized_results",
                       help="Directory to save results")
    
    parser.add_argument("--batch_size", type=int,
                       default=50000,
                       help="Batch size for processing")
    
    parser.add_argument("--max_per_class", type=int,
                       default=25000,
                       help="Maximum samples per class after balancing")
    
    parser.add_argument("--no_feature_selection", action="store_true",
                       help="Disable feature selection")
    
    parser.add_argument("--no_hyperparameter_tuning", action="store_true",
                       help="Disable hyperparameter tuning")
    
    parser.add_argument("--max_runtime_hours", type=float,
                       default=2.0,
                       help="Maximum runtime in hours")
    
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU acceleration if available")
    
    parser.add_argument("--deploy", action="store_true",
                       help="Deploy model after training")
    
    parser.add_argument("--port", type=int,
                       default=8000,
                       help="Port for model deployment")
    
    parser.add_argument("--demo", action="store_true",
                       help="Run stream prediction demo")
    
    args = parser.parse_args()
    
    try:
        # Create results directory if it doesn't exist
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize classification system
        system = OptimizedIoMTClassificationSystem(
            batch_size=args.batch_size,
            max_files=None,
            n_jobs=None,  # Auto-detect
            max_runtime_hours=args.max_runtime_hours,
            use_gpu=args.use_gpu,
            max_per_class=args.max_per_class
        )
        
        # Run classification workflow
        results_path = system.run(
            data_path=args.data_path,
            results_dir=args.results_dir,
            feature_selection=not args.no_feature_selection,
            hyperparameter_tuning=not args.no_hyperparameter_tuning
        )
        
        # Find best model
        best_model_path = None
        best_f1 = 0
        
        for model_file in (results_path / "models").glob("*_model.pkl"):
            metrics_file = results_path / "metrics" / f"{model_file.stem.replace('_model', '')}_metrics.json"
            
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    if metrics["macro_f1"] > best_f1:
                        best_f1 = metrics["macro_f1"]
                        best_model_path = model_file
        
        if best_model_path:
            print(f"\nBest model: {best_model_path.stem}")
            print(f"Macro F1 Score: {best_f1:.4f}")
            
            # Deploy model if requested
            if args.deploy:
                system.deploy_model(best_model_path, args.port)
            
            # Run demo if requested
            if args.demo:
                # Find a test data file from the original dataset
                test_data_path = Path(args.data_path) / "test" / "Benign_test.pcap.csv"
                if not test_data_path.exists():
                    test_data_path = list(Path(args.data_path).glob("**/*.csv"))[0]
                
                system.stream_prediction_demo(best_model_path, test_data_path)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        logging.error(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()