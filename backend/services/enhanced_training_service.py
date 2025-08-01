"""
Enhanced Training Service
Handles comprehensive algorithm training with sklearn comparison and rich metrics
"""
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes, load_iris, load_wine, load_breast_cancer
import logging

from models.enhanced_results import (
    EnhancedTrainingResult, AlgorithmResult, AlgorithmMetrics,
    ChartData, ComparisonSummary, DatasetInfo
)
from sklearn_integration.comparison import sklearn_comparison, SklearnMapper
from core_integration.discovery import core_discovery

logger = logging.getLogger(__name__)

class DatasetService:
    """Handles loading and preprocessing of datasets"""
    
    DATASETS = {
        "diabetes": {
            "loader": load_diabetes,
            "type": "regression",
            "description": "Diabetes progression prediction",
            "target_name": "progression"
        },
        "iris": {
            "loader": load_iris,
            "type": "classification", 
            "description": "Iris species classification",
            "target_name": "species"
        },
        "wine": {
            "loader": load_wine,
            "type": "classification",
            "description": "Wine quality classification", 
            "target_name": "wine_class"
        },
        "breast_cancer": {
            "loader": load_breast_cancer,
            "type": "classification",
            "description": "Breast cancer diagnosis",
            "target_name": "diagnosis"
        }
    }
    
    @classmethod
    def load_dataset(cls, dataset_name: str, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DatasetInfo]:
        """
        Load and split dataset
        
        Returns:
            X_train, X_test, y_train, y_test, dataset_info
        """
        if dataset_name not in cls.DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(cls.DATASETS.keys())}")
        
        dataset_config = cls.DATASETS[dataset_name]
        data = dataset_config["loader"](return_X_y=False)
        
        X, y = data.data, data.target
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Create dataset info
        dataset_info = DatasetInfo(
            name=dataset_name,
            n_samples=X.shape[0],
            n_features=X.shape[1],
            feature_names=getattr(data, 'feature_names', [f"feature_{i}" for i in range(X.shape[1])]),
            target_name=dataset_config["target_name"],
            train_size=X_train.shape[0],
            test_size=X_test.shape[0],
            split_ratio=test_size
        )
        
        return X_train, X_test, y_train, y_test, dataset_info

class ChartDataGenerator:
    """Generates data for various chart types based on algorithm results"""
    
    @staticmethod
    def generate_regression_charts(
        X_test: np.ndarray,
        y_test: np.ndarray,
        your_predictions: np.ndarray,
        sklearn_predictions: np.ndarray,
        your_model: Any,
        dataset_info: DatasetInfo
    ) -> ChartData:
        """Generate chart data for regression algorithms"""
        
        charts = ChartData()
        
        # Helper function to safely convert to list
        def safe_tolist(obj):
            return obj.tolist() if hasattr(obj, 'tolist') else obj
        
        # Scatter plot: actual vs predictions
        charts.scatter_plot = {
            "actual": safe_tolist(y_test),
            "your_predictions": safe_tolist(your_predictions),
            "sklearn_predictions": safe_tolist(sklearn_predictions),
            "x_label": "Actual Values",
            "y_label": "Predicted Values",
            "title": f"Predictions vs Actual - {dataset_info.name.title()}"
        }
        
        # Loss curve from training history
        if hasattr(your_model, 'loss_history') and your_model.loss_history:
            charts.loss_curve = {
                "iterations": list(range(len(your_model.loss_history))),
                "loss": your_model.loss_history,
                "x_label": "Iteration",
                "y_label": "Loss (MSE)",
                "title": "Training Loss Curve"
            }
        
        # Residuals plot
        your_residuals = y_test - your_predictions
        sklearn_residuals = y_test - sklearn_predictions
        
        charts.residuals_plot = {
            "your_predictions": safe_tolist(your_predictions),
            "your_residuals": safe_tolist(your_residuals),
            "sklearn_predictions": safe_tolist(sklearn_predictions),
            "sklearn_residuals": safe_tolist(sklearn_residuals),
            "x_label": "Predicted Values",
            "y_label": "Residuals",
            "title": "Residual Analysis"
        }
        
        # If single feature, create regression line plot
        if X_test.shape[1] == 1:
            x_range = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1, 1)
            your_line = your_model.predict(x_range)
            
            charts.regression_line = {
                "x_data": safe_tolist(X_test.flatten()),
                "y_data": safe_tolist(y_test),
                "x_line": safe_tolist(x_range.flatten()),
                "y_line": safe_tolist(your_line),
                "x_label": dataset_info.feature_names[0] if dataset_info.feature_names else "Feature",
                "y_label": dataset_info.target_name,
                "title": f"Linear Regression Fit - {dataset_info.name.title()}"
            }
        
        return charts
    
    @staticmethod
    def generate_classification_charts(
        X_test: np.ndarray,
        y_test: np.ndarray,
        your_predictions: np.ndarray,
        sklearn_predictions: np.ndarray,
        dataset_info: DatasetInfo
    ) -> ChartData:
        """Generate chart data for classification algorithms"""
        
        charts = ChartData()
        
        # Helper function to safely convert to list
        def safe_tolist(obj):
            return obj.tolist() if hasattr(obj, 'tolist') else obj
        
        # Confusion matrix for your implementation
        from sklearn.metrics import confusion_matrix
        
        your_cm = confusion_matrix(y_test, your_predictions)
        sklearn_cm = confusion_matrix(y_test, sklearn_predictions)
        
        charts.confusion_matrix = {
            "your_matrix": safe_tolist(your_cm),
            "sklearn_matrix": safe_tolist(sklearn_cm),
            "labels": [f"Class {i}" for i in range(len(your_cm))]
        }
        
        # If 2D data, create decision boundary visualization (simplified)
        if X_test.shape[1] == 2:
            charts.decision_boundary = {
                "x_data": safe_tolist(X_test[:, 0]),
                "y_data": safe_tolist(X_test[:, 1]),
                "labels": safe_tolist(y_test),
                "x_label": dataset_info.feature_names[0] if dataset_info.feature_names else "Feature 1",
                "y_label": dataset_info.feature_names[1] if dataset_info.feature_names else "Feature 2",
                "title": f"Classification Results - {dataset_info.name.title()}"
            }
        
        return charts
    
    @staticmethod
    def generate_clustering_charts(
        X_test: np.ndarray,
        your_predictions: np.ndarray,
        sklearn_predictions: np.ndarray,
        your_model: Any,
        sklearn_model: Any,
        dataset_info: DatasetInfo
    ) -> ChartData:
        """Generate chart data for clustering algorithms"""
        
        charts = ChartData()
        
        # Helper function to safely convert to list
        def safe_tolist(obj):
            return obj.tolist() if hasattr(obj, 'tolist') else obj
        
        # Cluster visualization (using first 2 features)
        charts.cluster_plot = {
            "x_data": safe_tolist(X_test[:, 0]),
            "y_data": safe_tolist(X_test[:, 1]) if X_test.shape[1] > 1 else [0] * len(X_test),
            "your_labels": safe_tolist(your_predictions),
            "sklearn_labels": safe_tolist(sklearn_predictions),
            "x_label": dataset_info.feature_names[0] if dataset_info.feature_names else "Feature 1",
            "y_label": dataset_info.feature_names[1] if dataset_info.feature_names and len(dataset_info.feature_names) > 1 else "Feature 2",
            "title": f"Clustering Results - {dataset_info.name.title()}"
        }
        
        # Cluster centers
        if hasattr(your_model, 'cluster_centers_'):
            charts.cluster_centers = safe_tolist(your_model.cluster_centers_)
        
        return charts

class EnhancedTrainingService:
    """Enhanced training service with sklearn comparison and rich visualization"""
    
    def __init__(self):
        self.dataset_service = DatasetService()
        self.chart_generator = ChartDataGenerator()
    
    async def train_with_comparison(
        self,
        algorithm_id: str,
        hyperparameters: Dict[str, Any],
        dataset_name: str = "diabetes"
    ) -> EnhancedTrainingResult:
        """
        Train algorithm with sklearn comparison and generate comprehensive results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Extract test_size from hyperparameters
            test_size = hyperparameters.pop('test_size', 0.2)
            
            # Load dataset
            X_train, X_test, y_train, y_test, dataset_info = self.dataset_service.load_dataset(
                dataset_name, test_size=test_size
            )
            
            # Get algorithm type
            algorithm_type = SklearnMapper.get_algorithm_type(algorithm_id)
            
            # Train your implementation
            your_result = await self._train_your_algorithm(
                algorithm_id, hyperparameters, X_train, y_train, X_test, y_test, algorithm_type
            )
            
            # Train sklearn equivalent
            sklearn_result = self._train_sklearn_algorithm(
                algorithm_id, hyperparameters, X_train, y_train, X_test, y_test
            )
            
            # Generate comparison summary
            comparison = self._generate_comparison_summary(
                your_result, sklearn_result, algorithm_type
            )
            
            # Generate chart data
            charts = self._generate_charts(
                algorithm_type, X_test, y_test, your_result, sklearn_result, dataset_info
            )
            
            total_duration = time.time() - start_time
            
            return EnhancedTrainingResult(
                success=True,
                algorithm_id=algorithm_id,
                algorithm_type=algorithm_type,
                dataset=dataset_info,
                hyperparameters=hyperparameters,
                your_implementation=self._create_algorithm_result(your_result),
                sklearn_implementation=self._create_algorithm_result(sklearn_result),
                comparison=comparison,
                charts=charts,
                total_duration=total_duration,
                timestamp=timestamp
            )
            
        except Exception as e:
            logger.error(f"Training failed for {algorithm_id}: {e}")
            logger.error(traceback.format_exc())
            
            return EnhancedTrainingResult(
                success=False,
                algorithm_id=algorithm_id,
                algorithm_type="unknown",
                dataset=DatasetInfo(name=dataset_name, n_samples=0, n_features=0, train_size=0, test_size=0, split_ratio=0.2),
                hyperparameters=hyperparameters,
                your_implementation=AlgorithmResult(metrics=AlgorithmMetrics(training_time=0.0), predictions=[]),
                sklearn_implementation=AlgorithmResult(metrics=AlgorithmMetrics(training_time=0.0), predictions=[]),
                comparison=ComparisonSummary(performance_differences={}, speed_comparison={}),
                charts=ChartData(),
                total_duration=time.time() - start_time,
                timestamp=timestamp,
                error=str(e)
            )
    
    async def _train_your_algorithm(
        self,
        algorithm_id: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        algorithm_type: str
    ) -> Dict[str, Any]:
        """Train your algorithm implementation"""
        
        # Get your algorithm class
        algorithm_class = core_discovery.get_algorithm_class(algorithm_id)
        if algorithm_class is None:
            raise ValueError(f"Algorithm {algorithm_id} not found")
        
        # Create and train your model
        your_model = algorithm_class(**hyperparameters)
        
        start_time = time.time()
        if algorithm_type == "clustering":
            your_model.fit(X_train)  # No y for clustering
        else:
            your_model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        your_predictions = your_model.predict(X_test)
        
        # Calculate metrics
        metrics = self._calculate_your_metrics(
            algorithm_type, y_test, your_predictions, training_time, your_model, X_test
        )
        
        return {
            "model": your_model,
            "predictions": your_predictions,
            "metrics": metrics,
            "training_time": training_time
        }
    
    def _train_sklearn_algorithm(
        self,
        algorithm_id: str,
        hyperparameters: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, Any]:
        """Train sklearn equivalent"""
        
        return sklearn_comparison.train_sklearn_equivalent(
            algorithm_id, X_train, y_train, X_test, y_test, hyperparameters
        )
    
    def _calculate_your_metrics(
        self,
        algorithm_type: str,
        y_test: np.ndarray,
        predictions: np.ndarray,
        training_time: float,
        model: Any,
        X_test: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate metrics for your algorithm"""
        
        # Use the same metric calculation as sklearn comparison
        metrics = sklearn_comparison._calculate_metrics(
            algorithm_type, y_test, predictions, training_time, X_test
        )
        
        # Add convergence information if available
        if hasattr(model, 'loss_history') and model.loss_history:
            metrics["convergence_iterations"] = len(model.loss_history)
            # Check if converged (loss stabilized)
            if len(model.loss_history) > 1:
                last_losses = model.loss_history[-5:]  # Last 5 iterations
                if len(last_losses) >= 2:
                    loss_std = np.std(last_losses)
                    metrics["converged"] = loss_std < 1e-6
        
        return metrics
    
    def _create_algorithm_result(self, result_dict: Dict[str, Any]) -> AlgorithmResult:
        """Convert result dictionary to AlgorithmResult model"""
        
        metrics_dict = result_dict["metrics"]
        algorithm_metrics = AlgorithmMetrics(**metrics_dict)
        
        # Convert numpy arrays to Python lists
        predictions = result_dict["predictions"]
        if hasattr(predictions, 'tolist'):
            predictions = predictions.tolist()
        
        algorithm_result = AlgorithmResult(
            metrics=algorithm_metrics,
            predictions=predictions
        )
        
        # Add optional fields if they exist, converting numpy arrays to lists
        if "coefficients" in result_dict:
            coefficients = result_dict["coefficients"]
            if hasattr(coefficients, 'tolist'):
                coefficients = coefficients.tolist()
            algorithm_result.coefficients = coefficients
            
        if "intercept" in result_dict:
            algorithm_result.intercept = result_dict["intercept"]
            
        if "feature_importance" in result_dict:
            feature_importance = result_dict["feature_importance"]
            if hasattr(feature_importance, 'tolist'):
                feature_importance = feature_importance.tolist()
            algorithm_result.feature_importance = feature_importance
        
        # Add training history if available
        model = result_dict.get("model")
        if model and hasattr(model, 'loss_history'):
            training_history = model.loss_history
            if hasattr(training_history, 'tolist'):
                training_history = training_history.tolist()
            algorithm_result.training_history = training_history
        
        return algorithm_result
    
    def _generate_comparison_summary(
        self,
        your_result: Dict[str, Any],
        sklearn_result: Dict[str, Any],
        algorithm_type: str
    ) -> ComparisonSummary:
        """Generate comparison summary between implementations"""
        
        your_metrics = your_result["metrics"]
        sklearn_metrics = sklearn_result["metrics"]
        
        # Calculate performance differences
        performance_differences = {}
        common_metrics = set(your_metrics.keys()) & set(sklearn_metrics.keys())
        
        for metric in common_metrics:
            if isinstance(your_metrics[metric], (int, float)) and isinstance(sklearn_metrics[metric], (int, float)):
                diff = your_metrics[metric] - sklearn_metrics[metric]
                performance_differences[metric] = diff
        
        # Speed comparison
        speed_comparison = {
            "your_implementation": your_result["training_time"],
            "sklearn_implementation": sklearn_result["training_time"],
            "speed_ratio": your_result["training_time"] / max(sklearn_result["training_time"], 1e-10)
        }
        
        # Generate recommendation
        recommendation = self._generate_recommendation(performance_differences, speed_comparison, algorithm_type)
        
        return ComparisonSummary(
            performance_differences=performance_differences,
            speed_comparison=speed_comparison,
            recommendation=recommendation
        )
    
    def _generate_recommendation(
        self,
        performance_diff: Dict[str, float],
        speed_comparison: Dict[str, float],
        algorithm_type: str
    ) -> str:
        """Generate recommendation text based on comparison"""
        
        # Determine primary metric based on algorithm type
        primary_metrics = {
            "regression": "r2_score",
            "classification": "accuracy", 
            "clustering": "silhouette_score"
        }
        
        primary_metric = primary_metrics.get(algorithm_type)
        
        if primary_metric and primary_metric in performance_diff:
            perf_diff = performance_diff[primary_metric]
            speed_ratio = speed_comparison["speed_ratio"]
            
            if abs(perf_diff) < 0.01:  # Very similar performance
                if speed_ratio < 2.0:
                    return "Excellent! Your implementation performs similarly to sklearn with comparable speed."
                else:
                    return "Good implementation! Performance matches sklearn, though sklearn is faster due to optimizations."
            elif perf_diff > 0:  # Your implementation is better
                return "Outstanding! Your implementation outperforms sklearn on this dataset."
            else:  # sklearn is better
                return "Your implementation is working correctly. Performance differences may be due to different optimization techniques."
        
        return "Both implementations are working correctly. Each has its own strengths."
    
    def _generate_charts(
        self,
        algorithm_type: str,
        X_test: np.ndarray,
        y_test: np.ndarray,
        your_result: Dict[str, Any],
        sklearn_result: Dict[str, Any],
        dataset_info: DatasetInfo
    ) -> ChartData:
        """Generate appropriate charts based on algorithm type"""
        
        your_predictions = your_result["predictions"]
        sklearn_predictions = sklearn_result["predictions"]
        your_model = your_result["model"]
        sklearn_model = sklearn_result.get("model")
        
        if algorithm_type == "regression":
            return self.chart_generator.generate_regression_charts(
                X_test, y_test, your_predictions, sklearn_predictions, your_model, dataset_info
            )
        elif algorithm_type == "classification":
            return self.chart_generator.generate_classification_charts(
                X_test, y_test, your_predictions, sklearn_predictions, dataset_info
            )
        elif algorithm_type == "clustering":
            return self.chart_generator.generate_clustering_charts(
                X_test, your_predictions, sklearn_predictions, your_model, sklearn_model, dataset_info
            )
        else:
            return ChartData()

# Global instance
enhanced_training_service = EnhancedTrainingService()
