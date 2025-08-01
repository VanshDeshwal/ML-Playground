"""
Sklearn Integration and Comparison System
Maps our algorithms to sklearn equivalents and handles comparison logic
"""
import importlib
import inspect
import time
from typing import Dict, Any, Optional, Tuple, Type
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, calinski_harabasz_score
)
import logging

logger = logging.getLogger(__name__)

class SklearnMapper:
    """Maps our algorithm implementations to sklearn equivalents"""
    
    # Algorithm mapping configuration
    ALGORITHM_MAPPINGS = {
        "linear_regression": {
            "sklearn_class": LinearRegression,
            "algorithm_type": "regression",
            "param_mapping": {
                # our_param: sklearn_param (None if sklearn doesn't have it)
                "alpha": None,  # Sklearn LinearRegression doesn't use learning rate
                "n_iters": None,  # Sklearn uses different solver
                "tolerance": None,  # Sklearn has different convergence criteria
                "fit_intercept": "fit_intercept"  # Direct mapping when available
            },
            "default_sklearn_params": {
                "fit_intercept": True,
                "copy_X": True
            }
        },
        "logistic_regression": {
            "sklearn_class": LogisticRegression,
            "algorithm_type": "classification", 
            "param_mapping": {
                "alpha": None,  # Different from sklearn's learning rate
                "n_iters": "max_iter",
                "tolerance": "tol",
                "regularization": "C"  # Note: sklearn uses inverse (1/C)
            },
            "default_sklearn_params": {
                "max_iter": 1000,
                "random_state": 42,
                "solver": "lbfgs"
            }
        },
        "kmeans": {
            "sklearn_class": KMeans,
            "algorithm_type": "clustering",
            "param_mapping": {
                "n_clusters": "n_clusters",
                "n_iters": "max_iter", 
                "tolerance": "tol",
                "random_state": "random_state",
                "init_method": "init"
            },
            "default_sklearn_params": {
                "random_state": 42,
                "n_init": 10
            }
        }
    }
    
    @classmethod
    def get_sklearn_equivalent(cls, algorithm_id: str) -> Optional[Tuple[Type, Dict[str, Any]]]:
        """
        Get sklearn equivalent class and parameter mapping
        
        Args:
            algorithm_id: Our algorithm identifier
            
        Returns:
            Tuple of (sklearn_class, param_mapping) or None if not found
        """
        if algorithm_id not in cls.ALGORITHM_MAPPINGS:
            logger.warning(f"No sklearn mapping found for algorithm: {algorithm_id}")
            return None
            
        mapping = cls.ALGORITHM_MAPPINGS[algorithm_id]
        return mapping["sklearn_class"], mapping
    
    @classmethod
    def map_hyperparameters(cls, algorithm_id: str, our_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Map our hyperparameters to sklearn equivalents
        
        Args:
            algorithm_id: Our algorithm identifier
            our_params: Our algorithm's hyperparameters
            
        Returns:
            Dictionary of sklearn-compatible parameters
        """
        if algorithm_id not in cls.ALGORITHM_MAPPINGS:
            return {}
            
        mapping_config = cls.ALGORITHM_MAPPINGS[algorithm_id]
        param_mapping = mapping_config["param_mapping"]
        sklearn_params = mapping_config["default_sklearn_params"].copy()
        
        # Map our parameters to sklearn parameters
        for our_param, sklearn_param in param_mapping.items():
            if our_param in our_params and sklearn_param is not None:
                value = our_params[our_param]
                
                # Special handling for specific parameters
                if sklearn_param == "C" and our_param == "regularization":
                    # sklearn's C is inverse of regularization strength
                    # Avoid division by zero
                    value = 1.0 / max(value, 1e-10) if value > 0 else 1.0
                
                sklearn_params[sklearn_param] = value
        
        logger.info(f"Mapped parameters for {algorithm_id}: {our_params} -> {sklearn_params}")
        return sklearn_params
    
    @classmethod
    def get_algorithm_type(cls, algorithm_id: str) -> str:
        """Get the algorithm type (regression, classification, clustering)"""
        if algorithm_id in cls.ALGORITHM_MAPPINGS:
            return cls.ALGORITHM_MAPPINGS[algorithm_id]["algorithm_type"]
        return "unknown"

class SklearnComparison:
    """Handles training and comparison with sklearn implementations"""
    
    def __init__(self):
        self.mapper = SklearnMapper()
    
    def train_sklearn_equivalent(
        self,
        algorithm_id: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        hyperparameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Train sklearn equivalent of our algorithm
        
        Returns:
            Dictionary with sklearn results
        """
        try:
            # Get sklearn equivalent
            sklearn_info = self.mapper.get_sklearn_equivalent(algorithm_id)
            if sklearn_info is None:
                raise ValueError(f"No sklearn equivalent found for {algorithm_id}")
            
            sklearn_class, mapping_config = sklearn_info
            sklearn_params = self.mapper.map_hyperparameters(algorithm_id, hyperparameters)
            
            # Create and train sklearn model
            sklearn_model = sklearn_class(**sklearn_params)
            
            start_time = time.time()
            if mapping_config["algorithm_type"] == "clustering":
                # Clustering doesn't use y_train
                sklearn_model.fit(X_train)
                sklearn_predictions = sklearn_model.predict(X_test)
            else:
                sklearn_model.fit(X_train, y_train)
                sklearn_predictions = sklearn_model.predict(X_test)
            training_time = time.time() - start_time
            
            # Calculate metrics
            algorithm_type = mapping_config["algorithm_type"]
            metrics = self._calculate_metrics(
                algorithm_type, y_test, sklearn_predictions, training_time
            )
            
            # Extract model information
            result = {
                "metrics": metrics,
                "predictions": sklearn_predictions.tolist(),
                "training_time": training_time,
                "model": sklearn_model
            }
            
            # Add algorithm-specific information
            if hasattr(sklearn_model, 'coef_'):
                result["coefficients"] = sklearn_model.coef_.flatten().tolist()
            if hasattr(sklearn_model, 'intercept_'):
                result["intercept"] = float(sklearn_model.intercept_)
            if hasattr(sklearn_model, 'feature_importances_'):
                result["feature_importance"] = sklearn_model.feature_importances_.tolist()
            if hasattr(sklearn_model, 'cluster_centers_'):
                result["cluster_centers"] = sklearn_model.cluster_centers_.tolist()
            
            return result
            
        except Exception as e:
            logger.error(f"Error training sklearn equivalent for {algorithm_id}: {e}")
            raise
    
    def _calculate_metrics(
        self,
        algorithm_type: str,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        training_time: float,
        X: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """Calculate appropriate metrics based on algorithm type"""
        
        metrics = {"training_time": training_time}
        
        try:
            if algorithm_type == "regression":
                metrics.update({
                    "r2_score": float(r2_score(y_true, y_pred)),
                    "mse": float(mean_squared_error(y_true, y_pred)),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred)))
                })
                
            elif algorithm_type == "classification":
                # Handle binary and multiclass
                average = 'binary' if len(np.unique(y_true)) == 2 else 'weighted'
                
                metrics.update({
                    "accuracy": float(accuracy_score(y_true, y_pred)),
                    "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
                    "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
                    "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0))
                })
                
            elif algorithm_type == "clustering" and X is not None:
                # For clustering, y_pred contains cluster labels
                if len(np.unique(y_pred)) > 1:  # Need at least 2 clusters for silhouette
                    metrics.update({
                        "silhouette_score": float(silhouette_score(X, y_pred)),
                        "calinski_harabasz_score": float(calinski_harabasz_score(X, y_pred))
                    })
                
        except Exception as e:
            logger.warning(f"Error calculating some metrics: {e}")
            
        return metrics

# Global instance
sklearn_comparison = SklearnComparison()
