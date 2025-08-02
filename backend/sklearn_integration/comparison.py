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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import PolynomialFeatures
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
        # ===== REGRESSION ALGORITHMS =====
        "linear_regression": {
            "sklearn_class": LinearRegression,
            "algorithm_type": "regression",
            "param_mapping": {
                "alpha": None,  # Sklearn LinearRegression doesn't use learning rate
                "n_iters": None,  # Sklearn uses different solver
                "tolerance": None,  # Sklearn has different convergence criteria
                "fit_intercept": "fit_intercept"
            },
            "default_sklearn_params": {
                "fit_intercept": True,
                "copy_X": True
            }
        },
        "ridge_regression": {
            "sklearn_class": Ridge,
            "algorithm_type": "regression",
            "param_mapping": {
                "alpha": "alpha",
                "fit_intercept": "fit_intercept",
                "max_iter": "max_iter",
                "tol": "tol"
            },
            "default_sklearn_params": {
                "alpha": 1.0,
                "fit_intercept": True,
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "lasso_regression": {
            "sklearn_class": Lasso,
            "algorithm_type": "regression",
            "param_mapping": {
                "alpha": "alpha",
                "fit_intercept": "fit_intercept",
                "max_iter": "max_iter",
                "tol": "tol"
            },
            "default_sklearn_params": {
                "alpha": 1.0,
                "fit_intercept": True,
                "max_iter": 1000,
                "random_state": 42
            }
        },
        "polynomial_regression": {
            "sklearn_class": PolynomialFeatures,
            "algorithm_type": "regression",
            "param_mapping": {
                "degree": "degree",
                "include_bias": "include_bias",
                "interaction_only": "interaction_only"
            },
            "default_sklearn_params": {
                "degree": 2,
                "include_bias": True,
                "interaction_only": False
            }
        },
        "decision_tree_regressor": {
            "sklearn_class": DecisionTreeRegressor,
            "algorithm_type": "regression",
            "param_mapping": {
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split",
                "min_samples_leaf": "min_samples_leaf",
                "max_features": "max_features"
            },
            "default_sklearn_params": {
                "random_state": 42,
                "max_depth": None
            }
        },
        "random_forest_regressor": {
            "sklearn_class": RandomForestRegressor,
            "algorithm_type": "regression",
            "param_mapping": {
                "n_estimators": "n_estimators",
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split",
                "min_samples_leaf": "min_samples_leaf"
            },
            "default_sklearn_params": {
                "n_estimators": 100,
                "random_state": 42
            }
        },
        "svr": {
            "sklearn_class": SVR,
            "algorithm_type": "regression",
            "param_mapping": {
                "C": "C",
                "kernel": "kernel",
                "gamma": "gamma",
                "epsilon": "epsilon"
            },
            "default_sklearn_params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale"
            }
        },
        "knn_regressor": {
            "sklearn_class": KNeighborsRegressor,
            "algorithm_type": "regression",
            "param_mapping": {
                "n_neighbors": "n_neighbors",
                "weights": "weights",
                "metric": "metric"
            },
            "default_sklearn_params": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski"
            }
        },
        
        # ===== CLASSIFICATION ALGORITHMS =====
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
        "decision_tree": {
            "sklearn_class": DecisionTreeClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split",
                "min_samples_leaf": "min_samples_leaf",
                "max_features": "max_features",
                "criterion": "criterion"
            },
            "default_sklearn_params": {
                "random_state": 42,
                "criterion": "gini"
            }
        },
        "random_forest": {
            "sklearn_class": RandomForestClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "n_estimators": "n_estimators",
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split",
                "min_samples_leaf": "min_samples_leaf",
                "max_features": "max_features"
            },
            "default_sklearn_params": {
                "n_estimators": 100,
                "random_state": 42
            }
        },
        "svm": {
            "sklearn_class": SVC,
            "algorithm_type": "classification",
            "param_mapping": {
                "C": "C",
                "kernel": "kernel",
                "gamma": "gamma",
                "degree": "degree"
            },
            "default_sklearn_params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
                "random_state": 42
            }
        },
        "knn": {
            "sklearn_class": KNeighborsClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "n_neighbors": "n_neighbors",
                "weights": "weights",
                "metric": "metric"
            },
            "default_sklearn_params": {
                "n_neighbors": 5,
                "weights": "uniform",
                "metric": "minkowski"
            }
        },
        "naive_bayes": {
            "sklearn_class": GaussianNB,
            "algorithm_type": "classification",
            "param_mapping": {
                "var_smoothing": "var_smoothing"
            },
            "default_sklearn_params": {
                "var_smoothing": 1e-9
            }
        },
        "naive_bayes_multinomial": {
            "sklearn_class": MultinomialNB,
            "algorithm_type": "classification",
            "param_mapping": {
                "alpha": "alpha",
                "fit_prior": "fit_prior"
            },
            "default_sklearn_params": {
                "alpha": 1.0,
                "fit_prior": True
            }
        },
        "naive_bayes_bernoulli": {
            "sklearn_class": BernoulliNB,
            "algorithm_type": "classification",
            "param_mapping": {
                "alpha": "alpha",
                "binarize": "binarize",
                "fit_prior": "fit_prior"
            },
            "default_sklearn_params": {
                "alpha": 1.0,
                "binarize": 0.0,
                "fit_prior": True
            }
        },
        "gradient_boosting": {
            "sklearn_class": GradientBoostingClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "n_estimators": "n_estimators",
                "learning_rate": "learning_rate",
                "max_depth": "max_depth",
                "min_samples_split": "min_samples_split"
            },
            "default_sklearn_params": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42
            }
        },
        "ada_boost": {
            "sklearn_class": AdaBoostClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "n_estimators": "n_estimators",
                "learning_rate": "learning_rate",
                "algorithm": "algorithm"
            },
            "default_sklearn_params": {
                "n_estimators": 50,
                "learning_rate": 1.0,
                "random_state": 42
            }
        },
        
        # ===== NEURAL NETWORKS =====
        "neural_network": {
            "sklearn_class": MLPClassifier,
            "algorithm_type": "classification",
            "param_mapping": {
                "hidden_layer_sizes": "hidden_layer_sizes",
                "activation": "activation",
                "solver": "solver",
                "alpha": "alpha",
                "learning_rate": "learning_rate",
                "max_iter": "max_iter"
            },
            "default_sklearn_params": {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "learning_rate": "constant",
                "max_iter": 200,
                "random_state": 42
            }
        },
        "neural_network_regressor": {
            "sklearn_class": MLPRegressor,
            "algorithm_type": "regression",
            "param_mapping": {
                "hidden_layer_sizes": "hidden_layer_sizes",
                "activation": "activation",
                "solver": "solver",
                "alpha": "alpha",
                "learning_rate": "learning_rate",
                "max_iter": "max_iter"
            },
            "default_sklearn_params": {
                "hidden_layer_sizes": (100,),
                "activation": "relu",
                "solver": "adam",
                "alpha": 0.0001,
                "learning_rate": "constant",
                "max_iter": 200,
                "random_state": 42
            }
        },
        
        # ===== CLUSTERING ALGORITHMS =====
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
        },
        "dbscan": {
            "sklearn_class": DBSCAN,
            "algorithm_type": "clustering",
            "param_mapping": {
                "eps": "eps",
                "min_samples": "min_samples",
                "metric": "metric"
            },
            "default_sklearn_params": {
                "eps": 0.5,
                "min_samples": 5,
                "metric": "euclidean"
            }
        },
        "hierarchical": {
            "sklearn_class": AgglomerativeClustering,
            "algorithm_type": "clustering",
            "param_mapping": {
                "n_clusters": "n_clusters",
                "linkage": "linkage",
                "metric": "metric"
            },
            "default_sklearn_params": {
                "n_clusters": 2,
                "linkage": "ward"
            }
        },
        
        # ===== DIMENSIONALITY REDUCTION =====
        "pca": {
            "sklearn_class": PCA,
            "algorithm_type": "dimensionality_reduction",
            "param_mapping": {
                "n_components": "n_components",
                "whiten": "whiten",
                "svd_solver": "svd_solver"
            },
            "default_sklearn_params": {
                "n_components": None,
                "whiten": False,
                "svd_solver": "auto",
                "random_state": 42
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
                # Handle both scalar and array intercepts
                intercept = sklearn_model.intercept_
                if hasattr(intercept, '__len__') and len(intercept) == 1:
                    result["intercept"] = float(intercept[0])
                elif hasattr(intercept, '__len__'):
                    result["intercept"] = intercept.tolist()
                else:
                    result["intercept"] = float(intercept)
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
                
            elif algorithm_type in ["classification", "binary_classification", "multiclass_classification"]:
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
