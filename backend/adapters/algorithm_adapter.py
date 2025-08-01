"""
Universal Algorithm Adapter
Provides a consistent interface for all algorithms regardless of their implementation
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import logging
import numpy as np
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)

@dataclass
class TrainingResult:
    """Standardized training result"""
    success: bool
    duration: float
    metrics: Dict[str, Any]
    model: Any = None
    predictions: Optional[np.ndarray] = None
    error: Optional[str] = None

class BaseAlgorithmAdapter(ABC):
    """Base adapter interface for all algorithms"""
    
    @abstractmethod
    async def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TrainingResult:
        """Train the algorithm"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata"""
        pass

class UniversalAlgorithmAdapter(BaseAlgorithmAdapter):
    """
    Universal adapter that can work with any algorithm from the core directory
    """
    
    def __init__(self, algorithm_id: str, algorithm_info: Dict[str, Any]):
        self.algorithm_id = algorithm_id
        self.algorithm_info = algorithm_info
        self.algorithm_instance = None
        self.is_trained = False
        self._training_results = None
        
        # Import here to avoid circular imports
        from core_integration.loader import core_loader
        self.loader = core_loader
    
    async def train(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TrainingResult:
        """Train the algorithm with universal interface"""
        start_time = time.time()
        
        try:
            # Create algorithm instance with hyperparameters
            self.algorithm_instance = self.loader.create_instance(self.algorithm_id, **kwargs)
            
            if self.algorithm_instance is None:
                return TrainingResult(
                    success=False,
                    duration=time.time() - start_time,
                    metrics={},
                    error=f"Failed to create instance of {self.algorithm_id}"
                )
            
            # Train the algorithm
            training_result = await self._train_algorithm(X, y, **kwargs)
            
            duration = time.time() - start_time
            self.is_trained = training_result.success
            self._training_results = training_result
            
            # Add duration to metrics
            training_result.duration = duration
            training_result.metrics["training_duration"] = duration
            
            return training_result
            
        except Exception as e:
            logger.error(f"Training error for {self.algorithm_id}: {e}")
            return TrainingResult(
                success=False,
                duration=time.time() - start_time,
                metrics={},
                error=str(e)
            )
    
    async def _train_algorithm(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> TrainingResult:
        """Internal training method that handles different algorithm types"""
        
        try:
            if hasattr(self.algorithm_instance, "train"):
                # Class-based algorithm with train method
                result = self.algorithm_instance.train(X, y, **kwargs)
            elif hasattr(self.algorithm_instance, "fit"):
                # Scikit-learn style algorithm
                result = self.algorithm_instance.fit(X, y, **kwargs)
                result = {"model": self.algorithm_instance}
            elif callable(self.algorithm_instance):
                # Function-based algorithm
                result = self.algorithm_instance(X, y, **kwargs)
            else:
                raise ValueError(f"Don't know how to train {type(self.algorithm_instance)}")
            
            # Standardize the result
            if isinstance(result, dict):
                metrics = result.copy()
                model = metrics.pop("model", self.algorithm_instance)
            else:
                metrics = {}
                model = result
            
            # Try to make predictions for validation
            predictions = None
            try:
                predictions = self.predict(X)
                if predictions is not None:
                    metrics["training_predictions_shape"] = predictions.shape
            except Exception as e:
                logger.warning(f"Could not generate training predictions: {e}")
            
            return TrainingResult(
                success=True,
                duration=0,  # Will be set by caller
                metrics=metrics,
                model=model,
                predictions=predictions
            )
            
        except Exception as e:
            logger.error(f"Algorithm training failed: {e}")
            return TrainingResult(
                success=False,
                duration=0,
                metrics={},
                error=str(e)
            )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained algorithm"""
        if not self.is_trained or self.algorithm_instance is None:
            raise ValueError(f"Algorithm {self.algorithm_id} must be trained before making predictions")
        
        try:
            if hasattr(self.algorithm_instance, "predict"):
                return self.algorithm_instance.predict(X)
            elif hasattr(self.algorithm_instance, "transform"):
                # For unsupervised algorithms
                return self.algorithm_instance.transform(X)
            elif self._training_results and self._training_results.model:
                # Try to predict with the model from training results
                model = self._training_results.model
                if hasattr(model, "predict"):
                    return model.predict(X)
            
            raise NotImplementedError(f"Don't know how to make predictions with {type(self.algorithm_instance)}")
            
        except Exception as e:
            logger.error(f"Prediction error for {self.algorithm_id}: {e}")
            raise
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get algorithm metadata"""
        return {
            **self.algorithm_info,
            "is_trained": self.is_trained,
            "instance_type": type(self.algorithm_instance).__name__ if self.algorithm_instance else None,
            "last_training_results": self._training_results.metrics if self._training_results else None
        }
    
    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get available hyperparameters for this algorithm"""
        return self.algorithm_info.get("hyperparameters", {})
    
    def validate_hyperparameters(self, **kwargs) -> Dict[str, Any]:
        """Validate and return cleaned hyperparameters"""
        hyperparams = self.get_hyperparameters()
        validated = {}
        
        for key, value in kwargs.items():
            if key in hyperparams:
                param_info = hyperparams[key]
                # Type conversion based on parameter info
                param_type = param_info.get("type", "string")
                try:
                    if param_type == "float":
                        validated[key] = float(value)
                    elif param_type == "int":
                        validated[key] = int(value)
                    elif param_type == "bool":
                        validated[key] = bool(value)
                    else:
                        validated[key] = str(value)
                except (ValueError, TypeError) as e:
                    logger.warning(f"Could not convert {key}={value} to {param_type}: {e}")
                    validated[key] = value
            else:
                logger.warning(f"Unknown hyperparameter: {key}")
                validated[key] = value
        
        return validated
