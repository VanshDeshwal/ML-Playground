"""
Dataset Manager for ML Playground
Generates synthetic datasets for training algorithms
"""
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.datasets import make_regression, make_classification, make_blobs
import logging

logger = logging.getLogger(__name__)

class DatasetManager:
    """Manages dataset generation for training algorithms"""
    
    def __init__(self):
        self.default_params = {
            "regression": {
                "n_samples": 500,
                "n_features": 10,
                "noise": 0.1,
                "random_state": 42
            },
            "classification": {
                "n_samples": 500,
                "n_features": 10,
                "n_classes": 2,
                "n_informative": 8,
                "n_redundant": 2,
                "random_state": 42
            },
            "clustering": {
                "n_samples": 300,
                "centers": 3,
                "n_features": 2,
                "cluster_std": 1.0,
                "random_state": 42
            }
        }
    
    def generate_synthetic_data(self, 
                              dataset_type: str = "regression",
                              **params) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic dataset for training
        
        Args:
            dataset_type: Type of dataset ('regression', 'classification', 'clustering')
            **params: Dataset generation parameters
        
        Returns:
            Tuple of (X, y) arrays
        """
        try:
            # Merge with defaults
            default_params = self.default_params.get(dataset_type, {})
            final_params = {**default_params, **params}
            
            if dataset_type == "regression":
                return make_regression(**final_params)
            elif dataset_type == "classification":
                return make_classification(**final_params)
            elif dataset_type == "clustering":
                return make_blobs(**final_params)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
                
        except Exception as e:
            logger.error(f"Error generating {dataset_type} dataset: {e}")
            raise
    
    def get_dataset_info(self, dataset_type: str) -> Dict[str, Any]:
        """Get information about a dataset type"""
        return {
            "type": dataset_type,
            "default_params": self.default_params.get(dataset_type, {}),
            "supported": dataset_type in self.default_params
        }
    
    def get_supported_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get all supported dataset types"""
        return {
            dataset_type: self.get_dataset_info(dataset_type)
            for dataset_type in self.default_params.keys()
        }

# Global instance - this is what gets imported
dataset_manager = DatasetManager()
