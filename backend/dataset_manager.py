# Dataset management with caching and optimization
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, Union
from sklearn.datasets import (
    make_regression, make_classification, make_blobs,
    load_diabetes, load_wine, load_iris, load_breast_cancer
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import get_settings, BUILTIN_DATASETS

class DatasetManager:
    """High-performance dataset management with caching"""
    
    def __init__(self):
        self.settings = get_settings()
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Dataset loaders mapping
        self._loaders = {
            "diabetes": load_diabetes,
            "wine": load_wine,
            "iris": load_iris,
            "breast_cancer": load_breast_cancer,
            "blobs": self._make_blobs_wrapper
        }
    
    def _make_blobs_wrapper(self):
        """Wrapper for make_blobs to match sklearn dataset interface"""
        class BlobsDataset:
            def __init__(self, n_samples=300, centers=4, n_features=2, cluster_std=1.0, random_state=42):
                self.data, self.target = make_blobs(
                    n_samples=n_samples,
                    centers=centers, 
                    n_features=n_features,
                    cluster_std=cluster_std,
                    random_state=random_state
                )
                self.feature_names = [f"feature_{i}" for i in range(n_features)]
                self.target_names = [f"cluster_{i}" for i in range(centers)]
                self.DESCR = f"Synthetic clustering dataset with {centers} clusters"
        
        return BlobsDataset()
    
    async def get_builtin_dataset(self, dataset_name: str, **kwargs) -> Dict[str, Any]:
        """Get a built-in dataset"""
        if dataset_name not in BUILTIN_DATASETS:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        dataset_info = BUILTIN_DATASETS[dataset_name]
        
        # Load dataset in thread pool
        loop = asyncio.get_event_loop()
        
        def load_data():
            if dataset_name == "blobs":
                # Special handling for configurable blobs
                loader = self._loaders[dataset_name]()
                return loader
            else:
                loader = self._loaders[dataset_name]
                return loader()
        
        try:
            dataset = await loop.run_in_executor(self.executor, load_data)
            
            return {
                "X": dataset.data,
                "y": dataset.target,
                "feature_names": getattr(dataset, 'feature_names', None),
                "target_names": getattr(dataset, 'target_names', None),
                "description": getattr(dataset, 'DESCR', dataset_info['description']),
                "type": dataset_info["type"],
                "info": dataset_info
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset '{dataset_name}': {e}")
    
    async def generate_synthetic_dataset(
        self, 
        algorithm_type: str,
        n_samples: int = 500,
        n_features: int = 10,
        noise: float = 0.1,
        random_state: int = 42,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate synthetic dataset based on algorithm type"""
        
        # Validate parameters
        n_samples = min(n_samples, self.settings.MAX_DATASET_SIZE)
        n_features = min(n_features, self.settings.MAX_FEATURES)
        
        loop = asyncio.get_event_loop()
        
        def generate_data():
            if algorithm_type == "regression":
                X, y = make_regression(
                    n_samples=n_samples,
                    n_features=n_features,
                    noise=noise,
                    random_state=random_state
                )
                
            elif algorithm_type == "classification":
                n_classes = kwargs.get("n_classes", 2)
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=n_features,
                    n_classes=n_classes,
                    n_redundant=0,
                    n_informative=min(n_features, n_classes * 2),
                    random_state=random_state
                )
                
            elif algorithm_type == "clustering":
                n_centers = kwargs.get("n_centers", 3)
                X, y = make_blobs(
                    n_samples=n_samples,
                    centers=n_centers,
                    n_features=n_features,
                    cluster_std=noise * 10,  # Scale noise for clustering
                    random_state=random_state
                )
                
            else:
                raise ValueError(f"Unknown algorithm type: {algorithm_type}")
            
            return X, y
        
        try:
            X, y = await loop.run_in_executor(self.executor, generate_data)
            
            return {
                "X": X,
                "y": y,
                "feature_names": [f"feature_{i}" for i in range(n_features)],
                "description": f"Synthetic {algorithm_type} dataset",
                "type": algorithm_type,
                "generation_params": {
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "noise": noise,
                    "random_state": random_state,
                    **kwargs
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic dataset: {e}")
    
    async def process_uploaded_dataset(self, file_content: bytes) -> Dict[str, Any]:
        """Process uploaded CSV dataset"""
        loop = asyncio.get_event_loop()
        
        def process_data():
            try:
                # Try to read as CSV
                import io
                df = pd.read_csv(io.BytesIO(file_content))
                
                # Basic validation
                if df.shape[0] > self.settings.MAX_DATASET_SIZE:
                    df = df.sample(n=self.settings.MAX_DATASET_SIZE, random_state=42)
                
                if df.shape[1] > self.settings.MAX_FEATURES:
                    df = df.iloc[:, :self.settings.MAX_FEATURES]
                
                # Assume last column is target (can be made configurable)
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                return {
                    "X": X,
                    "y": y,
                    "feature_names": df.columns[:-1].tolist(),
                    "target_name": df.columns[-1],
                    "description": f"Uploaded dataset ({df.shape[0]} samples, {df.shape[1]-1} features)",
                    "type": "uploaded",
                    "original_shape": df.shape
                }
                
            except Exception as e:
                raise ValueError(f"Failed to process uploaded file: {e}")
        
        return await loop.run_in_executor(self.executor, process_data)
    
    async def prepare_training_data(
        self,
        dataset: Dict[str, Any],
        test_size: float = None,
        scale_features: bool = False,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training with optional scaling and splitting"""
        
        test_size = test_size or self.settings.DEFAULT_TEST_SIZE
        
        loop = asyncio.get_event_loop()
        
        def prepare_data():
            X = dataset["X"]
            y = dataset["y"]
            
            # Split data
            if len(X) > 1:  # Only split if we have more than 1 sample
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
            else:
                X_train = X_test = X
                y_train = y_test = y
            
            # Scale features if requested
            if scale_features:
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)
            
            return X_train, X_test, y_train, y_test
        
        return await loop.run_in_executor(self.executor, prepare_data)
    
    def get_available_datasets(self) -> Dict[str, Any]:
        """Get information about available datasets"""
        return {
            "builtin_datasets": [
                {
                    "id": dataset_id,
                    **dataset_info
                }
                for dataset_id, dataset_info in BUILTIN_DATASETS.items()
            ],
            "synthetic_options": {
                "regression": {
                    "parameters": ["n_samples", "n_features", "noise", "random_state"]
                },
                "classification": {
                    "parameters": ["n_samples", "n_features", "n_classes", "random_state"]
                },
                "clustering": {
                    "parameters": ["n_samples", "n_features", "n_centers", "noise", "random_state"]
                }
            },
            "upload_formats": ["CSV"],
            "limits": {
                "max_samples": self.settings.MAX_DATASET_SIZE,
                "max_features": self.settings.MAX_FEATURES
            }
        }

# Global dataset manager
dataset_manager = DatasetManager()
