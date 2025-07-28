# Core configuration and settings for ML Playground Backend
import os
from typing import Dict, Any
from functools import lru_cache

class Settings:
    """Application settings and configuration"""
    
    # API Settings
    API_TITLE = "ML Playground API"
    API_VERSION = "2.0.0"
    API_DESCRIPTION = "High-performance ML algorithm playground with modular architecture"
    
    # Performance Settings
    CACHE_TTL_SHORT = 300      # 5 minutes for quick operations
    CACHE_TTL_MEDIUM = 1800    # 30 minutes for datasets
    CACHE_TTL_LONG = 3600      # 1 hour for algorithm metadata
    MAX_CACHE_SIZE = 100       # Maximum cached items
    
    # Data Settings
    MAX_DATASET_SIZE = 10000   # Maximum samples per dataset
    MAX_FEATURES = 50          # Maximum features
    DEFAULT_TEST_SIZE = 0.2    # Default train/test split
    
    # Training Settings
    MAX_ITERATIONS = 5000      # Maximum iterations for algorithms
    CONVERGENCE_THRESHOLD = 1e-6  # Convergence criteria
    
    # Security & CORS
    CORS_ORIGINS = [
        "http://localhost:3000",      # Local development
        "http://localhost:8080",      # Alternative local port
        "http://127.0.0.1:3000",      # Local IP
        "http://127.0.0.1:8080",      # Alternative local IP
        "https://*.github.io",        # GitHub Pages (all subdomains)
        "https://vansdeshwal.github.io",  # Your specific GitHub Pages URL
    ]
    CORS_CREDENTIALS = True
    CORS_METHODS = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    CORS_HEADERS = ["Content-Type", "Authorization", "X-Requested-With"]

@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()

# Algorithm type definitions
ALGORITHM_TYPES = {
    "regression": {
        "description": "Predict continuous numerical values",
        "metrics": ["mse", "r2", "mae"],
        "default_metric": "r2"
    },
    "classification": {
        "description": "Predict discrete categories or classes", 
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "default_metric": "accuracy"
    },
    "clustering": {
        "description": "Group similar data points together",
        "metrics": ["silhouette", "inertia", "calinski_harabasz"],
        "default_metric": "silhouette"
    }
}

# Dataset configurations
BUILTIN_DATASETS = {
    "diabetes": {
        "name": "Diabetes Dataset",
        "type": "regression",
        "description": "Diabetes progression prediction",
        "size": "small",
        "features": 10,
        "samples": 442
    },
    "wine": {
        "name": "Wine Quality Dataset", 
        "type": "classification",
        "description": "Wine quality classification",
        "size": "small",
        "features": 13,
        "samples": 178
    },
    "iris": {
        "name": "Iris Species Dataset",
        "type": "classification", 
        "description": "Iris species classification",
        "size": "small",
        "features": 4,
        "samples": 150
    },
    "breast_cancer": {
        "name": "Breast Cancer Dataset",
        "type": "classification",
        "description": "Breast cancer diagnosis",
        "size": "medium",
        "features": 30,
        "samples": 569
    },
    "blobs": {
        "name": "Clustering Blobs",
        "type": "clustering", 
        "description": "Synthetic clustering dataset",
        "size": "configurable",
        "features": "configurable",
        "samples": "configurable"
    }
}
