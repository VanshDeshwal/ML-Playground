"""
Enhanced Training API Endpoints
Provides rich training results with sklearn comparison and visualization data
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from models.enhanced_results import EnhancedTrainingResult
from services.enhanced_training_service import enhanced_training_service
from services.algorithm_service import algorithm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/enhanced-training", tags=["Enhanced Training"])

@router.post("/{algorithm_id}", response_model=EnhancedTrainingResult)
async def train_algorithm_enhanced(
    algorithm_id: str,
    hyperparameters: Dict[str, Any],
    dataset: str = Query(default="diabetes", description="Dataset to use for training"),
) -> EnhancedTrainingResult:
    """
    Train an algorithm with comprehensive results including sklearn comparison
    
    This endpoint provides:
    - Your algorithm training results
    - Sklearn equivalent training results  
    - Performance comparison and analysis
    - Rich metrics (R², accuracy, etc.)
    - Visualization data for charts
    - Training history and convergence analysis
    """
    try:
        # Validate algorithm exists
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Algorithm '{algorithm_id}' not found"
            )
        
        # Validate dataset
        valid_datasets = ["diabetes", "iris", "wine", "breast_cancer"]
        if dataset not in valid_datasets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset '{dataset}'. Valid options: {valid_datasets}"
            )
        
        logger.info(f"Starting enhanced training for {algorithm_id} with dataset {dataset}")
        logger.info(f"Hyperparameters: {hyperparameters}")
        
        # Train with comparison
        result = await enhanced_training_service.train_with_comparison(
            algorithm_id=algorithm_id,
            hyperparameters=hyperparameters,
            dataset_name=dataset
        )
        
        if result.success:
            logger.info(f"Enhanced training completed successfully for {algorithm_id}")
            logger.info(f"Your R² score: {result.your_implementation.metrics.r2_score}")
            logger.info(f"Sklearn R² score: {result.sklearn_implementation.metrics.r2_score}")
        else:
            logger.error(f"Enhanced training failed for {algorithm_id}: {result.error}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during training: {str(e)}"
        )

@router.get("/datasets")
async def get_available_datasets():
    """Get list of available datasets for training"""
    return {
        "datasets": [
            {
                "id": "diabetes",
                "name": "Diabetes Dataset",
                "type": "regression",
                "description": "Diabetes progression prediction",
                "samples": 442,
                "features": 10
            },
            {
                "id": "iris", 
                "name": "Iris Species Dataset",
                "type": "classification",
                "description": "Iris species classification",
                "samples": 150,
                "features": 4
            },
            {
                "id": "wine",
                "name": "Wine Quality Dataset", 
                "type": "classification",
                "description": "Wine quality classification",
                "samples": 178,
                "features": 13
            },
            {
                "id": "breast_cancer",
                "name": "Breast Cancer Dataset",
                "type": "classification", 
                "description": "Breast cancer diagnosis",
                "samples": 569,
                "features": 30
            }
        ]
    }

@router.get("/metrics/{algorithm_type}")
async def get_metrics_info(algorithm_type: str):
    """Get information about metrics used for different algorithm types"""
    
    metrics_info = {
        "regression": {
            "primary_metric": "r2_score",
            "metrics": {
                "r2_score": {
                    "name": "R² Score",
                    "description": "Coefficient of determination (1.0 = perfect fit)",
                    "range": "(-∞, 1.0]",
                    "higher_is_better": True
                },
                "mse": {
                    "name": "Mean Squared Error", 
                    "description": "Average squared difference between actual and predicted",
                    "range": "[0, ∞)",
                    "higher_is_better": False
                },
                "mae": {
                    "name": "Mean Absolute Error",
                    "description": "Average absolute difference between actual and predicted", 
                    "range": "[0, ∞)",
                    "higher_is_better": False
                },
                "rmse": {
                    "name": "Root Mean Squared Error",
                    "description": "Square root of MSE (same units as target)",
                    "range": "[0, ∞)", 
                    "higher_is_better": False
                }
            }
        },
        "classification": {
            "primary_metric": "accuracy",
            "metrics": {
                "accuracy": {
                    "name": "Accuracy",
                    "description": "Fraction of correct predictions",
                    "range": "[0, 1.0]",
                    "higher_is_better": True
                },
                "precision": {
                    "name": "Precision",
                    "description": "True positives / (True positives + False positives)",
                    "range": "[0, 1.0]",
                    "higher_is_better": True
                },
                "recall": {
                    "name": "Recall (Sensitivity)",
                    "description": "True positives / (True positives + False negatives)",
                    "range": "[0, 1.0]", 
                    "higher_is_better": True
                },
                "f1_score": {
                    "name": "F1 Score",
                    "description": "Harmonic mean of precision and recall",
                    "range": "[0, 1.0]",
                    "higher_is_better": True
                }
            }
        },
        "clustering": {
            "primary_metric": "silhouette_score", 
            "metrics": {
                "silhouette_score": {
                    "name": "Silhouette Score",
                    "description": "Measure of cluster cohesion and separation",
                    "range": "[-1, 1]",
                    "higher_is_better": True
                },
                "inertia": {
                    "name": "Inertia (WCSS)",
                    "description": "Within-cluster sum of squared distances",
                    "range": "[0, ∞)",
                    "higher_is_better": False
                },
                "calinski_harabasz_score": {
                    "name": "Calinski-Harabasz Index",
                    "description": "Ratio of between-cluster to within-cluster dispersion", 
                    "range": "[0, ∞)",
                    "higher_is_better": True
                }
            }
        }
    }
    
    if algorithm_type not in metrics_info:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown algorithm type: {algorithm_type}. Valid types: {list(metrics_info.keys())}"
        )
    
    return metrics_info[algorithm_type]
