"""
Training API Endpoints
Provides comprehensive training results with sklearn comparison and visualization data
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional
import logging

from models.training_results import TrainingResult
from services.training_service import training_service
from services.algorithm_service import algorithm_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/training", tags=["Training"])

@router.post("/{algorithm_id}", response_model=TrainingResult)
async def train_algorithm(
    algorithm_id: str,
    hyperparameters: Dict[str, Any],
    dataset: Optional[str] = Query(default=None, description="Dataset to use for training (auto-detected if not specified)"),
) -> TrainingResult:
    """
    Train an algorithm with comprehensive results including sklearn comparison
    
    This endpoint provides:
    - Your algorithm training results
    - Sklearn equivalent training results  
    - Performance comparison and analysis
    - Rich metrics (R², accuracy, etc.)
    - Visualization data for charts
    - Training history and convergence analysis
    - Auto-detection of best dataset for algorithm type
    """
    try:
        # Validate algorithm exists
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(
                status_code=404, 
                detail=f"Algorithm '{algorithm_id}' not found"
            )
        
        # Import here to avoid circular imports
        from services.training_service import DatasetService, AlgorithmTypeDetector
        
        # Validate dataset if provided, otherwise auto-detect
        valid_datasets = list(DatasetService.DATASETS.keys())
        if dataset is not None and dataset not in valid_datasets:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid dataset '{dataset}'. Valid options: {valid_datasets}"
            )
        elif dataset is None:
            dataset = AlgorithmTypeDetector.get_best_dataset(algorithm_id)
            logger.info(f"Auto-selected dataset '{dataset}' for algorithm '{algorithm_id}'")
        
        logger.info(f"Starting training for {algorithm_id} with dataset {dataset}")
        logger.info(f"Hyperparameters: {hyperparameters}")
        
        # Train with comparison
        result = await training_service.train_with_comparison(
            algorithm_id=algorithm_id,
            hyperparameters=hyperparameters,
            dataset_name=dataset
        )
        
        if result.success:
            logger.info(f"Training completed successfully for {algorithm_id}")
            logger.info(f"Your R² score: {result.your_implementation.metrics.r2_score}")
            logger.info(f"Sklearn R² score: {result.sklearn_implementation.metrics.r2_score}")
        else:
            logger.error(f"Training failed for {algorithm_id}: {result.error}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in training: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during training: {str(e)}"
        )
