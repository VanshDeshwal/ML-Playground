"""
Training API Endpoints
Endpoints for training algorithms and getting results
"""
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
import numpy as np

from services.algorithm_service import algorithm_service
from dataset_manager import dataset_manager
from models.requests import TrainingRequest, PredictionRequest
from models.responses import TrainingResponse, PredictionResponse, StatusResponse

router = APIRouter(prefix="/training", tags=["training"])

@router.post("/train", response_model=TrainingResponse)
async def train_algorithm(request: TrainingRequest):
    """Train an algorithm with specified parameters"""
    try:
        # Validate algorithm exists
        if not await algorithm_service.validate_algorithm_exists(request.algorithm_id):
            raise HTTPException(status_code=404, detail=f"Algorithm {request.algorithm_id} not found")
        
        # Generate dataset
        try:
            if request.dataset_type == "synthetic":
                X, y = dataset_manager.generate_synthetic_data(**request.dataset_params)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported dataset type: {request.dataset_type}")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error generating dataset: {str(e)}")
        
        # Train the algorithm
        result = await algorithm_service.train_algorithm(
            request.algorithm_id,
            X,
            y,
            **request.hyperparameters
        )
        
        # Format response
        response = TrainingResponse(
            success=result.success,
            duration=result.duration,
            metrics=result.metrics,
            algorithm_id=request.algorithm_id,
            error=result.error
        )
        
        if result.predictions is not None:
            response.predictions_shape = list(result.predictions.shape)
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@router.post("/predict/{algorithm_id}", response_model=PredictionResponse)
async def make_predictions(algorithm_id: str, request: PredictionRequest):
    """Make predictions with a trained algorithm"""
    try:
        # Validate algorithm exists and is trained
        if algorithm_id not in algorithm_service.get_trained_algorithms():
            raise HTTPException(
                status_code=400, 
                detail=f"Algorithm {algorithm_id} is not trained. Train it first."
            )
        
        # Convert input data to numpy array
        X = np.array(request.X)
        
        # Make predictions
        predictions = await algorithm_service.predict(algorithm_id, X)
        
        return PredictionResponse(
            algorithm_id=algorithm_id,
            predictions=predictions.tolist(),
            shape=list(predictions.shape)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@router.get("/status", response_model=StatusResponse)
async def get_training_status():
    """Get status of all trained algorithms"""
    try:
        trained_algorithms = algorithm_service.get_trained_algorithms()
        
        status = {}
        for alg_id in trained_algorithms:
            adapter = await algorithm_service.get_algorithm_adapter(alg_id)
            if adapter:
                metadata = adapter.get_metadata()
                status[alg_id] = {
                    "is_trained": metadata["is_trained"],
                    "instance_type": metadata["instance_type"],
                    "last_training_results": metadata["last_training_results"]
                }
        
        return StatusResponse(
            total_trained=len(trained_algorithms),
            trained_algorithms=trained_algorithms,
            status=status
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting training status: {str(e)}")

@router.delete("/clear/{algorithm_id}")
async def clear_training(algorithm_id: str):
    """Clear training state for an algorithm"""
    try:
        if not await algorithm_service.validate_algorithm_exists(algorithm_id):
            raise HTTPException(status_code=404, detail=f"Algorithm {algorithm_id} not found")
        
        algorithm_service.clear_algorithm_cache(algorithm_id)
        
        return {"message": f"Training state cleared for {algorithm_id}"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing training: {str(e)}")

@router.delete("/clear-all")
async def clear_all_training():
    """Clear all training states"""
    try:
        algorithm_service.clear_algorithm_cache()
        return {"message": "All training states cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing all training: {str(e)}")
