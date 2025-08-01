"""
Request models for the ML Playground API
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class TrainingRequest(BaseModel):
    """Request model for training an algorithm"""
    algorithm_id: str = Field(..., description="ID of the algorithm to train")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Algorithm hyperparameters")
    dataset_type: str = Field(default="synthetic", description="Type of dataset to use")
    dataset_params: Dict[str, Any] = Field(default_factory=dict, description="Dataset generation parameters")

class PredictionRequest(BaseModel):
    """Request model for making predictions"""
    X: List[List[float]] = Field(..., description="Input data for predictions")

class DatasetRequest(BaseModel):
    """Request model for dataset generation"""
    dataset_type: str = Field(..., description="Type of dataset to generate")
    params: Dict[str, Any] = Field(default_factory=dict, description="Dataset parameters")
    
class AlgorithmRefreshRequest(BaseModel):
    """Request model for refreshing algorithm discovery"""
    force_refresh: bool = Field(default=True, description="Force refresh of algorithm cache")
