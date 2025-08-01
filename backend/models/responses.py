"""
Response models for the ML Playground API
"""
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class AlgorithmInfo(BaseModel):
    """Information about an algorithm"""
    id: str = Field(..., description="Unique algorithm identifier")
    name: str = Field(..., description="Human-readable algorithm name")
    type: str = Field(..., description="Algorithm type (supervised, unsupervised, etc.)")
    description: str = Field(..., description="Algorithm description")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict, description="Available hyperparameters")
    file_path: Optional[str] = Field(None, description="Path to the algorithm file")
    is_trained: bool = Field(default=False, description="Whether the algorithm is currently trained")

class AlgorithmsListResponse(BaseModel):
    """Response for listing algorithms"""
    algorithms: List[AlgorithmInfo] = Field(..., description="List of available algorithms")
    total: int = Field(..., description="Total number of algorithms")

class TrainingResponse(BaseModel):
    """Response for algorithm training"""
    success: bool = Field(..., description="Whether training was successful")
    duration: float = Field(..., description="Training duration in seconds")
    metrics: Dict[str, Any] = Field(..., description="Training metrics and results")
    algorithm_id: str = Field(..., description="ID of the trained algorithm")
    error: Optional[str] = Field(None, description="Error message if training failed")
    predictions_shape: Optional[List[int]] = Field(None, description="Shape of training predictions")

class PredictionResponse(BaseModel):
    """Response for predictions"""
    algorithm_id: str = Field(..., description="ID of the algorithm used")
    predictions: List[Any] = Field(..., description="Prediction results")
    shape: List[int] = Field(..., description="Shape of predictions array")

class CodeSnippet(BaseModel):
    """Code snippet information"""
    filename: str = Field(..., description="Source filename")
    language: str = Field(default="python", description="Programming language")
    description: str = Field(..., description="Code description")
    code: str = Field(..., description="Source code content")

class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Health status")
    algorithms_discovered: Optional[int] = Field(None, description="Number of algorithms discovered")
    core_directory_exists: Optional[bool] = Field(None, description="Whether core directory exists")
    error: Optional[str] = Field(None, description="Error message if unhealthy")

class StatusResponse(BaseModel):
    """Training status response"""
    total_trained: int = Field(..., description="Number of trained algorithms")
    trained_algorithms: List[str] = Field(..., description="List of trained algorithm IDs")
    status: Dict[str, Any] = Field(..., description="Detailed status of each algorithm")
