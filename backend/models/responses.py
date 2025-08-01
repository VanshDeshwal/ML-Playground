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
