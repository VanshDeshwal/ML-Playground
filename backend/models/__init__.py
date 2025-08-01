# Models package for ML Playground - New Architecture
# Pydantic models for API requests and responses

from .requests import (
    TrainingRequest,
    PredictionRequest,
    DatasetRequest,
    AlgorithmRefreshRequest
)

from .responses import (
    AlgorithmInfo,
    AlgorithmsListResponse,
    TrainingResponse,
    PredictionResponse,
    CodeSnippet,
    HealthResponse,
    StatusResponse
)

__all__ = [
    # Request models
    'TrainingRequest',
    'PredictionRequest', 
    'DatasetRequest',
    'AlgorithmRefreshRequest',
    # Response models
    'AlgorithmInfo',
    'AlgorithmsListResponse',
    'TrainingResponse',
    'PredictionResponse',
    'CodeSnippet',
    'HealthResponse',
    'StatusResponse'
]
