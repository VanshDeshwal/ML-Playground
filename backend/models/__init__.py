# Models package for ML Playground - Cleaned Architecture
# Only includes models that are actually used

from .responses import (
    AlgorithmInfo,
    AlgorithmsListResponse,
    CodeSnippet,
    HealthResponse
)

from .training_results import TrainingResult

__all__ = [
    # Response models (actively used)
    'AlgorithmInfo',
    'AlgorithmsListResponse', 
    'CodeSnippet',
    'HealthResponse',
    # Training
    'TrainingResult'
]
