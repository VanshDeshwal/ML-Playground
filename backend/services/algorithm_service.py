"""
Algorithm Service - Business logic for algorithm management
"""
from typing import Dict, List, Any, Optional
import logging
import numpy as np

from core_integration.discovery import core_discovery
from adapters.algorithm_adapter import UniversalAlgorithmAdapter, TrainingResult

logger = logging.getLogger(__name__)

class AlgorithmService:
    """Service for managing algorithms and training operations"""
    
    def __init__(self):
        self._algorithm_adapters: Dict[str, UniversalAlgorithmAdapter] = {}
        self._discovery_cache = None
        self._cache_timestamp = None
    
    async def discover_algorithms(self, force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
        """Discover all available algorithms"""
        try:
            # Refresh discovery cache if needed
            if force_refresh or self._discovery_cache is None:
                self._discovery_cache = core_discovery.discover_algorithms()
                logger.info(f"Discovered {len(self._discovery_cache)} algorithms")
            
            return self._discovery_cache
        except Exception as e:
            logger.error(f"Error discovering algorithms: {e}")
            return {}
    
    async def get_algorithm_info(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific algorithm"""
        algorithms = await self.discover_algorithms()
        return algorithms.get(algorithm_id)
    
    async def get_all_algorithms(self) -> List[Dict[str, Any]]:
        """Get list of all available algorithms"""
        algorithms = await self.discover_algorithms()
        return list(algorithms.values())
    
    async def get_algorithm_adapter(self, algorithm_id: str) -> Optional[UniversalAlgorithmAdapter]:
        """Get or create an algorithm adapter"""
        try:
            # Return cached adapter if available
            if algorithm_id in self._algorithm_adapters:
                return self._algorithm_adapters[algorithm_id]
            
            # Get algorithm info
            algorithm_info = await self.get_algorithm_info(algorithm_id)
            if not algorithm_info:
                logger.error(f"Algorithm {algorithm_id} not found")
                return None
            
            # Create new adapter
            adapter = UniversalAlgorithmAdapter(algorithm_id, algorithm_info)
            self._algorithm_adapters[algorithm_id] = adapter
            
            return adapter
            
        except Exception as e:
            logger.error(f"Error getting adapter for {algorithm_id}: {e}")
            return None
    
    async def train_algorithm(
        self, 
        algorithm_id: str, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        **hyperparameters
    ) -> TrainingResult:
        """Train an algorithm with given data and hyperparameters"""
        
        try:
            # Get algorithm adapter
            adapter = await self.get_algorithm_adapter(algorithm_id)
            if adapter is None:
                return TrainingResult(
                    success=False,
                    duration=0,
                    metrics={},
                    error=f"Algorithm {algorithm_id} not found"
                )
            
            # Validate hyperparameters
            validated_params = adapter.validate_hyperparameters(**hyperparameters)
            
            # Train the algorithm
            result = await adapter.train(X, y, **validated_params)
            
            logger.info(f"Training completed for {algorithm_id}: success={result.success}")
            return result
            
        except Exception as e:
            logger.error(f"Training error for {algorithm_id}: {e}")
            return TrainingResult(
                success=False,
                duration=0,
                metrics={},
                error=str(e)
            )
    
    async def predict(self, algorithm_id: str, X: np.ndarray) -> Optional[np.ndarray]:
        """Make predictions with a trained algorithm"""
        try:
            adapter = self._algorithm_adapters.get(algorithm_id)
            if adapter is None:
                raise ValueError(f"Algorithm {algorithm_id} not trained yet")
            
            return adapter.predict(X)
            
        except Exception as e:
            logger.error(f"Prediction error for {algorithm_id}: {e}")
            raise
    
    def get_trained_algorithms(self) -> List[str]:
        """Get list of currently trained algorithms"""
        return [
            alg_id for alg_id, adapter in self._algorithm_adapters.items()
            if adapter.is_trained
        ]
    
    def clear_algorithm_cache(self, algorithm_id: Optional[str] = None):
        """Clear algorithm adapter cache"""
        if algorithm_id:
            self._algorithm_adapters.pop(algorithm_id, None)
        else:
            self._algorithm_adapters.clear()
        logger.info(f"Cleared algorithm cache for {algorithm_id or 'all algorithms'}")
    
    async def validate_algorithm_exists(self, algorithm_id: str) -> bool:
        """Check if an algorithm exists"""
        algorithms = await self.discover_algorithms()
        return algorithm_id in algorithms

# Global service instance
algorithm_service = AlgorithmService()
