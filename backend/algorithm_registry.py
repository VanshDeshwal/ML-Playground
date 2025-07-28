# Algorithm Registry - Auto-discovery and management of ML algorithms
import importlib
import inspect
from typing import Dict, List, Type, Any, Optional
from functools import lru_cache
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

class BaseAlgorithm(ABC):
    """Base class for all ML algorithms"""
    
    @classmethod
    @abstractmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata including id, name, type, description, hyperparameters"""
        pass
    
    @abstractmethod
    async def train(self, X, y=None, **kwargs) -> Dict[str, Any]:
        """Train the algorithm and return training results"""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions"""
        pass
    
    def get_sklearn_equivalent(self):
        """Get equivalent sklearn model for comparison"""
        return None
    
    def get_training_metadata(self) -> Dict[str, Any]:
        """Get additional training metadata (loss history, etc.)"""
        return {}

class AlgorithmRegistry:
    """Registry for managing and discovering ML algorithms"""
    
    def __init__(self):
        self._algorithms: Dict[str, Type[BaseAlgorithm]] = {}
        self._instances_cache: Dict[str, BaseAlgorithm] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    def register(self, algorithm_class: Type[BaseAlgorithm]) -> None:
        """Register an algorithm"""
        if not issubclass(algorithm_class, BaseAlgorithm):
            raise ValueError(f"Algorithm must inherit from BaseAlgorithm")
        
        # Get metadata from class method
        metadata = algorithm_class.get_metadata()
        algorithm_id = metadata["id"]
        
        self._algorithms[algorithm_id] = algorithm_class
        self._metadata_cache[algorithm_id] = metadata
        
        print(f"✅ Registered algorithm: {metadata['name']} ({algorithm_id})")
    
    def _generate_id(self, name: str) -> str:
        """Generate algorithm ID from name"""
        return name.lower().replace(" ", "_").replace("-", "_")
    
    @lru_cache(maxsize=None)
    def get_all_algorithms(self) -> List[Dict[str, Any]]:
        """Get all registered algorithms metadata"""
        return list(self._metadata_cache.values())
    
    def get_algorithm(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get specific algorithm metadata"""
        return self._metadata_cache.get(algorithm_id)
    
    async def create_instance(self, algorithm_id: str, hyperparameters: Dict[str, Any]) -> Optional[BaseAlgorithm]:
        """Create algorithm instance with hyperparameters"""
        if algorithm_id not in self._algorithms:
            return None
        
        # Run algorithm creation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        algorithm_class = self._algorithms[algorithm_id]
        
        def create_algo():
            return algorithm_class(**hyperparameters)
        
        try:
            instance = await loop.run_in_executor(self._executor, create_algo)
            return instance
        except Exception as e:
            print(f"❌ Failed to create {algorithm_id}: {e}")
            return None
    
    async def train_algorithm(self, algorithm_id: str, hyperparameters: Dict[str, Any], 
                            X, y=None) -> Dict[str, Any]:
        """Train algorithm asynchronously"""
        # Create instance
        instance = await self.create_instance(algorithm_id, hyperparameters)
        if not instance:
            raise ValueError(f"Failed to create algorithm: {algorithm_id}")
        
        # Use the new async train method
        try:
            result = await instance.train(X, y)
            return result
            
        except Exception as e:
            raise RuntimeError(f"Training failed for {algorithm_id}: {e}")
    
    def auto_discover(self, models_package="models"):
        """Auto-discover algorithms in models package"""
        try:
            import models
            import pkgutil
            
            for importer, modname, ispkg in pkgutil.iter_modules(models.__path__, models.__name__ + "."):
                if not ispkg:
                    try:
                        module = importlib.import_module(modname)
                        
                        # Look for classes that inherit from BaseAlgorithm
                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if (issubclass(obj, BaseAlgorithm) and 
                                obj != BaseAlgorithm and
                                getattr(obj, '__module__', None) == modname):
                                self.register(obj)
                                
                    except Exception as e:
                        print(f"⚠️ Failed to import {modname}: {e}")
                        
        except Exception as e:
            print(f"⚠️ Auto-discovery failed: {e}")

# Global registry instance
algorithm_registry = AlgorithmRegistry()
