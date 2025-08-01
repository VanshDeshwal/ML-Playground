"""
Core Algorithm Loader
Dynamically loads and instantiates algorithms from the core directory
"""
import importlib.util
import inspect
from pathlib import Path
from typing import Any, Dict, Optional, Type, Callable
import logging

from .discovery import core_discovery

logger = logging.getLogger(__name__)

class CoreAlgorithmLoader:
    """Loads and instantiates algorithms from core directory"""
    
    def __init__(self):
        self._loaded_modules: Dict[str, Any] = {}
        self._algorithm_classes: Dict[str, Type] = {}
        self._algorithm_functions: Dict[str, Callable] = {}
    
    def load_algorithm(self, algorithm_id: str) -> Optional[Any]:
        """
        Load and return an algorithm instance/function
        """
        try:
            # First discover if not already done
            if algorithm_id not in core_discovery.get_all_algorithms():
                core_discovery.discover_algorithms()
            
            algorithm_info = core_discovery.get_algorithm_info(algorithm_id)
            if not algorithm_info:
                logger.error(f"Algorithm {algorithm_id} not found")
                return None
            
            # Load the module if not already loaded
            if algorithm_id not in self._loaded_modules:
                module = self._load_module(algorithm_info["file_path"])
                if module is None:
                    return None
                self._loaded_modules[algorithm_id] = module
            
            module = self._loaded_modules[algorithm_id]
            
            # Find the main algorithm class or function
            algorithm_callable = self._find_algorithm_callable(module, algorithm_id)
            return algorithm_callable
            
        except Exception as e:
            logger.error(f"Error loading algorithm {algorithm_id}: {e}")
            return None
    
    def _load_module(self, file_path: str):
        """Load a Python module from file path"""
        try:
            module_name = Path(file_path).stem
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
            
        except Exception as e:
            logger.error(f"Failed to load module from {file_path}: {e}")
            return None
    
    def _find_algorithm_callable(self, module, algorithm_id: str):
        """Find the main algorithm class or function in the module"""
        
        # Strategy 1: Look for classes ending with "Algorithm" or "Model"
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Algorithm") or name.endswith("Model"):
                logger.info(f"Found algorithm class: {name}")
                return obj
        
        # Strategy 2: Look for class with same name as module (capitalized)
        class_name = algorithm_id.replace("_", "").capitalize()
        if hasattr(module, class_name):
            obj = getattr(module, class_name)
            if inspect.isclass(obj):
                logger.info(f"Found algorithm class by name: {class_name}")
                return obj
        
        # Strategy 3: Look for functions starting with "train_" or ending with "_algorithm"
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("train_") or name.endswith("_algorithm"):
                logger.info(f"Found algorithm function: {name}")
                return obj
        
        # Strategy 4: Look for main training function
        common_names = ["train", "fit", "run", "execute", algorithm_id]
        for name in common_names:
            if hasattr(module, name):
                obj = getattr(module, name)
                if callable(obj):
                    logger.info(f"Found algorithm function: {name}")
                    return obj
        
        # Strategy 5: Return the first callable that's not private
        for name, obj in inspect.getmembers(module):
            if not name.startswith("_") and callable(obj) and not inspect.ismodule(obj):
                logger.info(f"Found potential algorithm callable: {name}")
                return obj
        
        logger.warning(f"No suitable algorithm callable found in module for {algorithm_id}")
        return None
    
    def create_instance(self, algorithm_id: str, **kwargs) -> Optional[Any]:
        """
        Create an instance of the algorithm with given hyperparameters
        """
        algorithm_callable = self.load_algorithm(algorithm_id)
        if algorithm_callable is None:
            return None
        
        try:
            if inspect.isclass(algorithm_callable):
                # It's a class, instantiate it
                return algorithm_callable(**kwargs)
            else:
                # It's a function, return a wrapper
                return AlgorithmFunctionWrapper(algorithm_callable, **kwargs)
        except Exception as e:
            logger.error(f"Error creating instance of {algorithm_id}: {e}")
            return None
    
    def get_algorithm_signature(self, algorithm_id: str) -> Optional[inspect.Signature]:
        """Get the signature of the algorithm for parameter validation"""
        algorithm_callable = self.load_algorithm(algorithm_id)
        if algorithm_callable is None:
            return None
        
        try:
            if inspect.isclass(algorithm_callable):
                return inspect.signature(algorithm_callable.__init__)
            else:
                return inspect.signature(algorithm_callable)
        except Exception as e:
            logger.error(f"Error getting signature for {algorithm_id}: {e}")
            return None

class AlgorithmFunctionWrapper:
    """Wrapper for function-based algorithms to provide a consistent interface"""
    
    def __init__(self, func: Callable, **default_kwargs):
        self.func = func
        self.default_kwargs = default_kwargs
        self.is_trained = False
        self.model = None
        self.results = {}
    
    def train(self, X, y=None, **kwargs):
        """Train the algorithm"""
        # Merge default kwargs with training kwargs
        final_kwargs = {**self.default_kwargs, **kwargs}
        
        try:
            # Call the function
            result = self.func(X, y, **final_kwargs)
            self.results = result if isinstance(result, dict) else {"model": result}
            self.model = self.results.get("model", result)
            self.is_trained = True
            return self.results
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Algorithm must be trained before making predictions")
        
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        else:
            raise NotImplementedError("Prediction not available for this algorithm")

# Global instance
core_loader = CoreAlgorithmLoader()
