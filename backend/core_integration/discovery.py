"""
Core Algorithm Discovery System
Automatically discovers and loads algorithms from the /core directory
"""
import os
import sys
import importlib.util
import inspect
from pathlib import Path
from typing import Dict, List, Type, Any, Optional
import logging

logger = logging.getLogger(__name__)

class CoreAlgorithmDiscovery:
    """Discovers algorithms from the core directory"""
    
    def __init__(self, core_path: str = None):
        if core_path is None:
            # First try ./core (for Docker container)
            container_core = Path("./core")
            if container_core.exists():
                self.core_path = container_core
            else:
                # Fallback to ../core directory relative to backend (for local development)
                backend_dir = Path(__file__).parent.parent
                self.core_path = backend_dir.parent / "core"
        else:
            self.core_path = Path(core_path)
        
        # Add core path to Python path for imports
        import sys
        core_path_str = str(self.core_path.absolute())
        if core_path_str not in sys.path:
            sys.path.insert(0, core_path_str)
        
        self._discovered_algorithms: Dict[str, Dict[str, Any]] = {}
        
    def discover_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """
        Discover all algorithms in the core directory
        Returns dict with algorithm_id -> algorithm_info
        """
        if not self.core_path.exists():
            logger.warning(f"Core directory not found: {self.core_path}")
            return {}
        
        self._discovered_algorithms = {}
        
        # Scan all Python files in core directory
        for py_file in self.core_path.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private files
                
            try:
                algorithm_info = self._analyze_algorithm_file(py_file)
                if algorithm_info:
                    algorithm_id = py_file.stem  # filename without extension
                    self._discovered_algorithms[algorithm_id] = {
                        **algorithm_info,
                        "id": algorithm_id,
                        "file_path": str(py_file),
                        "module_name": py_file.stem
                    }
                    logger.info(f"Discovered algorithm: {algorithm_id}")
            except Exception as e:
                logger.error(f"Error analyzing {py_file}: {e}")
                
        return self._discovered_algorithms
    
    def _analyze_algorithm_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a Python file to extract algorithm information"""
        try:
            # Load the module dynamically
            spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
            if spec is None or spec.loader is None:
                return None
                
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Look for algorithm classes or functions
            algorithm_info = self._extract_algorithm_metadata(module, file_path)
            return algorithm_info
            
        except Exception as e:
            logger.error(f"Failed to load module {file_path}: {e}")
            return None
    
    def _extract_algorithm_metadata(self, module, file_path: Path) -> Dict[str, Any]:
        """Extract metadata from a loaded module"""
        # Default metadata
        metadata = {
            "name": file_path.stem.replace("_", " ").title(),
            "type": "supervised",  # default
            "description": f"Implementation of {file_path.stem.replace('_', ' ')}",
            "hyperparameters": {}
        }
        
        # Look for metadata in module docstring
        if hasattr(module, "__doc__") and module.__doc__:
            metadata["description"] = module.__doc__.strip().split('\n')[0]
        
        # Look for algorithm classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if name.endswith("Algorithm") or name.endswith("Model"):
                if hasattr(obj, "get_metadata"):
                    # If class has get_metadata method, use it
                    try:
                        class_metadata = obj.get_metadata()
                        metadata.update(class_metadata)
                    except Exception as e:
                        logger.warning(f"Error getting metadata from {name}: {e}")
                
                # Look for hyperparameters in __init__ method
                if hasattr(obj, "__init__"):
                    sig = inspect.signature(obj.__init__)
                    hyperparams = {}
                    for param_name, param in sig.parameters.items():
                        if param_name == "self":
                            continue
                        hyperparams[param_name] = {
                            "type": "float" if param.annotation == float else "int" if param.annotation == int else "string",
                            "default": param.default if param.default != inspect.Parameter.empty else None
                        }
                    if hyperparams:
                        metadata["hyperparameters"] = hyperparams
        
        # Look for functions that might be algorithms
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if name.startswith("train_") or name.endswith("_algorithm"):
                # This might be a functional algorithm
                sig = inspect.signature(obj)
                # Extract hyperparameters from function signature
                hyperparams = {}
                for param_name, param in sig.parameters.items():
                    if param_name in ["X", "y", "data", "labels"]:
                        continue  # Skip data parameters
                    hyperparams[param_name] = {
                        "type": "float" if param.annotation == float else "int" if param.annotation == int else "string",
                        "default": param.default if param.default != inspect.Parameter.empty else None
                    }
                if hyperparams:
                    metadata["hyperparameters"] = hyperparams
        
        return metadata

    def get_algorithm_info(self, algorithm_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific algorithm"""
        return self._discovered_algorithms.get(algorithm_id)
    
    def get_algorithm_class(self, algorithm_id: str) -> Optional[Type]:
        """Get the actual algorithm class for instantiation"""
        try:
            # Import the module
            module_name = algorithm_id  # algorithm_id should match filename
            module = importlib.import_module(module_name)
            
            # Look for classes with fit and predict methods
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (hasattr(obj, 'fit') and hasattr(obj, 'predict') and 
                    not name.startswith('_')):
                    logger.info(f"Found algorithm class: {name} in {module_name}")
                    return obj
            
            logger.warning(f"No suitable algorithm class found in {module_name}")
            return None
            
        except Exception as e:
            logger.error(f"Error importing algorithm {algorithm_id}: {e}")
            return None
    
    def get_all_algorithms(self) -> Dict[str, Dict[str, Any]]:
        """Get all discovered algorithms"""
        return self._discovered_algorithms.copy()

# Global instance
core_discovery = CoreAlgorithmDiscovery()
