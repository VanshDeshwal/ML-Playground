# High-performance caching system for ML Playground Backend
import asyncio
import time
import hashlib
import json
from typing import Any, Dict, Optional, Callable, Union
from functools import wraps
import numpy as np
import pandas as pd
from config import get_settings

class CacheManager:
    """High-performance caching manager with TTL and size limits"""
    
    def __init__(self):
        self.settings = get_settings()
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._creation_times: Dict[str, float] = {}
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        # Convert numpy arrays and pandas DataFrames to strings for hashing
        processed_args = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                processed_args.append(f"array_{arg.shape}_{hash(arg.tobytes())}")
            elif isinstance(arg, pd.DataFrame):
                processed_args.append(f"df_{arg.shape}_{hash(pd.util.hash_pandas_object(arg).values.tobytes())}")
            else:
                processed_args.append(str(arg))
        
        processed_kwargs = {k: str(v) for k, v in kwargs.items()}
        
        key_string = json.dumps({
            "args": processed_args,
            "kwargs": processed_kwargs
        }, sort_keys=True)
        
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _is_expired(self, key: str, ttl: int) -> bool:
        """Check if cache entry is expired"""
        if key not in self._creation_times:
            return True
        return time.time() - self._creation_times[key] > ttl
    
    def _cleanup_expired(self, ttl: int):
        """Remove expired entries"""
        current_time = time.time()
        expired_keys = [
            key for key, creation_time in self._creation_times.items()
            if current_time - creation_time > ttl
        ]
        
        for key in expired_keys:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._creation_times.pop(key, None)
    
    def _enforce_size_limit(self):
        """Enforce cache size limit using LRU"""
        if len(self._cache) <= self.settings.MAX_CACHE_SIZE:
            return
        
        # Sort by access time and remove oldest entries
        sorted_keys = sorted(self._access_times.items(), key=lambda x: x[1])
        keys_to_remove = [key for key, _ in sorted_keys[:len(self._cache) - self.settings.MAX_CACHE_SIZE]]
        
        for key in keys_to_remove:
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._creation_times.pop(key, None)
    
    def get(self, key: str, ttl: int = None) -> Optional[Any]:
        """Get cached value"""
        if ttl and self._is_expired(key, ttl):
            self._cache.pop(key, None)
            self._access_times.pop(key, None)
            self._creation_times.pop(key, None)
            return None
        
        if key in self._cache:
            self._access_times[key] = time.time()
            return self._cache[key]["value"]
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set cached value"""
        current_time = time.time()
        
        self._cache[key] = {"value": value}
        self._access_times[key] = current_time
        self._creation_times[key] = current_time
        
        # Cleanup and enforce limits
        if ttl:
            self._cleanup_expired(ttl)
        self._enforce_size_limit()
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries"""
        if pattern is None:
            # Clear all
            self._cache.clear()
            self._access_times.clear()
            self._creation_times.clear()
        else:
            # Clear matching pattern
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._access_times.pop(key, None)
                self._creation_times.pop(key, None)
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "total_entries": len(self._cache),
            "max_size": self.settings.MAX_CACHE_SIZE,
            "memory_usage_mb": sum(
                len(str(entry)) for entry in self._cache.values()
            ) / (1024 * 1024),
            "oldest_entry": min(self._creation_times.values()) if self._creation_times else None,
            "newest_entry": max(self._creation_times.values()) if self._creation_times else None
        }

# Global cache manager
cache_manager = CacheManager()

def cached(ttl: int = None, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}_{func.__name__}_{cache_manager._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, ttl)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = f"{key_prefix}_{func.__name__}_{cache_manager._generate_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key, ttl)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, ttl)
            return result
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Specific caching decorators for different use cases
def cache_dataset(ttl: int = None):
    """Cache dataset operations"""
    return cached(ttl or get_settings().CACHE_TTL_MEDIUM, "dataset")

def cache_algorithm(ttl: int = None):
    """Cache algorithm metadata"""
    return cached(ttl or get_settings().CACHE_TTL_LONG, "algorithm")

def cache_training(ttl: int = None):
    """Cache training results"""
    return cached(ttl or get_settings().CACHE_TTL_SHORT, "training")
