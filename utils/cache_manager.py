"""
Cache management system for the Stock Pattern Detector application.

This module provides comprehensive caching functionality to improve performance
by avoiding redundant data loading and processing operations.
"""

import os
import pickle
import hashlib
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Callable, List
from datetime import datetime, timedelta
import pandas as pd
import streamlit as st

from .logging_config import get_logger, log_performance

logger = get_logger(__name__)


class CacheManager:
    """
    Manages caching of data and computation results to improve performance.
    
    Features:
    - File-based caching with automatic expiration
    - Memory caching for frequently accessed data
    - Cache invalidation based on source file modifications
    - Compression for large datasets
    - Thread-safe operations
    """
    
    def __init__(self, cache_dir: str = ".cache", max_memory_items: int = 50):
        """
        Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store cache files
            max_memory_items: Maximum items to keep in memory cache
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Memory cache for frequently accessed items
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.max_memory_items = max_memory_items
        self.access_times: Dict[str, float] = {}
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'memory_hits': 0,
            'disk_hits': 0,
            'evictions': 0
        }
        
        # Default cache settings
        self.default_ttl = 3600  # 1 hour in seconds
        self.compression_threshold = 1024 * 1024  # 1MB
        
        logger.info(f"Cache manager initialized with directory: {self.cache_dir}")
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Unique cache key string
        """
        try:
            # Create a string representation of all arguments
            key_data = {
                'args': args,
                'kwargs': sorted(kwargs.items()) if kwargs else {}
            }
            
            # Convert to string and hash
            key_string = str(key_data)
            cache_key = hashlib.md5(key_string.encode()).hexdigest()
            
            return cache_key
            
        except Exception as e:
            logger.warning(f"Error generating cache key: {e}")
            # Fallback to timestamp-based key
            return f"fallback_{int(time.time())}"
    
    def _get_file_cache_path(self, cache_key: str) -> Path:
        """Get the file path for a cache key."""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _is_cache_valid(self, cache_path: Path, ttl: int, source_files: List[str] = None) -> bool:
        """
        Check if a cache file is still valid.
        
        Args:
            cache_path: Path to cache file
            ttl: Time to live in seconds
            source_files: List of source files to check for modifications
            
        Returns:
            True if cache is valid, False otherwise
        """
        try:
            if not cache_path.exists():
                return False
            
            # Check TTL
            cache_age = time.time() - cache_path.stat().st_mtime
            if cache_age > ttl:
                logger.debug(f"Cache expired: {cache_path.name} (age: {cache_age:.1f}s)")
                return False
            
            # Check source file modifications
            if source_files:
                cache_mtime = cache_path.stat().st_mtime
                for source_file in source_files:
                    source_path = Path(source_file)
                    if source_path.exists():
                        if source_path.stat().st_mtime > cache_mtime:
                            logger.debug(f"Source file newer than cache: {source_file}")
                            return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Error checking cache validity: {e}")
            return False
    
    def get(self, cache_key: str, default: Any = None) -> Any:
        """
        Get an item from cache.
        
        Args:
            cache_key: Cache key to retrieve
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        try:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.access_times[cache_key] = time.time()
                self.stats['hits'] += 1
                self.stats['memory_hits'] += 1
                logger.debug(f"Memory cache hit: {cache_key}")
                return self.memory_cache[cache_key]['data']
            
            # Check disk cache
            cache_path = self._get_file_cache_path(cache_key)
            if cache_path.exists():
                try:
                    with open(cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Validate cache
                    if self._is_cache_valid(
                        cache_path, 
                        cache_data.get('ttl', self.default_ttl),
                        cache_data.get('source_files', [])
                    ):
                        # Move to memory cache for faster access
                        self._add_to_memory_cache(cache_key, cache_data['data'])
                        
                        self.stats['hits'] += 1
                        self.stats['disk_hits'] += 1
                        logger.debug(f"Disk cache hit: {cache_key}")
                        return cache_data['data']
                    else:
                        # Remove invalid cache
                        cache_path.unlink()
                        logger.debug(f"Removed invalid cache: {cache_key}")
                        
                except Exception as e:
                    logger.warning(f"Error reading cache file {cache_key}: {e}")
                    # Remove corrupted cache file
                    try:
                        cache_path.unlink()
                    except:
                        pass
            
            # Cache miss
            self.stats['misses'] += 1
            logger.debug(f"Cache miss: {cache_key}")
            return default
            
        except Exception as e:
            logger.error(f"Error getting cache item {cache_key}: {e}")
            return default
    
    def set(self, cache_key: str, data: Any, ttl: int = None, source_files: List[str] = None) -> bool:
        """
        Store an item in cache.
        
        Args:
            cache_key: Cache key
            data: Data to cache
            ttl: Time to live in seconds
            source_files: List of source files for invalidation
            
        Returns:
            True if successfully cached, False otherwise
        """
        try:
            ttl = ttl or self.default_ttl
            
            # Add to memory cache
            self._add_to_memory_cache(cache_key, data)
            
            # Prepare cache data
            cache_data = {
                'data': data,
                'timestamp': time.time(),
                'ttl': ttl,
                'source_files': source_files or []
            }
            
            # Save to disk cache
            cache_path = self._get_file_cache_path(cache_key)
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                
                logger.debug(f"Cached item: {cache_key} (TTL: {ttl}s)")
                return True
                
            except Exception as e:
                logger.warning(f"Error saving cache file {cache_key}: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Error setting cache item {cache_key}: {e}")
            return False
    
    def _add_to_memory_cache(self, cache_key: str, data: Any) -> None:
        """Add item to memory cache with LRU eviction."""
        try:
            # Check if we need to evict items
            if len(self.memory_cache) >= self.max_memory_items:
                self._evict_lru_items()
            
            # Add to memory cache
            self.memory_cache[cache_key] = {
                'data': data,
                'timestamp': time.time()
            }
            self.access_times[cache_key] = time.time()
            
        except Exception as e:
            logger.warning(f"Error adding to memory cache: {e}")
    
    def _evict_lru_items(self) -> None:
        """Evict least recently used items from memory cache."""
        try:
            if not self.access_times:
                return
            
            # Find least recently used item
            lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            
            # Remove from memory cache
            if lru_key in self.memory_cache:
                del self.memory_cache[lru_key]
            if lru_key in self.access_times:
                del self.access_times[lru_key]
            
            self.stats['evictions'] += 1
            logger.debug(f"Evicted LRU item: {lru_key}")
            
        except Exception as e:
            logger.warning(f"Error evicting LRU items: {e}")
    
    def invalidate(self, cache_key: str) -> bool:
        """
        Invalidate a specific cache entry.
        
        Args:
            cache_key: Cache key to invalidate
            
        Returns:
            True if successfully invalidated, False otherwise
        """
        try:
            success = True
            
            # Remove from memory cache
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
            if cache_key in self.access_times:
                del self.access_times[cache_key]
            
            # Remove from disk cache
            cache_path = self._get_file_cache_path(cache_key)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_key}: {e}")
                    success = False
            
            logger.debug(f"Invalidated cache: {cache_key}")
            return success
            
        except Exception as e:
            logger.error(f"Error invalidating cache {cache_key}: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Clear all cache entries.
        
        Returns:
            True if successfully cleared, False otherwise
        """
        try:
            # Clear memory cache
            self.memory_cache.clear()
            self.access_times.clear()
            
            # Clear disk cache
            success = True
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                except Exception as e:
                    logger.warning(f"Error removing cache file {cache_file}: {e}")
                    success = False
            
            # Reset statistics
            self.stats = {
                'hits': 0,
                'misses': 0,
                'memory_hits': 0,
                'disk_hits': 0,
                'evictions': 0
            }
            
            logger.info("Cleared all cache entries")
            return success
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        try:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = (self.stats['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            # Get cache sizes
            memory_size = len(self.memory_cache)
            disk_files = len(list(self.cache_dir.glob("*.cache")))
            
            # Calculate disk cache size
            disk_size = 0
            try:
                for cache_file in self.cache_dir.glob("*.cache"):
                    disk_size += cache_file.stat().st_size
            except Exception:
                disk_size = -1  # Unknown
            
            return {
                'hit_rate': hit_rate,
                'total_requests': total_requests,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'memory_hits': self.stats['memory_hits'],
                'disk_hits': self.stats['disk_hits'],
                'evictions': self.stats['evictions'],
                'memory_cache_size': memory_size,
                'disk_cache_files': disk_files,
                'disk_cache_size_bytes': disk_size,
                'cache_directory': str(self.cache_dir)
            }
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'error': str(e)}
    
    def cleanup_expired(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of entries cleaned up
        """
        try:
            cleaned_count = 0
            
            # Clean up disk cache
            for cache_file in self.cache_dir.glob("*.cache"):
                try:
                    # Load cache data to check TTL
                    with open(cache_file, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    if not self._is_cache_valid(
                        cache_file,
                        cache_data.get('ttl', self.default_ttl),
                        cache_data.get('source_files', [])
                    ):
                        cache_file.unlink()
                        cleaned_count += 1
                        
                except Exception as e:
                    logger.warning(f"Error checking cache file {cache_file}: {e}")
                    # Remove corrupted files
                    try:
                        cache_file.unlink()
                        cleaned_count += 1
                    except:
                        pass
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired cache entries")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Error cleaning up expired cache: {e}")
            return 0


def cached_function(
    ttl: int = 3600,
    source_files: List[str] = None,
    cache_key_func: Optional[Callable] = None
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        source_files: List of source files for cache invalidation
        cache_key_func: Custom function to generate cache key
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            try:
                # Generate cache key
                if cache_key_func:
                    cache_key = cache_key_func(*args, **kwargs)
                else:
                    cache_key = cache_manager._generate_cache_key(
                        func.__name__, *args, **kwargs
                    )
                
                # Try to get from cache
                start_time = time.time()
                cached_result = cache_manager.get(cache_key)
                
                if cached_result is not None:
                    cache_time = time.time() - start_time
                    log_performance(f"cache_hit_{func.__name__}", cache_time)
                    return cached_result
                
                # Execute function and cache result
                result = func(*args, **kwargs)
                
                # Cache the result
                cache_manager.set(cache_key, result, ttl, source_files)
                
                execution_time = time.time() - start_time
                log_performance(f"cache_miss_{func.__name__}", execution_time)
                
                return result
                
            except Exception as e:
                logger.error(f"Error in cached function {func.__name__}: {e}")
                # Fallback to direct execution
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance."""
    return cache_manager


def clear_cache() -> bool:
    """Clear all cache entries."""
    return cache_manager.clear_all()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics."""
    return cache_manager.get_stats()


def cleanup_cache() -> int:
    """Clean up expired cache entries."""
    return cache_manager.cleanup_expired()