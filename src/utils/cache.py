"""Simple response caching for API responses."""

import json
import hashlib
import time
from typing import Any, Optional, Dict
from datetime import datetime, timedelta
import logging

class SimpleCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        """Initialize cache.

        Args:
            max_size: Maximum number of items in cache
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.logger = logging.getLogger(__name__)

    def _generate_key(self, func_name: str, *args, **kwargs) -> str:
        """Generate cache key from function name and arguments."""
        # Create a string representation of arguments
        args_str = json.dumps([args, kwargs], sort_keys=True, default=str)
        combined = f"{func_name}:{args_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache.

        Args:
            key: Cache key

        Returns:
            Cached item or None if not found or expired
        """
        if key not in self.cache:
            return None

        item = self.cache[key]
        if time.time() > item['expires_at']:
            # Remove expired item
            del self.cache[key]
            return None

        return item['data']

    def set(self, key: str, data: Any, ttl: Optional[int] = None) -> None:
        """Set item in cache.

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        # Remove oldest item if cache is full
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k]['expires_at'])
            del self.cache[oldest_key]

        # Calculate expiration time
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl

        self.cache[key] = {
            'data': data,
            'expires_at': expires_at,
            'created_at': time.time()
        }

    def clear(self) -> None:
        """Clear all items from cache."""
        self.cache.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Cache statistics
        """
        now = time.time()
        active_items = sum(1 for item in self.cache.values() if item['expires_at'] > now)

        return {
            'total_items': len(self.cache),
            'active_items': active_items,
            'max_size': self.max_size,
            'default_ttl': self.default_ttl
        }

    def cleanup_expired(self) -> None:
        """Remove expired items from cache."""
        now = time.time()
        expired_keys = [
            key for key, item in self.cache.items()
            if item['expires_at'] <= now
        ]

        for key in expired_keys:
            del self.cache[key]

        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache items")


# Global cache instance
cache = SimpleCache(max_size=1000, default_ttl=300)  # 5 minutes default TTL


def cached_response(ttl: int = 300):
    """Decorator for caching API responses.

    Args:
        ttl: Time-to-live in seconds
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            key = cache._generate_key(func.__name__, *args, **kwargs)

            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result

            # Call function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)

            return result
        return wrapper
    return decorator