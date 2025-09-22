"""Configuration settings for the disaster tweet classification system."""

import os
from typing import Dict, Any

class Config:
    """Configuration class for the disaster tweet classification system."""

    # Model configuration
    MODEL_NAME = "distilbert-base-uncased"
    MODEL_VERSION = "1.0.0"
    MAX_SEQUENCE_LENGTH = 512

    # API configuration
    API_RATE_LIMIT = 300  # requests per 15 minutes
    API_POLLING_INTERVAL = 60  # seconds

    # Performance targets
    MAX_RESPONSE_TIME = 1.0  # seconds
    MAX_INFERENCE_TIME = 0.1  # seconds
    MAX_MEMORY_USAGE = 1024  # MB

    # Data processing
    MAX_TWEET_LENGTH = 280
    BATCH_SIZE = 32

    # Streaming configuration
    MAX_STREAM_DURATION = 3600  # seconds (1 hour)
    MAX_TWEETS_PER_STREAM = 1000

    # Keywords for disaster detection
    DISASTER_KEYWORDS = [
        "earthquake", "flood", "wildfire", "hurricane", "tornado",
        "tsunami", "avalanche", "volcano", "disaster", "emergency",
        "evacuation", "rescue", "damage", "casualty", "injury"
    ]

    @classmethod
    def get_model_config(cls) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "model_name": cls.MODEL_NAME,
            "model_version": cls.MODEL_VERSION,
            "max_sequence_length": cls.MAX_SEQUENCE_LENGTH,
        }

    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration."""
        return {
            "rate_limit": cls.API_RATE_LIMIT,
            "polling_interval": cls.API_POLLING_INTERVAL,
            "max_stream_duration": cls.MAX_STREAM_DURATION,
            "max_tweets_per_stream": cls.MAX_TWEETS_PER_STREAM,
        }

    @classmethod
    def get_performance_targets(cls) -> Dict[str, Any]:
        """Get performance targets."""
        return {
            "max_response_time": cls.MAX_RESPONSE_TIME,
            "max_inference_time": cls.MAX_INFERENCE_TIME,
            "max_memory_usage": cls.MAX_MEMORY_USAGE,
        }