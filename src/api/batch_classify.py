"""Batch classification API endpoint for disaster tweet classification system."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from .classify import ClassificationAPI


class BatchClassificationAPI:
    """API for batch tweet classification operations."""

    def __init__(self, classification_api: ClassificationAPI):
        """Initialize batch classification API.

        Args:
            classification_api: Classification API instance
        """
        self.classification_api = classification_api
        self.logger = logging.getLogger(__name__)

        # Batch processing configuration
        self.max_batch_size = 100
        self.min_batch_size = 1
        self.timeout_seconds = 30

    def process_batch_classification(
        self,
        tweets: List[str],
        include_features: bool = True,
        include_keywords: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Process batch classification request.

        Args:
            tweets: List of tweet texts to classify
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include processing metadata

        Returns:
            Batch classification result dictionary

        Raises:
            ValueError: If input validation fails
            RuntimeError: If batch processing fails
        """
        try:
            # Validate batch input
            self._validate_batch_input(tweets)

            # Process batch classification
            result = self.classification_api.batch_classify(
                tweets=tweets,
                include_features=include_features,
                include_keywords=include_keywords,
                include_metadata=include_metadata
            )

            return result

        except Exception as e:
            self.logger.error(f"Batch classification failed: {str(e)}")
            raise RuntimeError(f"Batch classification failed: {str(e)}")

    def _validate_batch_input(self, tweets: List[str]) -> None:
        """Validate batch input parameters.

        Args:
            tweets: List of tweet texts to validate

        Raises:
            ValueError: If validation fails
        """
        if not isinstance(tweets, list):
            raise ValueError("Tweets must be a list")

        if len(tweets) < self.min_batch_size:
            raise ValueError(f"Batch must contain at least {self.min_batch_size} tweet")

        if len(tweets) > self.max_batch_size:
            raise ValueError(f"Batch cannot exceed {self.max_batch_size} tweets")

        # Validate individual tweets
        for i, tweet in enumerate(tweets):
            if not isinstance(tweet, str):
                raise ValueError(f"Tweet at index {i} must be a string")

            if len(tweet.strip()) == 0:
                raise ValueError(f"Tweet at index {i} cannot be empty")

            if len(tweet) > 280:
                raise ValueError(f"Tweet at index {i} exceeds 280 character limit")

    def get_batch_processing_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.

        Returns:
            Batch processing statistics dictionary
        """
        api_stats = self.classification_api.get_api_statistics()

        return {
            "batch_configuration": {
                "max_batch_size": self.max_batch_size,
                "min_batch_size": self.min_batch_size,
                "timeout_seconds": self.timeout_seconds
            },
            "api_statistics": api_stats,
            "timestamp": datetime.now().isoformat()
        }

    def validate_batch_size(self, batch_size: int) -> bool:
        """Validate if batch size is within acceptable limits.

        Args:
            batch_size: Size of batch to validate

        Returns:
            True if batch size is valid, False otherwise
        """
        return self.min_batch_size <= batch_size <= self.max_batch_size

    def estimate_processing_time(self, batch_size: int) -> float:
        """Estimate processing time for a batch.

        Args:
            batch_size: Size of batch to estimate

        Returns:
            Estimated processing time in seconds
        """
        # Base estimation: 0.1 seconds per tweet + overhead
        base_time_per_tweet = 0.1
        overhead = 0.5

        return (batch_size * base_time_per_tweet) + overhead