"""Classification API endpoints for disaster tweet classification system."""

import logging
from typing import Dict, Any, Optional, Union
from datetime import datetime
import uuid

from ..inference.predictor import TweetClassificationService
from ..models.classification_result import ClassificationResult
from ..models.tweet import Tweet


class ClassificationAPI:
    """API for tweet classification operations."""

    def __init__(self, classification_service: TweetClassificationService):
        """Initialize classification API.

        Args:
            classification_service: Tweet classification service instance
        """
        self.service = classification_service
        self.logger = logging.getLogger(__name__)

        # API configuration
        self.max_text_length = 280  # Twitter character limit
        self.min_text_length = 1
        self.confidence_threshold = 0.6
        self.request_timeout = 30  # seconds

    def classify_tweet(
        self,
        text: str,
        tweet_id: Optional[str] = None,
        include_features: bool = True,
        include_keywords: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Classify a single tweet text.

        Args:
            text: Tweet text to classify
            tweet_id: Optional tweet identifier
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include metadata

        Returns:
            Classification result dictionary

        Raises:
            ValueError: If input validation fails
            RuntimeError: If classification fails
        """
        try:
            # Validate input
            validation_result = self._validate_classification_input(text)
            if not validation_result['valid']:
                raise ValueError(validation_result['error'])

            # Generate tweet ID if not provided
            if not tweet_id:
                tweet_id = str(uuid.uuid4())

            # Perform classification
            result = self.service.classify_text(
                text=text,
                tweet_id=tweet_id,
                include_features=include_features,
                include_keywords=include_keywords
            )

            # Build response
            response = self._build_classification_response(
                result, include_features, include_keywords, include_metadata
            )

            return response

        except Exception as e:
            self.logger.error(f"Tweet classification failed: {str(e)}")
            raise RuntimeError(f"Classification failed: {str(e)}")

    def classify_tweet_object(
        self,
        tweet_data: Dict[str, Any],
        include_features: bool = True,
        include_keywords: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Classify a tweet object.

        Args:
            tweet_data: Tweet data dictionary
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include metadata

        Returns:
            Classification result dictionary

        Raises:
            ValueError: If input validation fails
            RuntimeError: If classification fails
        """
        try:
            # Validate tweet object
            validation_result = self._validate_tweet_object(tweet_data)
            if not validation_result['valid']:
                raise ValueError(validation_result['error'])

            # Create tweet object
            tweet = Tweet.from_dict(tweet_data)

            # Perform classification
            result = self.service.classify_tweet(
                tweet=tweet,
                include_features=include_features,
                include_keywords=include_keywords
            )

            # Build response
            response = self._build_classification_response(
                result, include_features, include_keywords, include_metadata
            )

            return response

        except Exception as e:
            self.logger.error(f"Tweet object classification failed: {str(e)}")
            raise RuntimeError(f"Classification failed: {str(e)}")

    def batch_classify(
        self,
        tweets: list,
        include_features: bool = True,
        include_keywords: bool = True,
        include_metadata: bool = False
    ) -> Dict[str, Any]:
        """Classify multiple tweets in batch.

        Args:
            tweets: List of tweet texts or tweet objects
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include metadata

        Returns:
            Batch classification result dictionary

        Raises:
            ValueError: If input validation fails
            RuntimeError: If classification fails
        """
        try:
            # Validate batch input
            validation_result = self._validate_batch_input(tweets)
            if not validation_result['valid']:
                raise ValueError(validation_result['error'])

            # Perform batch classification
            results = self.service.classify_batch(
                tweets=tweets,
                include_features=include_features,
                include_keywords=include_keywords
            )

            # Build batch response
            response = self._build_batch_response(
                results, include_features, include_keywords, include_metadata
            )

            return response

        except Exception as e:
            self.logger.error(f"Batch classification failed: {str(e)}")
            raise RuntimeError(f"Batch classification failed: {str(e)}")

    def _validate_classification_input(self, text: str) -> Dict[str, Any]:
        """Validate single tweet classification input.

        Args:
            text: Tweet text to validate

        Returns:
            Validation result dictionary
        """
        if not isinstance(text, str):
            return {
                'valid': False,
                'error': 'Text must be a string'
            }

        if len(text.strip()) < self.min_text_length:
            return {
                'valid': False,
                'error': f'Text must be at least {self.min_text_length} character long'
            }

        if len(text) > self.max_text_length:
            return {
                'valid': False,
                'error': f'Text must be less than {self.max_text_length} characters long'
            }

        return {'valid': True}

    def _validate_tweet_object(self, tweet_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate tweet object input.

        Args:
            tweet_data: Tweet data dictionary

        Returns:
            Validation result dictionary
        """
        if not isinstance(tweet_data, dict):
            return {
                'valid': False,
                'error': 'Tweet data must be a dictionary'
            }

        required_fields = ['id', 'text', 'author_id', 'created_at']
        missing_fields = [field for field in required_fields if field not in tweet_data]

        if missing_fields:
            return {
                'valid': False,
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }

        # Validate text field
        text_validation = self._validate_classification_input(tweet_data['text'])
        if not text_validation['valid']:
            return text_validation

        return {'valid': True}

    def _validate_batch_input(self, tweets: list) -> Dict[str, Any]:
        """Validate batch classification input.

        Args:
            tweets: List of tweets to validate

        Returns:
            Validation result dictionary
        """
        if not isinstance(tweets, list):
            return {
                'valid': False,
                'error': 'Tweets must be a list'
            }

        if len(tweets) == 0:
            return {
                'valid': False,
                'error': 'Tweets list cannot be empty'
            }

        if len(tweets) > 100:  # Batch size limit
            return {
                'valid': False,
                'error': f'Batch size cannot exceed 100 tweets'
            }

        # Validate individual tweets
        for i, tweet in enumerate(tweets):
            if isinstance(tweet, str):
                validation_result = self._validate_classification_input(tweet)
            elif isinstance(tweet, dict):
                validation_result = self._validate_tweet_object(tweet)
            else:
                validation_result = {
                    'valid': False,
                    'error': f'Item {i} must be string or dictionary'
                }

            if not validation_result['valid']:
                validation_result['error'] = f'Item {i}: {validation_result["error"]}'
                return validation_result

        return {'valid': True}

    def _build_classification_response(
        self,
        result: ClassificationResult,
        include_features: bool,
        include_keywords: bool,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Build classification response dictionary.

        Args:
            result: Classification result object
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include metadata

        Returns:
            Response dictionary
        """
        response = {
            'tweet_id': result.tweet_id,
            'prediction': result.prediction,
            'confidence': result.confidence,
            'probabilities': result.probabilities,
            'timestamp': result.timestamp.isoformat(),
            'model_version': result.model_version,
            'is_high_confidence': result.is_high_confidence(self.confidence_threshold),
            'is_low_confidence': result.is_low_confidence(self.confidence_threshold),
            'classification_summary': result.get_summary()
        }

        # Add processing metadata if requested
        if include_metadata:
            service_status = self.service.get_service_status()
            response['metadata'] = {
                'processing_time': service_status['statistics']['average_processing_time'],
                'total_predictions': service_status['statistics']['total_predictions'],
                'service_uptime': (
                    datetime.now() - datetime.fromisoformat(service_status['initialization_time'])
                ).total_seconds() if service_status.get('initialization_time') else None,
                'model_info': service_status.get('model_info', {}),
                'confidence_threshold': self.confidence_threshold
            }

        return response

    def _build_batch_response(
        self,
        results: list,
        include_features: bool,
        include_keywords: bool,
        include_metadata: bool
    ) -> Dict[str, Any]:
        """Build batch classification response dictionary.

        Args:
            results: List of classification results
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            include_metadata: Whether to include metadata

        Returns:
            Batch response dictionary
        """
        # Build individual results
        individual_results = []
        for result in results:
            response = self._build_classification_response(
                result, include_features, include_keywords, False  # Don't include metadata for individual results
            )
            individual_results.append(response)

        # Build batch summary
        batch_summary = self.service.get_classification_summary(results)

        response = {
            'results': individual_results,
            'batch_summary': batch_summary,
            'batch_size': len(results),
            'processing_timestamp': datetime.now().isoformat()
        }

        # Add batch metadata if requested
        if include_metadata:
            service_status = self.service.get_service_status()
            response['metadata'] = {
                'total_processing_time': service_status['statistics']['total_processing_time'],
                'average_processing_time': service_status['statistics']['average_processing_time'],
                'total_predictions': service_status['statistics']['total_predictions'],
                'batch_configuration': {
                    'include_features': include_features,
                    'include_keywords': include_keywords,
                    'max_batch_size': self.service.batch_size
                }
            }

        return response

    def get_api_statistics(self) -> Dict[str, Any]:
        """Get API usage statistics.

        Returns:
            API statistics dictionary
        """
        service_status = self.service.get_service_status()
        stats = service_status['statistics']

        return {
            'total_requests': stats['total_predictions'],
            'successful_requests': stats['successful_predictions'],
            'failed_requests': stats['failed_requests'],
            'success_rate': (
                stats['successful_predictions'] / stats['total_predictions']
                if stats['total_predictions'] > 0 else 0
            ),
            'average_processing_time': stats['average_processing_time'],
            'confidence_distribution': stats['predictions_by_confidence'],
            'disaster_detection_rate': stats['disaster_ratio'],
            'api_uptime': (
                datetime.now() - datetime.fromisoformat(service_status['initialization_time'])
            ).total_seconds() if service_status.get('initialization_time') else None,
            'model_performance': {
                'is_ready': service_status['is_ready'],
                'model_loaded': service_status['model_info'].get('is_loaded', False),
                'model_version': service_status['model_info'].get('version')
            }
        }

    def validate_api_health(self) -> Dict[str, Any]:
        """Validate API health status.

        Returns:
            Health validation dictionary
        """
        is_healthy = self.service.is_ready()

        return {
            'api_healthy': is_healthy,
            'service_ready': self.service.is_ready(),
            'model_loaded': self.service.classifier.is_model_loaded() if self.service.classifier else False,
            'validation_timestamp': datetime.now().isoformat(),
            'recommendations': [] if is_healthy else [
                'Service not ready for classification',
                'Check model loading status',
                'Verify service initialization'
            ]
        }

    def reset_api_statistics(self):
        """Reset API statistics."""
        self.service.reset_stats()
        self.logger.info("API statistics reset")

    def configure_api(
        self,
        confidence_threshold: Optional[float] = None,
        max_text_length: Optional[int] = None,
        request_timeout: Optional[int] = None
    ):
        """Configure API parameters.

        Args:
            confidence_threshold: New confidence threshold
            max_text_length: New maximum text length
            request_timeout: New request timeout
        """
        if confidence_threshold is not None:
            self.confidence_threshold = max(0.0, min(1.0, confidence_threshold))

        if max_text_length is not None:
            self.max_text_length = max(1, min(10000, max_text_length))

        if request_timeout is not None:
            self.request_timeout = max(1, min(300, request_timeout))

        self.logger.info("API configuration updated")