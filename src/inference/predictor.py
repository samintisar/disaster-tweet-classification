"""Simple model inference service for disaster tweet classification."""

import time
import logging
from typing import Union, List, Dict, Optional
from datetime import datetime
import uuid

from ..models.tweet import Tweet
from ..models.classification_result import ClassificationResult
from ..models.processed_tweet import ProcessedTweet
from ..preprocessing.text_cleaner import TextCleaner
from ..preprocessing.features import FeatureExtractor
from ..preprocessing.keyword_detector import DisasterKeywordDetector
from ..models.disaster_classifier import DisasterTweetClassifier


class TweetClassificationService:
    """Simple inference service for tweet classification."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        confidence_threshold: float = 0.6
    ):
        """Initialize the tweet classification service.

        Args:
            model_path: Path to saved model weights
            device: Device to run inference on
            confidence_threshold: Minimum confidence for predictions
        """
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize components
        self.classifier = DisasterTweetClassifier(
            model_path=model_path,
            device=device
        )
        self.text_cleaner = TextCleaner()
        self.feature_extractor = FeatureExtractor()
        self.keyword_detector = DisasterKeywordDetector()

        # Service state
        self.is_initialized = False
        self.initialization_time = None
        self.last_prediction_time = None

        # Basic stats
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }

        self.logger = logging.getLogger(__name__)

    def initialize(self) -> bool:
        """Initialize the classification service.

        Returns:
            True if initialization successful, False otherwise
        """
        try:
            self.logger.info("Initializing tweet classification service...")

            # Load the model
            if not self.classifier.load_model():
                self.logger.error("Failed to load model")
                return False

            # Initialize keyword detector patterns
            self.keyword_detector.compile_patterns()

            # Update service state
            self.is_initialized = True
            self.initialization_time = datetime.now()

            self.logger.info("Tweet classification service initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {str(e)}")
            return False

    def is_ready(self) -> bool:
        """Check if service is ready for predictions."""
        return self.is_initialized and self.classifier.is_model_loaded()

    def classify_tweet(
        self,
        tweet: Union[str, Tweet, Dict],
        include_features: bool = True,
        include_keywords: bool = True
    ) -> ClassificationResult:
        """Classify a single tweet.

        Args:
            tweet: Tweet text, Tweet object, or tweet dictionary
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis

        Returns:
            ClassificationResult object
        """
        if not self.is_ready():
            raise RuntimeError("Service not initialized. Call initialize() first.")

        start_time = time.time()

        try:
            # Parse input tweet
            parsed_tweet = self._parse_tweet_input(tweet)
            tweet_text = parsed_tweet.text

            # Clean text
            cleaned_text = self.text_cleaner.clean_disaster_tweet(tweet_text)

            # Extract features
            features = {}
            if include_features:
                features = self.feature_extractor.extract_all_features(tweet_text)

            # Extract keywords
            keyword_matches = []
            if include_keywords:
                keyword_matches = self.keyword_detector.detect_keywords(tweet_text)

            # Get model prediction with enhanced features
            if include_features:
                prediction_result = self.classifier.predict_with_enhanced_features(
                    cleaned_text,
                    features
                )
            else:
                prediction_result = self.classifier.predict_with_confidence(cleaned_text)

            # Create processed tweet
            processed_tweet = ProcessedTweet(
                original_tweet=parsed_tweet,
                cleaned_text=cleaned_text,
                features=features
            )

            # Convert numeric prediction to string label
            prediction_label = "disaster" if prediction_result['prediction'] == 1 else "non_disaster"

            # Create classification result
            result = ClassificationResult(
                tweet_id=parsed_tweet.id,
                prediction=prediction_label,
                confidence=prediction_result['confidence'],
                probabilities=prediction_result['probabilities'],
                timestamp=datetime.now()
            )

            # Update statistics
            self._update_stats(result, time.time() - start_time)

            return result

        except Exception as e:
            self.logger.error(f"Classification failed: {str(e)}")
            self.stats['failed_predictions'] += 1
            raise

    def classify_text(
        self,
        text: str,
        tweet_id: Optional[str] = None,
        include_features: bool = True,
        include_keywords: bool = True
    ) -> ClassificationResult:
        """Classify raw text as a tweet.

        Args:
            text: Text to classify
            tweet_id: Optional tweet ID
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis

        Returns:
            ClassificationResult object
        """
        if not tweet_id:
            tweet_id = str(uuid.uuid4())

        # Create tweet object
        tweet = Tweet(
            id=tweet_id,
            text=text,
            author_id="system",
            created_at=datetime.now(),
            language="en"
        )

        return self.classify_tweet(tweet, include_features, include_keywords)

    def get_service_status(self) -> Dict:
        """Get current service status.

        Returns:
            Service status information
        """
        return {
            'is_initialized': self.is_initialized,
            'is_ready': self.is_ready(),
            'initialization_time': self.initialization_time.isoformat() if self.initialization_time else None,
            'last_prediction_time': self.last_prediction_time.isoformat() if self.last_prediction_time else None,
            'model_info': self.classifier.get_model_info() if self.classifier else None,
            'statistics': self.stats.copy(),
            'confidence_threshold': self.confidence_threshold
        }

    def _parse_tweet_input(self, tweet: Union[str, Tweet, Dict]) -> Tweet:
        """Parse various tweet input formats to Tweet object.

        Args:
            tweet: Tweet input in various formats

        Returns:
            Tweet object
        """
        if isinstance(tweet, Tweet):
            return tweet
        elif isinstance(tweet, str):
            return Tweet(
                id=str(uuid.uuid4()),
                text=tweet,
                author_id="system",
                created_at=datetime.now(),
                language="en"
            )
        elif isinstance(tweet, dict):
            return Tweet.from_dict(tweet)
        else:
            raise ValueError(f"Unsupported tweet input type: {type(tweet)}")

    def _update_stats(self, result: ClassificationResult, processing_time: float):
        """Update service statistics.

        Args:
            result: Classification result
            processing_time: Processing time in seconds
        """
        self.stats['total_predictions'] += 1
        self.stats['successful_predictions'] += 1
        self.stats['total_processing_time'] += processing_time
        self.stats['average_processing_time'] = (
            self.stats['total_processing_time'] / self.stats['total_predictions']
        )

        self.last_prediction_time = datetime.now()

    def reset_stats(self):
        """Reset service statistics."""
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0
        }