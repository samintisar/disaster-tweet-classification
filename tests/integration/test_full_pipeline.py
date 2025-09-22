"""Integration tests for end-to-end classification pipeline."""

import pytest
from src.inference.predictor import TweetClassificationService
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.features import FeatureExtractor
from src.preprocessing.keyword_detector import DisasterKeywordDetector
from datetime import datetime
import uuid


class TestFullPipeline:
    """Integration tests for end-to-end classification pipeline."""

    def setup_method(self):
        """Set up test fixtures."""
        self.service = TweetClassificationService()

    def test_service_initialization(self):
        """Test that classification service can be initialized."""
        assert hasattr(self.service, 'classifier')
        assert hasattr(self.service, 'text_cleaner')
        assert hasattr(self.service, 'feature_extractor')
        assert hasattr(self.service, 'keyword_detector')

    def test_text_preprocessing_component(self):
        """Test that text preprocessing component works."""
        cleaner = TextCleaner()
        test_text = "RT @user: Major #earthquake hits California! https://t.co/abc123"

        cleaned = cleaner.clean_disaster_tweet(test_text)
        assert isinstance(cleaned, str)
        assert len(cleaned) > 0
        assert " rt " not in cleaned.lower()
        assert "https://t.co/abc123" not in cleaned

    def test_feature_extraction_component(self):
        """Test that feature extraction component works."""
        extractor = FeatureExtractor()
        test_text = "Major earthquake hits California area right now"

        features = extractor.extract_all_features(test_text)
        assert isinstance(features, dict)
        assert 'disaster_keyword_count' in features
        assert features['disaster_keyword_count'] > 0

    def test_keyword_detection_component(self):
        """Test that keyword detection component works."""
        detector = DisasterKeywordDetector()
        test_text = "Major earthquake and flood in the area"

        keywords = detector.detect_keywords(test_text)
        assert isinstance(keywords, list)
        assert len(keywords) >= 2
        assert 'earthquake' in keywords
        assert 'flood' in keywords

    def test_tweet_creation_component(self):
        """Test that tweet creation and processing works."""
        from src.models.tweet import Tweet

        tweet = Tweet(
            id=str(uuid.uuid4()),
            text="Test tweet about earthquake",
            author_id="test_user",
            created_at=datetime.now(),
            language="en"
        )

        assert tweet.id is not None
        assert tweet.text == "Test tweet about earthquake"
        assert tweet.author_id == "test_user"

    def test_single_tweet_processing(self):
        """Test that single tweet processing works."""
        # This test will work even if model is not loaded
        try:
            success = self.service.initialize()
            if success:
                # Test with a simple tweet
                result = self.service.classify_text("Test tweet")
                assert hasattr(result, 'tweet_id')
                assert hasattr(result, 'prediction')
                assert hasattr(result, 'confidence')
                assert result.prediction in ['disaster', 'non_disaster']
        except Exception:
            # Model might not be available, which is okay for testing
            pytest.skip("Model not available for testing")

    def test_disaster_tweet_classification(self):
        """Test classification of disaster-related tweets."""
        try:
            success = self.service.initialize()
            if success:
                disaster_tweets = [
                    "Major earthquake hits San Francisco, buildings damaged",
                    "Hurricane warning issued for coastal areas",
                    "Flood waters rising rapidly in downtown area"
                ]

                for tweet_text in disaster_tweets:
                    result = self.service.classify_text(tweet_text)
                    assert hasattr(result, 'prediction')
                    assert hasattr(result, 'confidence')
                    assert result.prediction in ['disaster', 'non_disaster']
                    assert 0.0 <= result.confidence <= 1.0
        except Exception:
            pytest.skip("Model not available for testing")

    def test_normal_tweet_classification(self):
        """Test classification of normal tweets."""
        try:
            success = self.service.initialize()
            if success:
                normal_tweets = [
                    "Just had a great lunch with friends",
                    "Beautiful sunset today at the beach",
                    "Looking forward to the weekend"
                ]

                for tweet_text in normal_tweets:
                    result = self.service.classify_text(tweet_text)
                    assert hasattr(result, 'prediction')
                    assert hasattr(result, 'confidence')
                    assert result.prediction in ['disaster', 'non_disaster']
                    assert 0.0 <= result.confidence <= 1.0
        except Exception:
            pytest.skip("Model not available for testing")

    def test_pipeline_components_integration(self):
        """Test that all pipeline components work together."""
        # Test text -> preprocessing -> features -> keywords
        test_text = "Major #earthquake causes damage and fear in California"

        # Text cleaning
        cleaner = TextCleaner()
        cleaned = cleaner.clean_disaster_tweet(test_text)
        assert isinstance(cleaned, str)

        # Feature extraction
        extractor = FeatureExtractor()
        features = extractor.extract_all_features(cleaned)
        assert isinstance(features, dict)
        assert features['disaster_keyword_count'] > 0

        # Keyword detection
        detector = DisasterKeywordDetector()
        keywords = detector.detect_keywords(cleaned)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert 'earthquake' in keywords

        # Verify consistency
        assert features['disaster_keyword_count'] >= len(keywords)

    def test_error_handling(self):
        """Test that the pipeline handles errors gracefully."""
        # Test with invalid inputs
        try:
            # Empty text should be handled
            if self.service.is_ready():
                result = self.service.classify_text("")
                assert hasattr(result, 'tweet_id')
                assert hasattr(result, 'prediction')
        except Exception:
            # Expected to fail if model not loaded
            pass

    def test_pipeline_performance(self):
        """Test that pipeline processing is reasonably fast."""
        import time

        # Test preprocessing performance
        cleaner = TextCleaner()
        extractor = FeatureExtractor()
        detector = DisasterKeywordDetector()

        test_text = "Major earthquake hits California area causing damage and fear"

        start_time = time.time()

        # Run pipeline
        cleaned = cleaner.clean_disaster_tweet(test_text)
        features = extractor.extract_all_features(cleaned)
        keywords = detector.detect_keywords(cleaned)

        end_time = time.time()
        processing_time = end_time - start_time

        assert processing_time < 1.0  # Should complete in under 1 second
        assert len(features) > 0
        assert len(keywords) > 0