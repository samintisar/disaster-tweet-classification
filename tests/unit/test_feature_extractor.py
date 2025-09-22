"""Unit tests for the FeatureExtractor class."""

import pytest
from src.preprocessing.features import FeatureExtractor


class TestFeatureExtractor:
    """Unit tests for FeatureExtractor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = FeatureExtractor()

    def test_initialization(self):
        """Test extractor initialization."""
        assert hasattr(self.extractor, 'disaster_keywords')
        assert isinstance(self.extractor.disaster_keywords, list)
        assert len(self.extractor.disaster_keywords) > 0
        assert hasattr(self.extractor, 'text_cleaner')

    def test_extract_basic_features(self):
        """Test basic feature extraction."""
        text = "Hello World! This is a test tweet."
        features = self.extractor.extract_basic_features(text)

        assert isinstance(features, dict)
        assert 'text_length' in features
        assert 'word_count' in features
        assert 'char_count' in features
        assert 'average_word_length' in features
        assert 'punctuation_count' in features
        assert 'exclamation_count' in features
        assert 'question_count' in features

        assert features['text_length'] == len(text)
        assert features['word_count'] == 7
        assert features['char_count'] == len(text)
        assert features['exclamation_count'] == 1

    def test_extract_tweet_entities(self):
        """Test tweet entity extraction."""
        text = "This is a #test tweet with @mention and https://example.com"
        features = self.extractor.extract_tweet_entities(text)

        assert isinstance(features, dict)
        assert 'hashtag_count' in features
        assert 'mention_count' in features
        assert 'url_count' in features
        assert 'has_hashtags' in features
        assert 'has_mentions' in features
        assert 'has_urls' in features

        assert features['hashtag_count'] == 1
        assert features['mention_count'] == 1
        assert features['url_count'] == 1
        assert features['has_hashtags'] == 1
        assert features['has_mentions'] == 1
        assert features['has_urls'] == 1

    def test_extract_disaster_features(self):
        """Test disaster feature extraction."""
        text = "Major earthquake hits California area right now"
        features = self.extractor.extract_disaster_features(text)

        assert isinstance(features, dict)
        assert 'disaster_keyword_count' in features
        assert 'has_disaster_keywords' in features
        assert 'disaster_keywords' in features
        assert 'has_location_indicators' in features
        assert 'has_time_indicators' in features

        assert features['disaster_keyword_count'] >= 1
        assert 'earthquake' in features['disaster_keywords']
        assert features['has_disaster_keywords'] == 1
        assert features['has_location_indicators'] == 1
        assert features['has_time_indicators'] == 1

    def test_extract_sentiment_features(self):
        """Test sentiment feature extraction."""
        text = "This is terrible and dangerous, causing fear"
        features = self.extractor.extract_sentiment_features(text)

        assert isinstance(features, dict)
        assert 'positive_word_count' in features
        assert 'negative_word_count' in features
        assert 'emotion_word_count' in features
        assert 'sentiment_score' in features
        assert 'has_emotion_words' in features

        assert features['negative_word_count'] >= 1  # 'terrible', 'dangerous'
        assert features['emotion_word_count'] >= 1  # 'fear'
        assert features['has_emotion_words'] == 1

    def test_extract_all_features(self):
        """Test complete feature extraction."""
        text = "Major #earthquake! Causes damage and fear in California right now"
        features = self.extractor.extract_all_features(text)

        assert isinstance(features, dict)
        # Check that all feature types are included
        assert 'text_length' in features  # basic features
        assert 'hashtag_count' in features  # tweet entities
        assert 'disaster_keyword_count' in features  # disaster features
        assert 'sentiment_score' in features  # sentiment features

    def test_empty_text_handling(self):
        """Test handling of empty text."""
        features = self.extractor.extract_all_features("")
        assert isinstance(features, dict)
        assert features['text_length'] == 0
        assert features['word_count'] == 0
        assert features['disaster_keyword_count'] == 0

    def test_none_text_handling(self):
        """Test handling of None text."""
        features = self.extractor.extract_all_features(None)
        assert isinstance(features, dict)
        assert features['text_length'] == 0
        assert features['word_count'] == 0

    def test_get_empty_features(self):
        """Test empty feature set structure."""
        empty_features = self.extractor._get_empty_features()
        assert isinstance(empty_features, dict)
        assert empty_features['text_length'] == 0
        assert empty_features['word_count'] == 0
        assert empty_features['disaster_keyword_count'] == 0
        assert empty_features['sentiment_score'] == 0.0

    def test_disaster_keywords_detection(self):
        """Test that disaster keywords are properly detected."""
        disaster_terms = ['earthquake', 'flood', 'hurricane', 'disaster', 'emergency']
        for term in disaster_terms:
            text = f"There was a major {term} in the area"
            features = self.extractor.extract_disaster_features(text)
            assert term in features['disaster_keywords']
            assert features['has_disaster_keywords'] == 1

    def test_feature_values_types(self):
        """Test that feature values have correct types."""
        text = "Test tweet with #hashtag and @mention"
        features = self.extractor.extract_all_features(text)

        # Numeric features should be int or float
        numeric_features = ['text_length', 'word_count', 'punctuation_count',
                          'uppercase_ratio', 'sentiment_score']
        for feature in numeric_features:
            if feature in features:
                assert isinstance(features[feature], (int, float))

        # Boolean features should be int (0 or 1)
        boolean_features = ['has_all_caps', 'has_hashtags', 'has_mentions',
                           'has_disaster_keywords', 'has_emotion_words']
        for feature in boolean_features:
            if feature in features:
                assert features[feature] in [0, 1]