"""Simple feature extraction for disaster tweet classification system."""

import string
from typing import Dict, Any, List
from .text_cleaner import TextCleaner


class FeatureExtractor:
    """Simple feature extraction for tweets."""

    def __init__(self):
        """Initialize feature extractor."""
        self.text_cleaner = TextCleaner()
        self.disaster_keywords = [
            'earthquake', 'flood', 'hurricane', 'tornado', 'wildfire', 'tsunami',
            'avalanche', 'drought', 'volcano', 'storm', 'cyclone', 'typhoon',
            'disaster', 'emergency', 'evacuate', 'rescue', 'damage', 'destruction',
            'casualty', 'victim', 'injured', 'fatal', 'death', 'missing', 'trapped',
            'collapse', 'explosion', 'fire', 'burning', 'smoke', 'chemical', 'spill',
            'outbreak', 'pandemic', 'quarantine', 'lockdown', 'curfew', 'riot',
            'violence', 'shooting', 'attack', 'terror', 'bomb',
            'warning', 'alert', 'watch', 'danger', 'threat', 'risk', 'hazard',
            'crisis', 'catastrophe', 'devastation', 'ruin', 'wreckage',
            'debris', 'aftermath', 'survivor', 'shelter', 'aid', 'relief', 'donation',
            'redcross', 'fema', 'emergency', 'sos', 'help', 'save', 'urgent',
            'breaking', 'breakingnews', 'news', 'update', 'situation', 'ongoing'
        ]

    def extract_basic_features(self, text: str) -> Dict[str, Any]:
        """Extract basic text features."""
        if not text:
            return self._get_empty_features()

        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text),
            'average_word_length': sum(len(word) for word in text.split()) / max(len(text.split()), 1),
        }

        # Punctuation features
        punctuation_count = sum(1 for char in text if char in string.punctuation)
        exclamation_count = text.count('!')
        question_count = text.count('?')

        features.update({
            'punctuation_count': punctuation_count,
            'exclamation_count': exclamation_count,
            'question_count': question_count,
            'punctuation_ratio': punctuation_count / max(len(text), 1),
        })

        # Capitalization features
        uppercase_count = sum(1 for char in text if char.isupper())
        features.update({
            'uppercase_count': uppercase_count,
            'uppercase_ratio': uppercase_count / max(len(text), 1),
            'has_all_caps': int(any(word.isupper() and len(word) > 1 for word in text.split())),
        })

        return features

    def extract_tweet_entities(self, text: str) -> Dict[str, Any]:
        """Extract tweet entity features."""
        hashtags = self.text_cleaner.extract_hashtags(text)
        mentions = self.text_cleaner.extract_mentions(text)
        urls = self.text_cleaner.extract_urls(text)

        return {
            'hashtag_count': len(hashtags),
            'mention_count': len(mentions),
            'url_count': len(urls),
            'has_hashtags': int(len(hashtags) > 0),
            'has_mentions': int(len(mentions) > 0),
            'has_urls': int(len(urls) > 0),
        }

    def extract_disaster_features(self, text: str) -> Dict[str, Any]:
        """Extract disaster-related features."""
        text_lower = text.lower()

        # Find disaster keywords
        disaster_matches = [kw for kw in self.disaster_keywords if kw in text_lower]

        # Count features
        features = {
            'disaster_keyword_count': len(disaster_matches),
            'has_disaster_keywords': int(len(disaster_matches) > 0),
            'disaster_keywords': disaster_matches,
        }

        # Location indicators
        location_indicators = ['in', 'at', 'near', 'around', 'outside', 'inside', 'area']
        features['has_location_indicators'] = int(
            any(indicator in text_lower for indicator in location_indicators)
        )

        # Time indicators
        time_indicators = ['now', 'today', 'right now', 'happening', 'current', 'breaking']
        features['has_time_indicators'] = int(
            any(indicator in text_lower for indicator in time_indicators)
        )

        return features

    def extract_sentiment_features(self, text: str) -> Dict[str, Any]:
        """Extract sentiment-related features."""
        text_lower = text.lower()

        # Simple positive and negative word lists
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'safe', 'ok']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'dangerous', 'deadly', 'fatal', 'worse']

        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)

        # Emotion indicators
        emotion_words = ['scared', 'afraid', 'terrified', 'panic', 'fear', 'worried', 'anxious']
        emotion_count = sum(1 for word in emotion_words if word in text_lower)

        return {
            'positive_word_count': positive_count,
            'negative_word_count': negative_count,
            'emotion_word_count': emotion_count,
            'sentiment_score': (positive_count - negative_count) / max(len(text.split()), 1),
            'has_emotion_words': int(emotion_count > 0),
        }

    def extract_all_features(self, text: str) -> Dict[str, Any]:
        """Extract all features from text."""
        if not text:
            return self._get_empty_features()

        # Basic features
        features = self.extract_basic_features(text)

        # Tweet entities
        features.update(self.extract_tweet_entities(text))

        # Disaster features
        features.update(self.extract_disaster_features(text))

        # Sentiment features
        features.update(self.extract_sentiment_features(text))

        return features

    def _get_empty_features(self) -> Dict[str, Any]:
        """Get empty feature set for empty text."""
        return {
            'text_length': 0,
            'word_count': 0,
            'char_count': 0,
            'average_word_length': 0,
            'punctuation_count': 0,
            'exclamation_count': 0,
            'question_count': 0,
            'punctuation_ratio': 0,
            'uppercase_count': 0,
            'uppercase_ratio': 0,
            'has_all_caps': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'url_count': 0,
            'has_hashtags': 0,
            'has_mentions': 0,
            'has_urls': 0,
            'disaster_keyword_count': 0,
            'has_disaster_keywords': 0,
            'disaster_keywords': [],
            'has_location_indicators': 0,
            'has_time_indicators': 0,
            'positive_word_count': 0,
            'negative_word_count': 0,
            'emotion_word_count': 0,
            'sentiment_score': 0,
            'has_emotion_words': 0,
        }