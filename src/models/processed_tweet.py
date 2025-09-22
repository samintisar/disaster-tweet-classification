"""ProcessedTweet entity model for disaster tweet classification system."""

from datetime import datetime
from typing import Dict, Any, List, Optional
from .tweet import Tweet


class ProcessedTweet:
    """Tweet after preprocessing pipeline."""

    def __init__(
        self,
        original_tweet: Tweet,
        cleaned_text: str,
        features: Optional[Dict[str, Any]] = None,
        processing_timestamp: Optional[datetime] = None,
    ):
        """Initialize ProcessedTweet entity.

        Args:
            original_tweet: Original tweet object
            cleaned_text: Preprocessed text
            features: Extracted features dictionary
            processing_timestamp: When tweet was processed
        """
        self.original_tweet = original_tweet
        self.cleaned_text = cleaned_text
        self.features = features or {}
        self.processing_timestamp = processing_timestamp or datetime.now()

    def get_text_length(self) -> int:
        """Get text length of cleaned text."""
        return len(self.cleaned_text)

    def get_word_count(self) -> int:
        """Get word count of cleaned text."""
        return len(self.cleaned_text.split())

    def get_hashtag_count(self) -> int:
        """Get hashtag count from features."""
        return self.features.get("hashtag_count", 0)

    def get_mention_count(self) -> int:
        """Get mention count from features."""
        return self.features.get("mention_count", 0)

    def get_url_count(self) -> int:
        """Get URL count from features."""
        return self.features.get("url_count", 0)

    def get_sentiment_score(self) -> float:
        """Get sentiment score from features."""
        return self.features.get("sentiment_score", 0.0)

    def get_disaster_keywords(self) -> List[str]:
        """Get disaster keywords from features."""
        return self.features.get("disaster_keywords", [])

    def get_feature_summary(self) -> Dict[str, Any]:
        """Get a summary of extracted features."""
        return {
            "text_length": self.get_text_length(),
            "word_count": self.get_word_count(),
            "hashtag_count": self.get_hashtag_count(),
            "mention_count": self.get_mention_count(),
            "url_count": self.get_url_count(),
            "sentiment_score": self.get_sentiment_score(),
            "disaster_keywords": self.get_disaster_keywords(),
        }

    def contains_disaster_keywords(self, keywords: List[str]) -> bool:
        """Check if cleaned text contains any disaster-related keywords."""
        text_lower = self.cleaned_text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def has_high_sentiment_impact(self, threshold: float = 0.5) -> bool:
        """Check if tweet has high sentiment impact."""
        sentiment = abs(self.get_sentiment_score())
        return sentiment >= threshold

    def is_likely_disaster(self, keywords: List[str]) -> bool:
        """Heuristic to determine if tweet is likely about disaster."""
        return (
            self.contains_disaster_keywords(keywords) or
            self.has_high_sentiment_impact() or
            len(self.get_disaster_keywords()) > 0
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert processed tweet to dictionary."""
        return {
            "original_tweet": self.original_tweet.to_dict(),
            "cleaned_text": self.cleaned_text,
            "features": self.features,
            "processing_timestamp": self.processing_timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedTweet":
        """Create ProcessedTweet from dictionary."""
        original_tweet = Tweet.from_dict(data["original_tweet"])
        processing_timestamp = (
            datetime.fromisoformat(data["processing_timestamp"])
            if isinstance(data["processing_timestamp"], str)
            else data["processing_timestamp"]
        )
        return cls(
            original_tweet=original_tweet,
            cleaned_text=data["cleaned_text"],
            features=data.get("features"),
            processing_timestamp=processing_timestamp,
        )

    def __str__(self) -> str:
        """String representation of processed tweet."""
        return f"ProcessedTweet(id={self.original_tweet.id[:8]}..., cleaned_text='{self.cleaned_text[:50]}...')"

    def __repr__(self) -> str:
        """Detailed string representation of processed tweet."""
        return (
            f"ProcessedTweet(original_tweet={self.original_tweet}, "
            f"cleaned_text='{self.cleaned_text}', features={self.features}, "
            f"processing_timestamp={self.processing_timestamp})"
        )

    def __eq__(self, other) -> bool:
        """Check if two processed tweets are equal."""
        if not isinstance(other, ProcessedTweet):
            return False
        return (
            self.original_tweet == other.original_tweet and
            self.cleaned_text == other.cleaned_text and
            self.features == other.features
        )

    def __hash__(self) -> int:
        """Hash function for processed tweet."""
        return hash((self.original_tweet.id, self.cleaned_text, frozenset(self.features.items())))