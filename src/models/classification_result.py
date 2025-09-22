"""ClassificationResult entity model for disaster tweet classification system."""

from datetime import datetime
from typing import Dict, Any, Optional


class ClassificationResult:
    """Model prediction output for tweet classification."""

    def __init__(
        self,
        tweet_id: str,
        prediction: str,
        confidence: float,
        probabilities: Dict[str, float],
        timestamp: Optional[datetime] = None,
        model_version: str = "1.0.0",
    ):
        """Initialize ClassificationResult entity.

        Args:
            tweet_id: Reference to original tweet
            prediction: Binary classification ("disaster" | "non_disaster")
            confidence: Confidence score (0.0 - 1.0)
            probabilities: Class probabilities dictionary
            timestamp: Prediction timestamp
            model_version: Model version identifier
        """
        self.tweet_id = tweet_id
        self.prediction = prediction
        self.confidence = confidence
        self.probabilities = probabilities
        self.timestamp = timestamp or datetime.now()
        self.model_version = model_version

    def get_highest_probability_class(self) -> str:
        """Get the class with the highest probability."""
        return max(self.probabilities, key=self.probabilities.get)

    def get_disaster_probability(self) -> float:
        """Get disaster class probability."""
        return self.probabilities.get("disaster", 0.0)

    def get_non_disaster_probability(self) -> float:
        """Get non-disaster class probability."""
        return self.probabilities.get("non_disaster", 0.0)

    def is_disaster(self) -> bool:
        """Check if classification indicates disaster."""
        return self.prediction == "disaster"

    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if prediction is high confidence."""
        return self.confidence >= threshold

    def is_low_confidence(self, threshold: float = 0.6) -> bool:
        """Check if prediction is low confidence."""
        return self.confidence < threshold

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the classification result."""
        return {
            "tweet_id": self.tweet_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "disaster_probability": self.get_disaster_probability(),
            "non_disaster_probability": self.get_non_disaster_probability(),
            "is_high_confidence": self.is_high_confidence(),
            "is_low_confidence": self.is_low_confidence(),
            "model_version": self.model_version,
            "timestamp": self.timestamp.isoformat(),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert classification result to dictionary."""
        return {
            "tweet_id": self.tweet_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "timestamp": self.timestamp.isoformat(),
            "model_version": self.model_version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClassificationResult":
        """Create ClassificationResult from dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"]) if isinstance(data["timestamp"], str) else data["timestamp"]
        return cls(
            tweet_id=data["tweet_id"],
            prediction=data["prediction"],
            confidence=data["confidence"],
            probabilities=data["probabilities"],
            timestamp=timestamp,
            model_version=data.get("model_version", "1.0.0"),
        )

    def __str__(self) -> str:
        """String representation of classification result."""
        return f"ClassificationResult(tweet_id={self.tweet_id[:8]}..., prediction={self.prediction}, confidence={self.confidence:.2f})"

    def __repr__(self) -> str:
        """Detailed string representation of classification result."""
        return (
            f"ClassificationResult(tweet_id='{self.tweet_id}', prediction='{self.prediction}', "
            f"confidence={self.confidence}, probabilities={self.probabilities}, "
            f"timestamp={self.timestamp}, model_version='{self.model_version}')"
        )

    def __eq__(self, other) -> bool:
        """Check if two classification results are equal."""
        if not isinstance(other, ClassificationResult):
            return False
        return (
            self.tweet_id == other.tweet_id and
            self.prediction == other.prediction and
            abs(self.confidence - other.confidence) < 1e-6 and
            self.probabilities == other.probabilities
        )

    def __hash__(self) -> int:
        """Hash function for classification result."""
        return hash((self.tweet_id, self.prediction, round(self.confidence, 6)))