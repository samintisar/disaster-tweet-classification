"""Tweet entity model for disaster tweet classification system."""

from datetime import datetime
from typing import Dict, Any, Optional
import re


class Tweet:
    """Individual tweet object collected from X API v2."""

    def __init__(
        self,
        id: str,
        text: str,
        author_id: str,
        created_at: datetime,
        language: str = "en",
        public_metrics: Optional[Dict[str, int]] = None,
        entities: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Tweet entity.

        Args:
            id: Unique tweet identifier
            text: Tweet content text
            author_id: Twitter user ID of author
            created_at: Tweet creation timestamp
            language: Language code (e.g., "en")
            public_metrics: Engagement metrics dictionary
            entities: Tweet entities (hashtags, mentions, etc.)
        """
        self.id = id
        self.text = text
        self.author_id = author_id
        self.created_at = created_at
        self.language = language
        self.public_metrics = public_metrics or {}
        self.entities = entities or {}

    def get_hashtags(self) -> list:
        """Extract hashtags from tweet entities."""
        if "hashtags" in self.entities:
            return [tag["text"] for tag in self.entities["hashtags"]]
        return []

    def get_mentions(self) -> list:
        """Extract mentions from tweet entities."""
        if "mentions" in self.entities:
            return [mention["username"] for mention in self.entities["mentions"]]
        return []

    def get_urls(self) -> list:
        """Extract URLs from tweet entities."""
        if "urls" in self.entities:
            return [url["expanded_url"] or url["url"] for url in self.entities["urls"]]
        return []

    def contains_disaster_keywords(self, keywords: list) -> bool:
        """Check if tweet contains any disaster-related keywords."""
        text_lower = self.text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)

    def get_word_count(self) -> int:
        """Get word count of tweet text."""
        return len(self.text.split())

    def get_character_count(self) -> int:
        """Get character count of tweet text."""
        return len(self.text)

    def to_dict(self) -> Dict[str, Any]:
        """Convert tweet to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "author_id": self.author_id,
            "created_at": self.created_at.isoformat(),
            "language": self.language,
            "public_metrics": self.public_metrics,
            "entities": self.entities,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Tweet":
        """Create Tweet from dictionary."""
        created_at = datetime.fromisoformat(data["created_at"]) if isinstance(data["created_at"], str) else data["created_at"]
        return cls(
            id=data["id"],
            text=data["text"],
            author_id=data["author_id"],
            created_at=created_at,
            language=data.get("language", "en"),
            public_metrics=data.get("public_metrics"),
            entities=data.get("entities"),
        )

    @classmethod
    def from_api_response(cls, tweet_data: Dict, includes: Dict = None) -> "Tweet":
        """Create Tweet from X API v2 response.

        Args:
            tweet_data: Tweet data from X API response
            includes: Includes data from X API response (for user info)

        Returns:
            Tweet object
        """
        # Extract basic fields
        tweet_id = tweet_data.get("id", "")
        text = tweet_data.get("text", "")
        author_id = tweet_data.get("author_id", "")

        # Parse created_at timestamp
        created_at_str = tweet_data.get("created_at", "")
        try:
            created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            created_at = datetime.now()

        # Extract language
        language = tweet_data.get("lang", "en")

        # Extract public metrics
        public_metrics = tweet_data.get("public_metrics", {})
        if isinstance(public_metrics, dict):
            public_metrics = {k: int(v) for k, v in public_metrics.items()}

        # Extract entities
        entities = tweet_data.get("entities", {})
        if isinstance(entities, dict):
            # Normalize entities structure
            normalized_entities = {}
            for key, value in entities.items():
                if isinstance(value, list) and value:
                    normalized_entities[key] = value
            entities = normalized_entities

        # Add author information from includes if available
        if includes and "users" in includes:
            users = {user["id"]: user for user in includes["users"]}
            if author_id in users:
                author = users[author_id]
                # Add author info to entities or keep separate as needed
                pass

        return cls(
            id=tweet_id,
            text=text,
            author_id=author_id,
            created_at=created_at,
            language=language,
            public_metrics=public_metrics,
            entities=entities
        )

    def __str__(self) -> str:
        """String representation of tweet."""
        return f"Tweet(id={self.id[:8]}..., text='{self.text[:50]}...', author_id={self.author_id})"

    def __repr__(self) -> str:
        """Detailed string representation of tweet."""
        return (
            f"Tweet(id='{self.id}', text='{self.text}', author_id='{self.author_id}', "
            f"created_at={self.created_at}, language='{self.language}', "
            f"public_metrics={self.public_metrics})"
        )

    def __eq__(self, other) -> bool:
        """Check if two tweets are equal."""
        if not isinstance(other, Tweet):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash function for tweet."""
        return hash(self.id)