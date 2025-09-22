"""Streaming API endpoints for disaster tweet classification system."""

import logging
import uuid
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from ..streaming.tweet_streamer import TweetStreamer, StreamConfig, StreamStatus
from ..inference.predictor import TweetClassificationService
from ..api.x_api_client import XAPIClient


@dataclass
class StreamSession:
    """Represents an active streaming session."""
    stream_id: str
    keywords: List[str]
    interval: int
    created_at: datetime
    last_activity: datetime
    status: str = "active"
    total_tweets_processed: int = 0
    total_disaster_tweets: int = 0
    errors: List[str] = field(default_factory=list)


class StreamingAPI:
    """API for tweet streaming operations."""

    def __init__(
        self,
        x_api_client: Optional[XAPIClient] = None,
        classification_service: Optional[TweetClassificationService] = None,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize streaming API.

        Args:
            x_api_client: X API client for tweet collection
            classification_service: Classification service for real-time processing
            error_handler: Optional error handler instance
        """
        self.logger = logging.getLogger(__name__)
        self.error_handler = error_handler or ErrorHandler()
        self.active_streams: Dict[str, StreamSession] = {}

        # Real-time streaming
        self.tweet_streamer: Optional[TweetStreamer] = None
        self.real_time_streams: Dict[str, StreamConfig] = {}

        # Initialize real-time components if available
        if x_api_client and classification_service:
            self.tweet_streamer = TweetStreamer(
                x_api_client=x_api_client,
                classification_service=classification_service,
                error_handler=self.error_handler
            )

        # Streaming configuration
        self.default_interval = 60
        self.min_interval = 30
        self.max_interval = 300
        self.max_keywords = 50
        self.session_timeout = timedelta(hours=1)

    def start_stream(
        self,
        keywords: Optional[List[str]] = None,
        interval: Optional[int] = None
    ) -> Dict[str, Any]:
        """Start a new tweet streaming session.

        Args:
            keywords: List of disaster-related keywords to filter tweets
            interval: Polling interval in seconds

        Returns:
            Stream session information dictionary

        Raises:
            ValueError: If input validation fails
        """
        try:
            # Validate input parameters
            validated_keywords, validated_interval = self._validate_stream_params(
                keywords, interval
            )

            # Generate stream ID
            stream_id = str(uuid.uuid4())

            # Create stream session
            now = datetime.now()
            session = StreamSession(
                stream_id=stream_id,
                keywords=validated_keywords,
                interval=validated_interval,
                created_at=now,
                last_activity=now
            )

            # Store session
            self.active_streams[stream_id] = session

            self.logger.info(f"Started stream {stream_id} with {len(validated_keywords)} keywords")

            # Start real-time streaming if available
            real_stream_id = None
            if self.tweet_streamer:
                try:
                    stream_config = StreamConfig(
                        keywords=validated_keywords,
                        interval_seconds=validated_interval,
                        auto_classify=True,
                        include_features=True,
                        include_keywords=True
                    )
                    real_stream_id = self.tweet_streamer.start_stream(stream_config)
                    self.real_time_streams[stream_id] = stream_config
                    self.logger.info(f"Started real-time stream {real_stream_id}")
                except Exception as e:
                    self.logger.warning(f"Failed to start real-time stream: {e}")

            return {
                "status": "streaming_started",
                "stream_id": stream_id,
                "real_stream_id": real_stream_id,
                "keywords": validated_keywords,
                "interval": validated_interval,
                "created_at": now.isoformat(),
                "session_info": {
                    "max_keywords": self.max_keywords,
                    "interval_range": f"{self.min_interval}-{self.max_interval}s",
                    "session_timeout_hours": self.session_timeout.total_seconds() / 3600,
                    "real_time_enabled": real_stream_id is not None
                }
            }

        except Exception as e:
            self.logger.error(f"Stream start failed: {str(e)}")
            raise

    def stop_stream(self, stream_id: str) -> Dict[str, Any]:
        """Stop an active streaming session.

        Args:
            stream_id: Stream identifier to stop

        Returns:
            Stream stop result dictionary

        Raises:
            ValueError: If stream ID is invalid or stream not found
        """
        try:
            # Validate stream ID
            if not stream_id or not isinstance(stream_id, str):
                raise ValueError("Invalid stream ID format")

            # Find and remove stream
            if stream_id not in self.active_streams:
                raise ValueError(f"Stream {stream_id} not found")

            session = self.active_streams.pop(stream_id)
            session.status = "stopped"

            # Stop real-time stream if exists
            real_stream_stopped = False
            if stream_id in self.real_time_streams and self.tweet_streamer:
                try:
                    real_stream_stopped = self.tweet_streamer.stop_stream(stream_id)
                    self.real_time_streams.pop(stream_id, None)
                except Exception as e:
                    self.logger.error(f"Error stopping real-time stream: {e}")

            # Calculate duration
            duration = datetime.now() - session.created_at

            self.logger.info(f"Stopped stream {stream_id} after {duration.total_seconds():.1f}s")

            return {
                "status": "streaming_stopped",
                "stream_id": stream_id,
                "real_stream_stopped": real_stream_stopped,
                "stopped_at": datetime.now().isoformat(),
                "duration_seconds": duration.total_seconds(),
                "session_summary": {
                    "total_tweets_processed": session.total_tweets_processed,
                    "total_disaster_tweets": session.total_disaster_tweets,
                    "disaster_ratio": (
                        session.total_disaster_tweets / session.total_tweets_processed
                        if session.total_tweets_processed > 0 else 0
                    ),
                    "errors_count": len(session.errors),
                    "keywords_count": len(session.keywords)
                }
            }

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Stream stop failed: {str(e)}")
            raise RuntimeError(f"Failed to stop stream: {str(e)}")

    def get_stream_status(self, stream_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of streaming sessions.

        Args:
            stream_id: Specific stream ID, or None for all streams

        Returns:
            Stream status dictionary
        """
        try:
            if stream_id:
                # Get specific stream status
                if stream_id not in self.active_streams:
                    raise ValueError(f"Stream {stream_id} not found")

                session = self.active_streams[stream_id]
                return self._build_session_response(session)
            else:
                # Get all streams status
                return self._build_all_streams_response()

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Stream status check failed: {str(e)}")
            raise RuntimeError(f"Failed to get stream status: {str(e)}")

    def list_active_streams(self) -> Dict[str, Any]:
        """List all active streaming sessions.

        Returns:
            Dictionary with active streams information
        """
        try:
            # Clean up expired sessions
            self._cleanup_expired_sessions()

            active_streams_list = []
            for session in self.active_streams.values():
                if session.status == "active":
                    active_streams_list.append(self._build_session_summary(session))

            return {
                "active_streams_count": len(active_streams_list),
                "active_streams": active_streams_list,
                "max_concurrent_streams": 100,  # Configurable limit
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"List streams failed: {str(e)}")
            raise RuntimeError(f"Failed to list streams: {str(e)}")

    def update_stream_keywords(
        self,
        stream_id: str,
        keywords: List[str]
    ) -> Dict[str, Any]:
        """Update keywords for an active stream.

        Args:
            stream_id: Stream identifier
            keywords: New list of keywords

        Returns:
            Update result dictionary
        """
        try:
            # Validate stream ID
            if stream_id not in self.active_streams:
                raise ValueError(f"Stream {stream_id} not found")

            # Validate keywords
            if not isinstance(keywords, list):
                raise ValueError("Keywords must be a list")

            if len(keywords) > self.max_keywords:
                raise ValueError(f"Cannot exceed {self.max_keywords} keywords")

            if len(keywords) == 0:
                raise ValueError("Keywords list cannot be empty")

            # Update session
            session = self.active_streams[stream_id]
            session.keywords = keywords
            session.last_activity = datetime.now()

            self.logger.info(f"Updated stream {stream_id} with {len(keywords)} keywords")

            return {
                "status": "keywords_updated",
                "stream_id": stream_id,
                "new_keywords": keywords,
                "keywords_count": len(keywords),
                "updated_at": session.last_activity.isoformat()
            }

        except ValueError:
            raise
        except Exception as e:
            self.logger.error(f"Update stream keywords failed: {str(e)}")
            raise RuntimeError(f"Failed to update stream keywords: {str(e)}")

    def _validate_stream_params(
        self,
        keywords: Optional[List[str]],
        interval: Optional[int]
    ) -> tuple[List[str], int]:
        """Validate streaming parameters.

        Args:
            keywords: Keywords to validate
            interval: Interval to validate

        Returns:
            Tuple of validated keywords and interval

        Raises:
            ValueError: If validation fails
        """
        # Validate keywords
        if keywords is None:
            validated_keywords = [
                "earthquake", "flood", "hurricane", "wildfire", "disaster",
                "emergency", "evacuate", "rescue", "damage", "casualty"
            ]
        else:
            if not isinstance(keywords, list):
                raise ValueError("Keywords must be a list")

            if len(keywords) > self.max_keywords:
                raise ValueError(f"Cannot exceed {self.max_keywords} keywords")

            if len(keywords) == 0:
                raise ValueError("Keywords list cannot be empty")

            # Validate individual keywords
            for keyword in keywords:
                if not isinstance(keyword, str):
                    raise ValueError("All keywords must be strings")
                if len(keyword.strip()) == 0:
                    raise ValueError("Keywords cannot be empty strings")

            validated_keywords = [kw.strip().lower() for kw in keywords]

        # Validate interval
        if interval is None:
            validated_interval = self.default_interval
        else:
            if not isinstance(interval, int):
                raise ValueError("Interval must be an integer")

            if interval < self.min_interval or interval > self.max_interval:
                raise ValueError(f"Interval must be between {self.min_interval} and {self.max_interval} seconds")

            validated_interval = interval

        return validated_keywords, validated_interval

    def _cleanup_expired_sessions(self):
        """Clean up expired streaming sessions."""
        expired_streams = []
        current_time = datetime.now()

        for stream_id, session in self.active_streams.items():
            if current_time - session.last_activity > self.session_timeout:
                expired_streams.append(stream_id)

        for stream_id in expired_streams:
            session = self.active_streams.pop(stream_id)
            session.status = "expired"
            self.logger.info(f"Expired stream {stream_id} due to inactivity")

    def _build_session_response(self, session: StreamSession) -> Dict[str, Any]:
        """Build response for a single session."""
        return {
            "stream_id": session.stream_id,
            "status": session.status,
            "keywords": session.keywords,
            "interval": session.interval,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "uptime_seconds": (datetime.now() - session.created_at).total_seconds(),
            "statistics": {
                "total_tweets_processed": session.total_tweets_processed,
                "total_disaster_tweets": session.total_disaster_tweets,
                "disaster_ratio": (
                    session.total_disaster_tweets / session.total_tweets_processed
                    if session.total_tweets_processed > 0 else 0
                ),
                "errors_count": len(session.errors)
            }
        }

    def _build_all_streams_response(self) -> Dict[str, Any]:
        """Build response for all streams."""
        self._cleanup_expired_sessions()

        streams_response = {}
        for stream_id, session in self.active_streams.items():
            streams_response[stream_id] = self._build_session_response(session)

        return {
            "total_streams": len(self.active_streams),
            "streams": streams_response,
            "timestamp": datetime.now().isoformat()
        }

    def _build_session_summary(self, session: StreamSession) -> Dict[str, Any]:
        """Build summary for a session."""
        return {
            "stream_id": session.stream_id,
            "keywords_count": len(session.keywords),
            "interval": session.interval,
            "uptime_seconds": (datetime.now() - session.created_at).total_seconds(),
            "total_tweets_processed": session.total_tweets_processed,
            "status": session.status
        }

    def get_streaming_statistics(self) -> Dict[str, Any]:
        """Get streaming service statistics.

        Returns:
            Streaming statistics dictionary
        """
        self._cleanup_expired_sessions()

        total_tweets = sum(session.total_tweets_processed for session in self.active_streams.values())
        total_disaster = sum(session.total_disaster_tweets for session in self.active_streams.values())
        total_errors = sum(len(session.errors) for session in self.active_streams.values())

        return {
            "active_streams_count": len(self.active_streams),
            "total_tweets_processed": total_tweets,
            "total_disaster_tweets": total_disaster,
            "overall_disaster_ratio": total_disaster / total_tweets if total_tweets > 0 else 0,
            "total_errors": total_errors,
            "average_tweets_per_stream": total_tweets / len(self.active_streams) if len(self.active_streams) > 0 else 0,
            "configuration": {
                "default_interval": self.default_interval,
                "min_interval": self.min_interval,
                "max_interval": self.max_interval,
                "max_keywords": self.max_keywords,
                "session_timeout_hours": self.session_timeout.total_seconds() / 3600
            },
            "timestamp": datetime.now().isoformat()
        }

    def get_real_time_stream_status(self, stream_id: str) -> Optional[Dict]:
        """Get status of real-time streaming session.

        Args:
            stream_id: Stream ID

        Returns:
            Real-time stream status or None if not found
        """
        if not self.tweet_streamer or stream_id not in self.real_time_streams:
            return None

        return self.tweet_streamer.get_stream_status(stream_id)

    def get_real_time_streaming_stats(self) -> Dict:
        """Get real-time streaming statistics.

        Returns:
            Real-time streaming statistics
        """
        if not self.tweet_streamer:
            return {
                "real_time_enabled": False,
                "message": "Real-time streaming not available"
            }

        base_stats = self.tweet_streamer.get_streaming_stats()
        return {
            **base_stats,
            "real_time_enabled": True,
            "active_real_time_streams": len(self.real_time_streams)
        }