"""Real-time tweet streaming service with configurable polling."""

import asyncio
import threading
import time
import logging
from typing import List, Dict, Optional, Callable, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
from enum import Enum

from ..api.x_api_client import XAPIClient
from ..inference.predictor import TweetClassificationService
from ..models.tweet import Tweet
from ..models.classification_result import ClassificationResult
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext, RecoveryAction


class StreamStatus(Enum):
    """Stream status enumeration."""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class StreamConfig:
    """Configuration for tweet streaming."""
    keywords: List[str]
    interval_seconds: int = 60
    max_tweets_per_batch: int = 10
    max_tweets_total: Optional[int] = None
    auto_classify: bool = True
    include_features: bool = True
    include_keywords: bool = True
    stream_duration_minutes: Optional[int] = None


@dataclass
class StreamMetrics:
    """Metrics for streaming session."""
    start_time: datetime
    tweets_collected: int = 0
    tweets_classified: int = 0
    disaster_tweets_found: int = 0
    api_calls_made: int = 0
    errors_encountered: int = 0
    last_tweet_time: Optional[datetime] = None
    last_classification_time: Optional[datetime] = None
    processing_times: List[float] = field(default_factory=list)


class TweetStreamer:
    """Real-time tweet streaming service."""

    def __init__(
        self,
        x_api_client: XAPIClient,
        classification_service: TweetClassificationService,
        max_concurrent_streams: int = 5,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize tweet streamer.

        Args:
            x_api_client: X API v2 client
            classification_service: Tweet classification service
            max_concurrent_streams: Maximum concurrent streams
            error_handler: Optional error handler instance
        """
        self.x_api_client = x_api_client
        self.classification_service = classification_service
        self.max_concurrent_streams = max_concurrent_streams

        # Initialize error handler
        self.error_handler = error_handler or ErrorHandler()

        # Active streams
        self.active_streams: Dict[str, Dict] = {}
        self.stream_threads: Dict[str, threading.Thread] = {}
        self.stream_metrics: Dict[str, StreamMetrics] = {}

        # Control flags
        self.running = False
        self.shutdown_event = threading.Event()

        self.logger = logging.getLogger(__name__)

    @self.error_handler.with_error_handling(
    component="TweetStreamer",
    operation="start_stream",
    category=ErrorCategory.STREAMING,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.RETRY
)
def start_stream(
        self,
        config: StreamConfig,
        callback: Optional[Callable[[List[Tweet], List[ClassificationResult]], None]] = None
    ) -> str:
        """Start a new tweet streaming session.

        Args:
            config: Stream configuration
            callback: Optional callback function for processing results

        Returns:
            Stream ID

        Raises:
            RuntimeError: If cannot start stream
        """
        if len(self.active_streams) >= self.max_concurrent_streams:
            raise RuntimeError(f"Maximum {self.max_concurrent_streams} concurrent streams reached")

        # Generate stream ID
        stream_id = str(uuid.uuid4())

        # Initialize stream data
        self.active_streams[stream_id] = {
            'config': config,
            'status': StreamStatus.RUNNING,
            'callback': callback,
            'start_time': datetime.now(),
            'stop_event': threading.Event()
        }

        # Initialize metrics
        self.stream_metrics[stream_id] = StreamMetrics(start_time=datetime.now())

        # Start streaming thread
        thread = threading.Thread(
            target=self._stream_worker,
            args=(stream_id,),
            daemon=True
        )
        thread.start()
        self.stream_threads[stream_id] = thread

        self.logger.info(f"Started stream {stream_id} with {len(config.keywords)} keywords")
        return stream_id

    def stop_stream(self, stream_id: str) -> bool:
        """Stop a streaming session.

        Args:
            stream_id: Stream ID to stop

        Returns:
            True if stopped successfully, False otherwise
        """
        if stream_id not in self.active_streams:
            return False

        try:
            # Signal thread to stop
            stream_data = self.active_streams[stream_id]
            stream_data['stop_event'].set()
            stream_data['status'] = StreamStatus.STOPPED

            # Wait for thread to finish (with timeout)
            if stream_id in self.stream_threads:
                thread = self.stream_threads[stream_id]
                thread.join(timeout=10.0)
                if thread.is_alive():
                    self.logger.warning(f"Stream {stream_id} thread did not stop gracefully")

            # Clean up
            self.active_streams.pop(stream_id, None)
            self.stream_threads.pop(stream_id, None)

            duration = datetime.now() - self.stream_metrics[stream_id].start_time
            self.logger.info(f"Stopped stream {stream_id} after {duration.total_seconds():.1f}s")

            return True

        except Exception as e:
            self.logger.error(f"Error stopping stream {stream_id}: {e}")
            return False

    def get_stream_status(self, stream_id: str) -> Optional[Dict]:
        """Get status of a streaming session.

        Args:
            stream_id: Stream ID

        Returns:
            Stream status dictionary or None if not found
        """
        if stream_id not in self.active_streams:
            return None

        stream_data = self.active_streams[stream_id]
        metrics = self.stream_metrics.get(stream_id)

        return {
            'stream_id': stream_id,
            'status': stream_data['status'].value,
            'keywords': stream_data['config'].keywords,
            'interval': stream_data['config'].interval_seconds,
            'start_time': stream_data['start_time'].isoformat(),
            'uptime_seconds': (datetime.now() - stream_data['start_time']).total_seconds(),
            'metrics': self._build_metrics_dict(metrics) if metrics else {}
        }

    def list_active_streams(self) -> List[Dict]:
        """List all active streaming sessions.

        Returns:
            List of stream status dictionaries
        """
        streams = []
        for stream_id in list(self.active_streams.keys()):
            status = self.get_stream_status(stream_id)
            if status:
                streams.append(status)
        return streams

    def get_streaming_stats(self) -> Dict:
        """Get overall streaming statistics.

        Returns:
            Streaming statistics dictionary
        """
        total_tweets = sum(m.tweets_collected for m in self.stream_metrics.values())
        total_classified = sum(m.tweets_classified for m in self.stream_metrics.values())
        total_disaster = sum(m.disaster_tweets_found for m in self.stream_metrics.values())
        total_api_calls = sum(m.api_calls_made for m in self.stream_metrics.values())

        # Calculate average processing time
        all_processing_times = []
        for metrics in self.stream_metrics.values():
            all_processing_times.extend(metrics.processing_times)
        avg_processing_time = (
            sum(all_processing_times) / len(all_processing_times)
            if all_processing_times else 0.0
        )

        return {
            'active_streams_count': len(self.active_streams),
            'total_tweets_collected': total_tweets,
            'total_tweets_classified': total_classified,
            'total_disaster_tweets': total_disaster,
            'overall_disaster_ratio': total_disaster / total_classified if total_classified > 0 else 0,
            'total_api_calls': total_api_calls,
            'average_processing_time': avg_processing_time,
            'max_concurrent_streams': self.max_concurrent_streams,
            'timestamp': datetime.now().isoformat()
        }

    def stop_all_streams(self) -> int:
        """Stop all active streaming sessions.

        Returns:
            Number of streams stopped
        """
        stream_ids = list(self.active_streams.keys())
        stopped_count = 0

        for stream_id in stream_ids:
            if self.stop_stream(stream_id):
                stopped_count += 1

        return stopped_count

    def _stream_worker(self, stream_id: str):
        """Worker thread for streaming tweets.

        Args:
            stream_id: Stream ID to process
        """
        stream_data = self.active_streams.get(stream_id)
        if not stream_data:
            return

        config = stream_data['config']
        stop_event = stream_data['stop_event']
        metrics = self.stream_metrics.get(stream_id)

        self.logger.info(f"Stream worker started for {stream_id}")

        try:
            while not stop_event.is_set() and not self.shutdown_event.is_set():
                batch_start_time = time.time()

                try:
                    # Check if stream duration exceeded
                    if config.stream_duration_minutes:
                        elapsed = (datetime.now() - metrics.start_time).total_seconds() / 60
                        if elapsed >= config.stream_duration_minutes:
                            self.logger.info(f"Stream {stream_id} duration exceeded, stopping")
                            break

                    # Collect tweets
                    tweets = self._collect_tweets_batch(config, metrics)

                    if tweets:
                        # Classify tweets if enabled
                        results = []
                        if config.auto_classify and self.classification_service.is_ready():
                            results = self._classify_tweets_batch(tweets, config, metrics)

                        # Call callback if provided
                        if stream_data['callback']:
                            try:
                                stream_data['callback'](tweets, results)
                            except Exception as e:
                                self.logger.error(f"Callback error for stream {stream_id}: {e}")

                        # Update metrics
                        metrics.last_tweet_time = datetime.now()
                        if results:
                            metrics.last_classification_time = datetime.now()

                    # Check max tweets limit
                    if config.max_tweets_total and metrics.tweets_collected >= config.max_tweets_total:
                        self.logger.info(f"Stream {stream_id} reached max tweets limit, stopping")
                        break

                    # Calculate processing time
                    processing_time = time.time() - batch_start_time
                    metrics.processing_times.append(processing_time)

                    # Sleep for remaining interval
                    sleep_time = max(0, config.interval_seconds - processing_time)
                    stop_event.wait(sleep_time)

                except Exception as e:
                    metrics.errors_encountered += 1
                    self.logger.error(f"Error in stream {stream_id} worker: {e}")
                    # Sleep before retry
                    stop_event.wait(5.0)

        except Exception as e:
            stream_data['status'] = StreamStatus.ERROR
            self.logger.error(f"Fatal error in stream {stream_id}: {e}")

        finally:
            # Update final status
            if stream_id in self.active_streams:
                stream_data = self.active_streams[stream_id]
                stream_data['status'] = StreamStatus.STOPPED

            self.logger.info(f"Stream worker stopped for {stream_id}")

    @self.error_handler.with_error_handling(
    component="TweetStreamer",
    operation="collect_tweets_batch",
    category=ErrorCategory.API,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.CONTINUE
)
def _collect_tweets_batch(self, config: StreamConfig, metrics: StreamMetrics) -> List[Tweet]:
        """Collect a batch of tweets from X API.

        Args:
            config: Stream configuration
            metrics: Stream metrics to update

        Returns:
            List of collected tweets
        """
        # Build search query from keywords
        query = " OR ".join(config.keywords)
        if len(config.keywords) > 1:
            query = f"({query})"

        # Search for tweets
        tweets = self.x_api_client.search_tweets(
            query=query,
            max_results=config.max_tweets_per_batch,
            start_time=datetime.now() - timedelta(minutes=5)  # Last 5 minutes
        )

        if tweets:
            metrics.tweets_collected += len(tweets)
            metrics.api_calls_made += 1
            self.logger.debug(f"Collected {len(tweets)} tweets for stream")

        return tweets

    @self.error_handler.with_error_handling(
    component="TweetStreamer",
    operation="classify_tweets_batch",
    category=ErrorCategory.MODEL,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.CONTINUE
)
def _classify_tweets_batch(
        self,
        tweets: List[Tweet],
        config: StreamConfig,
        metrics: StreamMetrics
    ) -> List[ClassificationResult]:
        """Classify a batch of tweets.

        Args:
            tweets: List of tweets to classify
            config: Stream configuration
            metrics: Stream metrics to update

        Returns:
            List of classification results
        """
        # Convert tweets to text list
        tweet_texts = [tweet.text for tweet in tweets]

        # Classify batch
        results = self.classification_service.classify_batch(
            tweet_texts,
            include_features=config.include_features,
            include_keywords=config.include_keywords
        )

        # Update metrics
        metrics.tweets_classified += len(results)
        disaster_count = sum(1 for r in results if r.is_disaster())
        metrics.disaster_tweets_found += disaster_count

        # Associate results with tweets
        for tweet, result in zip(tweets, results):
            result.tweet_id = tweet.id

        self.logger.debug(f"Classified {len(results)} tweets, {disaster_count} disasters")

        return results

    def _build_metrics_dict(self, metrics: StreamMetrics) -> Dict:
        """Build metrics dictionary.

        Args:
            metrics: Stream metrics object

        Returns:
            Metrics dictionary
        """
        return {
            'tweets_collected': metrics.tweets_collected,
            'tweets_classified': metrics.tweets_classified,
            'disaster_tweets_found': metrics.disaster_tweets_found,
            'api_calls_made': metrics.api_calls_made,
            'errors_encountered': metrics.errors_encountered,
            'last_tweet_time': metrics.last_tweet_time.isoformat() if metrics.last_tweet_time else None,
            'last_classification_time': metrics.last_classification_time.isoformat() if metrics.last_classification_time else None,
            'average_processing_time': (
                sum(metrics.processing_times) / len(metrics.processing_times)
                if metrics.processing_times else 0.0
            ),
            'uptime_seconds': (datetime.now() - metrics.start_time).total_seconds()
        }

    def cleanup_expired_streams(self, timeout_minutes: int = 60):
        """Clean up streams that have been inactive.

        Args:
            timeout_minutes: Inactivity timeout in minutes
        """
        timeout = timedelta(minutes=timeout_minutes)
        current_time = datetime.now()

        expired_streams = []
        for stream_id, stream_data in self.active_streams.items():
            last_activity = stream_data.get('last_activity', stream_data['start_time'])
            if current_time - last_activity > timeout:
                expired_streams.append(stream_id)

        for stream_id in expired_streams:
            self.logger.info(f"Cleaning up expired stream {stream_id}")
            self.stop_stream(stream_id)

    def __del__(self):
        """Cleanup on destruction."""
        self.stop_all_streams()
        self.shutdown_event.set()