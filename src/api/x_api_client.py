"""X API v2 integration with rate limiting and exponential backoff."""

import time
import logging
import asyncio
from typing import List, Dict, Optional, AsyncGenerator, Generator, Callable
from datetime import datetime, timedelta
import tweepy
from tweepy.errors import TweepyException, TooManyRequests, Forbidden, TwitterServerError
from dataclasses import dataclass
from enum import Enum

from ..models.tweet import Tweet
from ..utils.error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, ErrorContext, RecoveryAction
from ..utils.cache import cached_response, cache


class RateLimitStatus(Enum):
    """Rate limit status enumeration."""
    OK = "ok"
    NEAR_LIMIT = "near_limit"
    EXCEEDED = "exceeded"
    COOLDOWN = "cooldown"


@dataclass
class RateLimitInfo:
    """Rate limit information."""
    limit: int
    remaining: int
    reset_time: datetime
    window_seconds: int
    status: RateLimitStatus

    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage."""
        return ((self.limit - self.remaining) / self.limit) * 100

    @property
    def time_until_reset(self) -> timedelta:
        """Get time until rate limit resets."""
        return max(timedelta(0), self.reset_time - datetime.now())


class XAPIRateLimiter:
    """Rate limiting and backoff manager for X API v2."""

    def __init__(
        self,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 300.0,
        safety_threshold: float = 0.9,  # 90% usage triggers safety mode
        cooldown_multiplier: float = 1.5
    ):
        """Initialize rate limiter.

        Args:
            max_retries: Maximum number of retry attempts
            initial_backoff: Initial backoff time in seconds
            max_backoff: Maximum backoff time in seconds
            safety_threshold: Usage percentage to trigger safety mode
            cooldown_multiplier: Multiplier for cooldown periods
        """
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.safety_threshold = safety_threshold
        self.cooldown_multiplier = cooldown_multiplier

        # Rate limit tracking for different endpoints
        self.rate_limits = {
            'tweets_search': None,
            'tweets_lookup': None,
            'users_lookup': None
        }

        # Backoff state
        self.current_backoff = 0.0
        self.last_request_time = {}
        self.request_count = {}

        self.logger = logging.getLogger(__name__)

    def update_rate_limit(self, endpoint: str, headers: Dict):
        """Update rate limit information from response headers.

        Args:
            endpoint: API endpoint name
            headers: Response headers containing rate limit info
        """
        try:
            limit = int(headers.get('x-rate-limit-limit', 0))
            remaining = int(headers.get('x-rate-limit-remaining', 0))
            reset_timestamp = int(headers.get('x-rate-limit-reset', 0))

            if limit > 0:
                reset_time = datetime.fromtimestamp(reset_timestamp)
                window_seconds = 15 * 60  # X API uses 15-minute windows

                # Determine status
                usage_percentage = ((limit - remaining) / limit) * 100
                if usage_percentage >= 100:
                    status = RateLimitStatus.EXCEEDED
                elif usage_percentage >= self.safety_threshold * 100:
                    status = RateLimitStatus.NEAR_LIMIT
                else:
                    status = RateLimitStatus.OK

                self.rate_limits[endpoint] = RateLimitInfo(
                    limit=limit,
                    remaining=remaining,
                    reset_time=reset_time,
                    window_seconds=window_seconds,
                    status=status
                )

                self.logger.info(f"Updated rate limit for {endpoint}: {remaining}/{limit} ({usage_percentage:.1f}%)")

        except (ValueError, KeyError) as e:
            self.logger.warning(f"Failed to parse rate limit headers for {endpoint}: {e}")

    def get_rate_limit_status(self, endpoint: str) -> Optional[RateLimitInfo]:
        """Get current rate limit status for an endpoint.

        Args:
            endpoint: API endpoint name

        Returns:
            Rate limit information or None if not available
        """
        return self.rate_limits.get(endpoint)

    def can_make_request(self, endpoint: str) -> bool:
        """Check if a request can be made to an endpoint.

        Args:
            endpoint: API endpoint name

        Returns:
            True if request can be made, False otherwise
        """
        rate_limit = self.get_rate_limit_status(endpoint)
        if not rate_limit:
            return True  # No rate limit info available

        if rate_limit.status == RateLimitStatus.EXCEEDED:
            return False

        if rate_limit.status == RateLimitStatus.NEAR_LIMIT:
            # Apply additional safety check
            if rate_limit.remaining <= 5:
                return False

        return True

    def get_wait_time(self, endpoint: str) -> float:
        """Get required wait time before making a request.

        Args:
            endpoint: API endpoint name

        Returns:
            Wait time in seconds
        """
        rate_limit = self.get_rate_limit_status(endpoint)
        if not rate_limit:
            return 0.0

        # If rate limit is exceeded, wait until reset
        if rate_limit.status == RateLimitStatus.EXCEEDED:
            return rate_limit.time_until_reset.total_seconds()

        # If near limit, apply backoff
        if rate_limit.status == RateLimitStatus.NEAR_LIMIT:
            return self.current_backoff or self.initial_backoff

        # Apply minimum delay between requests
        last_request = self.last_request_time.get(endpoint)
        if last_request:
            time_since_last = (datetime.now() - last_request).total_seconds()
            if time_since_last < 1.0:  # Minimum 1 second between requests
                return 1.0 - time_since_last

        return 0.0

    def record_request(self, endpoint: str):
        """Record that a request was made to an endpoint.

        Args:
            endpoint: API endpoint name
        """
        self.last_request_time[endpoint] = datetime.now()
        self.request_count[endpoint] = self.request_count.get(endpoint, 0) + 1

    def handle_rate_limit_error(self, endpoint: str) -> float:
        """Handle rate limit error and calculate backoff.

        Args:
            endpoint: API endpoint that hit rate limit

        Returns:
            Backoff time in seconds
        """
        rate_limit = self.get_rate_limit_status(endpoint)
        if rate_limit and rate_limit.status == RateLimitStatus.EXCEEDED:
            # Use precise wait time based on reset time
            wait_time = rate_limit.time_until_reset.total_seconds()
        else:
            # Use exponential backoff
            if self.current_backoff == 0:
                self.current_backoff = self.initial_backoff
            else:
                self.current_backoff = min(
                    self.current_backoff * self.cooldown_multiplier,
                    self.max_backoff
                )
            wait_time = self.current_backoff

        self.logger.warning(f"Rate limit hit for {endpoint}, backing off for {wait_time:.1f}s")
        return wait_time

    def reset_backoff(self):
        """Reset backoff state after successful request."""
        self.current_backoff = 0.0

    def get_stats(self) -> Dict:
        """Get rate limiter statistics.

        Returns:
            Rate limiter statistics
        """
        return {
            'current_backoff': self.current_backoff,
            'request_counts': self.request_count.copy(),
            'rate_limits': {
                endpoint: {
                    'status': rl.status.value,
                    'usage_percentage': rl.usage_percentage,
                    'remaining': rl.remaining,
                    'time_until_reset': rl.time_until_reset.total_seconds()
                }
                for endpoint, rl in self.rate_limits.items()
                if rl
            }
        }


class XAPIClient:
    """X API v2 client with rate limiting and error handling."""

    def __init__(
        self,
        bearer_token: str,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        access_token: Optional[str] = None,
        access_token_secret: Optional[str] = None,
        max_retries: int = 5,
        error_handler: Optional[ErrorHandler] = None
    ):
        """Initialize X API client.

        Args:
            bearer_token: Bearer token for authentication
            api_key: API key (for additional authentication)
            api_secret: API secret
            access_token: Access token
            access_token_secret: Access token secret
            max_retries: Maximum number of retry attempts
            error_handler: Optional error handler instance
        """
        self.bearer_token = bearer_token
        self.max_retries = max_retries

        # Initialize error handler
        self.error_handler = error_handler or ErrorHandler()

        # Initialize rate limiter
        self.rate_limiter = XAPIRateLimiter()

        # Initialize tweepy client
        self.client = tweepy.Client(
            bearer_token=bearer_token,
            consumer_key=api_key,
            consumer_secret=api_secret,
            access_token=access_token,
            access_token_secret=access_token_secret,
            wait_on_rate_limit=False  # We handle rate limiting manually
        )

        self.logger = logging.getLogger(__name__)
        self.is_authorized = self._test_authorization()

    def _test_authorization(self) -> bool:
        """Test if API credentials are valid.

        Returns:
            True if authorized, False otherwise
        """
        try:
            # Try to make a simple API call
            self.client.get_me()
            return True
        except Exception as e:
            self.error_handler.handle_error(
                error=e,
                category=ErrorCategory.AUTHENTICATION,
                severity=ErrorSeverity.HIGH,
                context=ErrorContext(
                    component="XAPIClient",
                    operation="authorization_test",
                    message="Failed to authorize with X API"
                ),
                recovery_action=RecoveryAction.RETRY
            )
            self.logger.error(f"Authorization failed: {e}")
            return False

    def _make_request_with_retry(self, endpoint: str, func, *args, **kwargs):
        """Make API request with retry logic and rate limiting.

        Args:
            endpoint: API endpoint name
            func: API function to call
            args: Function arguments
            kwargs: Function keyword arguments

        Returns:
            API response

        Raises:
            TweepyException: If all retries exhausted
        """
        last_error = None

        for attempt in range(self.max_retries + 1):
            # Check rate limits
            if not self.rate_limiter.can_make_request(endpoint):
                wait_time = self.rate_limiter.get_wait_time(endpoint)
                if wait_time > 0:
                    self.logger.info(f"Rate limited, waiting {wait_time:.1f}s for {endpoint}")
                    time.sleep(wait_time)

            try:
                # Make the request
                response = func(*args, **kwargs)

                # Update rate limits from response headers
                if hasattr(response, 'response') and hasattr(response.response, 'headers'):
                    self.rate_limiter.update_rate_limit(endpoint, response.response.headers)

                # Record successful request
                self.rate_limiter.record_request(endpoint)
                self.rate_limiter.reset_backoff()

                return response

            except TooManyRequests as e:
                last_error = e
                wait_time = self.rate_limiter.handle_rate_limit_error(endpoint)
                if attempt < self.max_retries:
                    self.logger.warning(f"Rate limit exceeded, retry {attempt + 1}/{self.max_retries}")
                    time.sleep(wait_time)
                else:
                    self.logger.error(f"Rate limit retries exhausted for {endpoint}")
                    raise

            except (Forbidden, TwitterServerError) as e:
                last_error = e
                if attempt < self.max_retries:
                    backoff = (2 ** attempt) * 1.0  # Exponential backoff
                    self.logger.warning(f"API error, retry {attempt + 1}/{self.max_retries} in {backoff}s")
                    time.sleep(backoff)
                else:
                    raise

            except TweepyException as e:
                last_error = e
                self.logger.error(f"API error for {endpoint}: {e}")
                raise

        # If we get here, all retries failed
        raise last_error if last_error else TweepyException("Unknown API error")

    @self.error_handler.with_error_handling(
    component="XAPIClient",
    operation="search_tweets",
    category=ErrorCategory.API,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.RETRY
)
@cached_response(ttl=180)  # Cache for 3 minutes
def search_tweets(
        self,
        query: str,
        max_results: int = 10,
        tweet_fields: Optional[List[str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Tweet]:
        """Search for tweets using X API v2.

        Args:
            query: Search query
            max_results: Maximum number of results (10-100)
            tweet_fields: Additional tweet fields to retrieve
            start_time: Start time for search
            end_time: End time for search

        Returns:
            List of Tweet objects
        """
        if not self.is_authorized:
            raise RuntimeError("X API client not authorized")

        # Set default tweet fields
        if tweet_fields is None:
            tweet_fields = [
                'created_at', 'author_id', 'public_metrics',
                'context_annotations', 'entities', 'lang'
            ]

        response = self._make_request_with_retry(
            'tweets_search',
            self.client.search_recent_tweets,
            query=query,
            max_results=max_results,
            tweet_fields=tweet_fields,
            start_time=start_time,
            end_time=end_time
        )

        if response.data:
            return [
                Tweet.from_api_response(tweet, response.includes or {})
                for tweet in response.data
            ]
        else:
            return []

    @self.error_handler.with_error_handling(
    component="XAPIClient",
    operation="get_tweets_by_ids",
    category=ErrorCategory.API,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.RETRY
)
@cached_response(ttl=600)  # Cache for 10 minutes
def get_tweets_by_ids(
        self,
        tweet_ids: List[str],
        tweet_fields: Optional[List[str]] = None
    ) -> List[Tweet]:
        """Get tweets by their IDs.

        Args:
            tweet_ids: List of tweet IDs
            tweet_fields: Additional tweet fields to retrieve

        Returns:
            List of Tweet objects
        """
        if not self.is_authorized:
            raise RuntimeError("X API client not authorized")

        if not tweet_ids:
            return []

        # Set default tweet fields
        if tweet_fields is None:
            tweet_fields = [
                'created_at', 'author_id', 'public_metrics',
                'context_annotations', 'entities', 'lang'
            ]

        # X API v2 allows up to 100 IDs per request
        all_tweets = []
        for i in range(0, len(tweet_ids), 100):
            batch_ids = tweet_ids[i:i + 100]

            response = self._make_request_with_retry(
                'tweets_lookup',
                self.client.get_tweets,
                ids=batch_ids,
                tweet_fields=tweet_fields
            )

            if response.data:
                all_tweets.extend([
                    Tweet.from_api_response(tweet, response.includes or {})
                    for tweet in response.data
                ])

        return all_tweets

    @self.error_handler.with_error_handling(
    component="XAPIClient",
    operation="stream_search_tweets",
    category=ErrorCategory.API,
    severity=ErrorSeverity.MEDIUM,
    recovery_action=RecoveryAction.CONTINUE
)
def stream_search_tweets(
        self,
        query: str,
        max_results_per_batch: int = 10,
        interval_seconds: int = 60,
        max_batches: Optional[int] = None
    ) -> Generator[List[Tweet], None, None]:
        """Stream tweets with periodic search.

        Args:
            query: Search query
            max_results_per_batch: Maximum results per search batch
            interval_seconds: Interval between searches
            max_batches: Maximum number of batches (None for unlimited)

        Yields:
            List of Tweet objects for each batch
        """
        batch_count = 0

        while max_batches is None or batch_count < max_batches:
            tweets = self.search_tweets(query, max_results_per_batch)

            if tweets:
                yield tweets

            batch_count += 1

            # Wait before next search
            if max_batches is None or batch_count < max_batches:
                time.sleep(interval_seconds)

    def get_rate_limit_stats(self) -> Dict:
        """Get rate limit statistics.

        Returns:
            Rate limit statistics
        """
        return {
            'is_authorized': self.is_authorized,
            'rate_limiter': self.rate_limiter.get_stats(),
            'cache_stats': cache.get_stats()
        }

    def is_ready(self) -> bool:
        """Check if client is ready for requests.

        Returns:
            True if ready, False otherwise
        """
        return self.is_authorized

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        # Cleanup if needed
        pass