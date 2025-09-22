"""Batch processing optimization for disaster tweet classification."""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Dict, Union, Optional, Callable, Tuple, Iterator
from dataclasses import dataclass, field
from datetime import datetime
import logging
import queue
from enum import Enum

from .predictor import TweetClassificationService
from ..models.classification_result import ClassificationResult


class BatchProcessingStrategy(Enum):
    """Batch processing strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ASYNC = "async"
    HYBRID = "hybrid"


@dataclass
class BatchConfig:
    """Configuration for batch processing."""
    max_batch_size: int = 64
    timeout_seconds: float = 5.0
    max_workers: int = 4
    strategy: BatchProcessingStrategy = BatchProcessingStrategy.HYBRID
    enable_chunking: bool = True
    chunk_size: int = 1000
    memory_limit_mb: int = 1024
    progress_tracking: bool = True


@dataclass
class BatchResult:
    """Result of batch processing operation."""
    results: List[ClassificationResult] = field(default_factory=list)
    errors: List[Dict] = field(default_factory=list)
    processing_time: float = 0.0
    batch_size: int = 0
    success_rate: float = 0.0
    metadata: Dict = field(default_factory=dict)


class BatchProcessor:
    """Optimized batch processing for tweet classification."""

    def __init__(
        self,
        classification_service: TweetClassificationService,
        config: Optional[BatchConfig] = None
    ):
        """Initialize batch processor.

        Args:
            classification_service: Tweet classification service
            config: Batch processing configuration
        """
        self.service = classification_service
        self.config = config or BatchConfig()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.stats = {
            'total_batches_processed': 0,
            'total_tweets_processed': 0,
            'average_batch_time': 0.0,
            'average_tweets_per_second': 0.0,
            'success_rate': 1.0,
            'memory_usage_mb': 0.0
        }

        # Resource management
        self._executor = None
        self._memory_monitor_thread = None
        self._should_monitor = False

    def process_batch(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool = True,
        include_keywords: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> BatchResult:
        """Process a batch of tweets with optimization.

        Args:
            tweets: List of tweets to process
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            progress_callback: Optional progress callback function

        Returns:
            BatchResult object
        """
        start_time = time.time()

        try:
            # Validate input
            if not tweets:
                return BatchResult(
                    processing_time=time.time() - start_time,
                    batch_size=0,
                    success_rate=1.0
                )

            # Apply batch size limit
            batch_tweets = tweets[:self.config.max_batch_size]

            # Choose processing strategy
            if self.config.strategy == BatchProcessingStrategy.SEQUENTIAL:
                result = self._process_sequential(
                    batch_tweets, include_features, include_keywords, progress_callback
                )
            elif self.config.strategy == BatchProcessingStrategy.PARALLEL:
                result = self._process_parallel(
                    batch_tweets, include_features, include_keywords, progress_callback
                )
            elif self.config.strategy == BatchProcessingStrategy.ASYNC:
                result = self._process_async(
                    batch_tweets, include_features, include_keywords, progress_callback
                )
            else:  # HYBRID
                result = self._process_hybrid(
                    batch_tweets, include_features, include_keywords, progress_callback
                )

            # Update result metadata
            result.processing_time = time.time() - start_time
            result.batch_size = len(batch_tweets)
            result.success_rate = len(result.results) / len(batch_tweets) if batch_tweets else 1.0

            # Update statistics
            self._update_stats(result)

            return result

        except Exception as e:
            self.logger.error(f"Batch processing failed: {str(e)}")
            return BatchResult(
                errors=[{"error": str(e), "timestamp": datetime.now().isoformat()}],
                processing_time=time.time() - start_time,
                batch_size=len(tweets),
                success_rate=0.0
            )

    def process_large_dataset(
        self,
        tweets: List[Union[str, Dict]],
        chunk_size: Optional[int] = None,
        include_features: bool = True,
        include_keywords: bool = True,
        progress_callback: Optional[Callable] = None
    ) -> Iterator[BatchResult]:
        """Process large dataset in chunks.

        Args:
            tweets: List of tweets to process
            chunk_size: Size of each chunk
            include_features: Whether to include feature analysis
            include_keywords: Whether to include keyword analysis
            progress_callback: Optional progress callback function

        Yields:
            BatchResult for each chunk
        """
        chunk_size = chunk_size or self.config.chunk_size

        if not self.config.enable_chunking:
            yield self.process_batch(tweets, include_features, include_keywords, progress_callback)
            return

        total_chunks = (len(tweets) + chunk_size - 1) // chunk_size
        processed_chunks = 0

        for i in range(0, len(tweets), chunk_size):
            chunk = tweets[i:i + chunk_size]
            processed_chunks += 1

            if progress_callback:
                progress_callback(processed_chunks, total_chunks, len(chunk))

            yield self.process_batch(chunk, include_features, include_keywords)

    def _process_sequential(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool,
        include_keywords: bool,
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """Process tweets sequentially."""
        results = []
        errors = []

        for i, tweet in enumerate(tweets):
            try:
                result = self.service.classify_tweet(tweet, include_features, include_keywords)
                results.append(result)

                if progress_callback:
                    progress_callback(i + 1, len(tweets), None)

            except Exception as e:
                error_info = {
                    "tweet_index": i,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                errors.append(error_info)
                self.logger.warning(f"Failed to process tweet {i}: {str(e)}")

        return BatchResult(results=results, errors=errors)

    def _process_parallel(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool,
        include_keywords: bool,
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """Process tweets in parallel using thread pool."""
        results = []
        errors = []

        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)

        def process_single_tweet(args):
            index, tweet = args
            try:
                result = self.service.classify_tweet(tweet, include_features, include_keywords)
                return index, result, None
            except Exception as e:
                return index, None, {"error": str(e), "tweet_index": index}

        # Create tasks
        tasks = [(i, tweet) for i, tweet in enumerate(tweets)]

        # Process in parallel
        future_to_task = {
            self._executor.submit(process_single_tweet, task): task
            for task in tasks
        }

        completed = 0
        for future in as_completed(future_to_task):
            index, result, error = future.result()

            if error:
                errors.append({
                    **error,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                results.append(result)

            completed += 1
            if progress_callback:
                progress_callback(completed, len(tweets), None)

        # Sort results by original order
        results.sort(key=lambda x: tweets.index(x.tweet_id) if hasattr(x, 'tweet_id') else 0)

        return BatchResult(results=results, errors=errors)

    async def _process_async(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool,
        include_keywords: bool,
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """Process tweets asynchronously."""
        results = []
        errors = []

        async def process_single_tweet_async(index, tweet):
            try:
                # Run classification in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.service.classify_tweet,
                    tweet, include_features, include_keywords
                )
                return index, result, None
            except Exception as e:
                return index, None, {"error": str(e), "tweet_index": index}

        # Create async tasks
        tasks = [
            process_single_tweet_async(i, tweet)
            for i, tweet in enumerate(tweets)
        ]

        # Process concurrently
        completed = 0
        for future in asyncio.as_completed(tasks):
            index, result, error = await future

            if error:
                errors.append({
                    **error,
                    "timestamp": datetime.now().isoformat()
                })
            else:
                results.append(result)

            completed += 1
            if progress_callback:
                progress_callback(completed, len(tweets), None)

        # Sort results by original order
        results.sort(key=lambda x: tweets.index(x.tweet_id) if hasattr(x, 'tweet_id') else 0)

        return BatchResult(results=results, errors=errors)

    def _process_hybrid(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool,
        include_keywords: bool,
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """Process tweets using hybrid approach (async + parallel)."""
        # For small batches, use sequential for simplicity
        if len(tweets) <= 10:
            return self._process_sequential(tweets, include_features, include_keywords, progress_callback)

        # For medium batches, use parallel
        if len(tweets) <= 100:
            return self._process_parallel(tweets, include_features, include_keywords, progress_callback)

        # For large batches, use async with chunks
        if len(tweets) <= 1000:
            # Split into chunks and process with asyncio
            import asyncio

            async def process_chunk_async(chunk_tweets):
                chunk_results = []
                chunk_errors = []

                for tweet in chunk_tweets:
                    try:
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            None,
                            self.service.classify_tweet,
                            tweet, include_features, include_keywords
                        )
                        chunk_results.append(result)
                    except Exception as e:
                        chunk_errors.append({
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        })

                return chunk_results, chunk_errors

            # Split into chunks
            chunk_size = min(len(tweets) // self.config.max_workers, 100)
            chunks = [tweets[i:i + chunk_size] for i in range(0, len(tweets), chunk_size)]

            # Process chunks asynchronously
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                chunk_tasks = [process_chunk_async(chunk) for chunk in chunks]
                chunk_results_list = loop.run_until_complete(asyncio.gather(*chunk_tasks))

                # Combine results
                results = []
                errors = []

                for chunk_results, chunk_errors in chunk_results_list:
                    results.extend(chunk_results)
                    errors.extend(chunk_errors)

                return BatchResult(results=results, errors=errors)

            finally:
                loop.close()

        # For very large batches, use process pool
        return self._process_with_process_pool(tweets, include_features, include_keywords, progress_callback)

    def _process_with_process_pool(
        self,
        tweets: List[Union[str, Dict]],
        include_features: bool,
        include_keywords: bool,
        progress_callback: Optional[Callable]
    ) -> BatchResult:
        """Process tweets using process pool for CPU-bound tasks."""
        # Note: This is more complex due to serialization requirements
        # For now, fall back to parallel processing
        self.logger.warning("Process pool processing not fully implemented, using parallel instead")
        return self._process_parallel(tweets, include_features, include_keywords, progress_callback)

    def benchmark_strategies(
        self,
        test_tweets: List[Union[str, Dict]],
        num_runs: int = 3
    ) -> Dict[str, Dict]:
        """Benchmark different processing strategies.

        Args:
            test_tweets: Tweets to use for benchmarking
            num_runs: Number of runs per strategy

        Returns:
            Benchmark results for each strategy
        """
        strategies = [
            BatchProcessingStrategy.SEQUENTIAL,
            BatchProcessingStrategy.PARALLEL,
            BatchProcessingStrategy.HYBRID
        ]

        results = {}

        for strategy in strategies:
            original_strategy = self.config.strategy
            self.config.strategy = strategy

            try:
                times = []
                for _ in range(num_runs):
                    start_time = time.time()
                    result = self.process_batch(test_tweets)
                    times.append(time.time() - start_time)

                results[strategy.value] = {
                    'average_time': sum(times) / len(times),
                    'min_time': min(times),
                    'max_time': max(times),
                    'tweets_per_second': len(test_tweets) / (sum(times) / len(times)),
                    'success_rate': result.success_rate
                }

            except Exception as e:
                results[strategy.value] = {
                    'error': str(e),
                    'success': False
                }

            finally:
                self.config.strategy = original_strategy

        return results

    def _update_stats(self, result: BatchResult):
        """Update processing statistics."""
        self.stats['total_batches_processed'] += 1
        self.stats['total_tweets_processed'] += result.batch_size

        # Update average batch time
        current_avg = self.stats['average_batch_time']
        total_batches = self.stats['total_batches_processed']
        self.stats['average_batch_time'] = (
            (current_avg * (total_batches - 1) + result.processing_time) / total_batches
        )

        # Update tweets per second
        total_time = self.stats['average_batch_time'] * total_batches
        total_tweets = self.stats['total_tweets_processed']
        self.stats['average_tweets_per_second'] = total_tweets / total_time if total_time > 0 else 0

        # Update success rate
        self.stats['success_rate'] = (
            (self.stats['success_rate'] * (total_batches - 1) + result.success_rate) / total_batches
        )

    def get_stats(self) -> Dict:
        """Get current processing statistics."""
        return {
            **self.stats,
            'config': {
                'max_batch_size': self.config.max_batch_size,
                'timeout_seconds': self.config.timeout_seconds,
                'max_workers': self.config.max_workers,
                'strategy': self.config.strategy.value,
                'enable_chunking': self.config.enable_chunking,
                'chunk_size': self.config.chunk_size
            }
        }

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            'total_batches_processed': 0,
            'total_tweets_processed': 0,
            'average_batch_time': 0.0,
            'average_tweets_per_second': 0.0,
            'success_rate': 1.0,
            'memory_usage_mb': 0.0
        }

    def cleanup(self):
        """Clean up resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
            self._executor = None

        self._should_monitor = False
        if self._memory_monitor_thread:
            self._memory_monitor_thread.join(timeout=1.0)