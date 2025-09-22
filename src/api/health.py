"""Health check endpoint for disaster tweet classification system."""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from ..models.api_status import APIStatus
from ..inference.predictor import TweetClassificationService


class HealthCheckService:
    """Health check service for monitoring system status."""

    def __init__(self, classification_service: Optional[TweetClassificationService] = None):
        """Initialize health check service.

        Args:
            classification_service: Tweet classification service instance
        """
        self.classification_service = classification_service
        self.start_time = datetime.now()
        self.logger = logging.getLogger(__name__)

        # Health check configuration
        self.health_config = {
            'model_load_timeout': 30.0,  # seconds
            'classification_timeout': 5.0,  # seconds
            'memory_threshold_mb': 2048,  # MB
            'error_rate_threshold': 0.1,  # 10%
            'response_time_threshold': 1.0,  # seconds
            'max_uptime_hours': 24 * 7  # 1 week
        }

    def get_health_status(self, detailed: bool = False) -> Dict[str, Any]:
        """Get comprehensive health status.

        Args:
            detailed: Whether to include detailed health information

        Returns:
            Health status dictionary
        """
        try:
            # Basic health check
            basic_health = self._perform_basic_health_check()

            if detailed:
                # Detailed health check
                detailed_health = self._perform_detailed_health_check()
                health_status = {**basic_health, **detailed_health}
            else:
                health_status = basic_health

            # Determine overall status
            overall_status = self._determine_overall_status(health_status)
            health_status['status'] = overall_status
            health_status['timestamp'] = datetime.now().isoformat()

            return health_status

        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': (datetime.now() - self.start_time).total_seconds()
            }

    def _perform_basic_health_check(self) -> Dict[str, Any]:
        """Perform basic health checks."""
        basic_checks = {
            'service_available': True,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
            'timestamp': datetime.now().isoformat()
        }

        # Check if classification service is available
        if self.classification_service:
            service_status = self.classification_service.get_service_status()
            basic_checks['classification_service'] = {
                'is_initialized': service_status.get('is_initialized', False),
                'is_ready': service_status.get('is_ready', False),
                'model_loaded': service_status.get('model_info', {}).get('is_loaded', False)
            }
        else:
            basic_checks['classification_service'] = {
                'is_initialized': False,
                'is_ready': False,
                'model_loaded': False
            }

        return basic_checks

    def _perform_detailed_health_check(self) -> Dict[str, Any]:
        """Perform detailed health checks."""
        detailed_checks = {
            'system_resources': self._check_system_resources(),
            'model_health': self._check_model_health(),
            'performance_metrics': self._check_performance_metrics(),
            'recent_activity': self._check_recent_activity(),
            'configuration_status': self._check_configuration()
        }

        return detailed_checks

    def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource usage."""
        try:
            import psutil

            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage_mb = memory.used / 1024 / 1024

            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent

            return {
                'memory_usage_mb': round(memory_usage_mb, 2),
                'memory_usage_percent': memory.percent,
                'memory_available_mb': round(memory.available / 1024 / 1024, 2),
                'cpu_usage_percent': cpu_percent,
                'disk_usage_percent': disk_usage_percent,
                'memory_threshold_exceeded': memory_usage_mb > self.health_config['memory_threshold_mb']
            }

        except ImportError:
            return {
                'error': 'psutil not available',
                'memory_usage_mb': 'unknown',
                'memory_usage_percent': 'unknown',
                'cpu_usage_percent': 'unknown',
                'disk_usage_percent': 'unknown'
            }

    def _check_model_health(self) -> Dict[str, Any]:
        """Check model health status."""
        if not self.classification_service or not self.classification_service.is_ready():
            return {
                'model_loaded': False,
                'model_info': None,
                'model_performance': None,
                'model_age_hours': None
            }

        try:
            # Get model information
            model_info = self.classification_service.classifier.get_model_info()
            service_stats = self.classification_service.get_service_status()['statistics']

            # Calculate model age (if available)
            model_age_hours = None
            if model_info.get('created_at'):
                created_time = datetime.fromisoformat(model_info['created_at'].replace('Z', '+00:00'))
                model_age = datetime.now() - created_time
                model_age_hours = model_age.total_seconds() / 3600

            # Test model inference
            test_start = time.time()
            try:
                test_result = self.classification_service.classify_text("Test tweet for health check")
                inference_time = time.time() - test_start
                model_responsive = True
            except Exception as e:
                inference_time = None
                model_responsive = False
                self.logger.warning(f"Model inference test failed: {str(e)}")

            return {
                'model_loaded': True,
                'model_info': {
                    'version': model_info.get('version'),
                    'model_type': model_info.get('model_type'),
                    'parameters': model_info.get('parameters'),
                    'device': model_info.get('device')
                },
                'model_performance': {
                    'inference_time_seconds': inference_time,
                    'total_predictions': service_stats.get('total_predictions', 0),
                    'successful_predictions': service_stats.get('successful_predictions', 0),
                    'failed_predictions': service_stats.get('failed_predictions', 0),
                    'average_processing_time': service_stats.get('average_processing_time', 0),
                    'model_responsive': model_responsive
                },
                'model_age_hours': model_age_hours,
                'model_too_old': model_age_hours and model_age_hours > 24 * 30  # 30 days
            }

        except Exception as e:
            self.logger.error(f"Model health check failed: {str(e)}")
            return {
                'model_loaded': False,
                'error': str(e),
                'model_performance': None
            }

    def _check_performance_metrics(self) -> Dict[str, Any]:
        """Check performance metrics."""
        if not self.classification_service:
            return {
                'response_time_ok': False,
                'error_rate_ok': False,
                'throughput_ok': False
            }

        try:
            service_stats = self.classification_service.get_service_status()['statistics']

            # Calculate error rate
            total_predictions = service_stats.get('total_predictions', 0)
            failed_predictions = service_stats.get('failed_predictions', 0)
            error_rate = failed_predictions / total_predictions if total_predictions > 0 else 0

            # Check response time
            avg_processing_time = service_stats.get('average_processing_time', 0)
            response_time_ok = avg_processing_time <= self.health_config['response_time_threshold']

            # Check error rate
            error_rate_ok = error_rate <= self.health_config['error_rate_threshold']

            # Calculate throughput
            uptime = service_stats.get('total_processing_time', 1)
            throughput = total_predictions / uptime if uptime > 0 else 0
            throughput_ok = throughput > 0.1  # At least 0.1 predictions per second

            return {
                'average_response_time_seconds': avg_processing_time,
                'error_rate': error_rate,
                'predictions_per_second': throughput,
                'total_predictions': total_predictions,
                'response_time_ok': response_time_ok,
                'error_rate_ok': error_rate_ok,
                'throughput_ok': throughput_ok,
                'performance_degraded': not (response_time_ok and error_rate_ok and throughput_ok)
            }

        except Exception as e:
            self.logger.error(f"Performance metrics check failed: {str(e)}")
            return {
                'error': str(e),
                'response_time_ok': False,
                'error_rate_ok': False,
                'throughput_ok': False
            }

    def _check_recent_activity(self) -> Dict[str, Any]:
        """Check recent system activity."""
        if not self.classification_service:
            return {
                'recent_activity': False,
                'last_prediction_time': None,
                'activity_stale': True
            }

        try:
            service_status = self.classification_service.get_service_status()
            last_prediction = service_status.get('last_prediction_time')

            if last_prediction:
                last_prediction_time = datetime.fromisoformat(last_prediction)
                time_since_last = datetime.now() - last_prediction_time
                activity_stale = time_since_last > timedelta(minutes=5)  # 5 minutes threshold
                recent_activity = not activity_stale
            else:
                last_prediction_time = None
                activity_stale = True
                recent_activity = False

            return {
                'recent_activity': recent_activity,
                'last_prediction_time': last_prediction_time.isoformat() if last_prediction_time else None,
                'time_since_last_prediction_seconds': time_since_last.total_seconds() if last_prediction_time else None,
                'activity_stale': activity_stale
            }

        except Exception as e:
            self.logger.error(f"Recent activity check failed: {str(e)}")
            return {
                'error': str(e),
                'recent_activity': False,
                'activity_stale': True
            }

    def _check_configuration(self) -> Dict[str, Any]:
        """Check configuration status."""
        config_status = {
            'health_check_config': self.health_config.copy(),
            'configuration_valid': True,
            'configuration_issues': []
        }

        # Check if service is configured properly
        if not self.classification_service:
            config_status['configuration_valid'] = False
            config_status['configuration_issues'].append('Classification service not configured')

        # Check uptime
        uptime_hours = (datetime.now() - self.start_time).total_seconds() / 3600
        if uptime_hours > self.health_config['max_uptime_hours']:
            config_status['configuration_issues'].append(f'System uptime ({uptime_hours:.1f}h) exceeds recommended maximum')

        return config_status

    def _determine_overall_status(self, health_status: Dict[str, Any]) -> str:
        """Determine overall health status."""
        # Check for critical errors
        if health_status.get('error'):
            return 'error'

        # Check system resources
        system_resources = health_status.get('system_resources', {})
        if system_resources.get('memory_threshold_exceeded'):
            return 'degraded'

        # Check model health
        model_health = health_status.get('model_health', {})
        if not model_health.get('model_loaded'):
            return 'degraded'

        model_performance = model_health.get('model_performance', {})
        if not model_performance.get('model_responsive'):
            return 'degraded'

        # Check performance metrics
        performance_metrics = health_status.get('performance_metrics', {})
        if performance_metrics.get('performance_degraded'):
            return 'degraded'

        # Check recent activity
        recent_activity = health_status.get('recent_activity', {})
        if recent_activity.get('activity_stale'):
            return 'degraded'

        # Check configuration
        configuration_status = health_status.get('configuration_status', {})
        if not configuration_status.get('configuration_valid'):
            return 'degraded'

        # All checks passed
        return 'healthy'

    def create_api_status(self) -> APIStatus:
        """Create API status object from health check."""
        health_status = self.get_health_status(detailed=True)

        # Extract relevant information for API status
        api_status = APIStatus(
            status=health_status.get('status', 'healthy'),
            last_tweet_collected=None,  # Will be updated by tweet collection service
            last_prediction_made=datetime.fromisoformat(
                health_status.get('recent_activity', {}).get('last_prediction_time') or datetime.now().isoformat()
            ),
            api_rate_limit_remaining=300,  # Default value
            model_loaded=health_status.get('model_health', {}).get('model_loaded', False),
            error_message=None if health_status.get('status') == 'healthy' else 'System degraded',
            uptime_seconds=int(health_status.get('uptime_seconds', 0))
        )

        return api_status

    def perform_readiness_check(self) -> bool:
        """Perform readiness check (simplified health check)."""
        try:
            # Quick check of essential services
            if not self.classification_service or not self.classification_service.is_ready():
                return False

            # Quick model test
            test_result = self.classification_service.classify_text("Test")
            return test_result is not None

        except Exception:
            return False

    def perform_liveness_check(self) -> bool:
        """Perform liveness check (very basic check)."""
        try:
            # Just check if service is running
            return True
        except Exception:
            return False