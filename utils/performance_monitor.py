"""
Performance monitoring system for the Stock Pattern Detector application.

This module provides performance monitoring, timeout handling, and optimization
suggestions to ensure operations complete within acceptable time limits.
"""

import time
import threading
from typing import Dict, Any, Optional, Callable, List
from datetime import datetime, timedelta
from contextlib import contextmanager
import streamlit as st

from .logging_config import get_logger, log_performance

logger = get_logger(__name__)


class PerformanceMonitor:
    """
    Monitors application performance and enforces time limits.
    
    Features:
    - Operation timeout enforcement
    - Performance metrics collection
    - Bottleneck identification
    - Optimization suggestions
    """
    
    def __init__(self, default_timeout: int = 30):
        """
        Initialize the performance monitor.
        
        Args:
            default_timeout: Default timeout in seconds
        """
        self.default_timeout = default_timeout
        self.active_operations: Dict[str, Dict[str, Any]] = {}
        self.performance_history: List[Dict[str, Any]] = []
        self.max_history_size = 100
        
        # Performance thresholds
        self.thresholds = {
            'data_loading': 10.0,      # 10 seconds
            'data_aggregation': 5.0,   # 5 seconds
            'pattern_detection': 15.0, # 15 seconds
            'chart_rendering': 3.0,    # 3 seconds
            'total_analysis': 30.0     # 30 seconds total
        }
        
        # Optimization suggestions
        self.optimization_suggestions = {
            'data_loading': [
                "Consider using data caching to avoid reloading files",
                "Check if CSV files are too large and consider data sampling",
                "Verify file system performance and disk space",
                "Use SSD storage for better I/O performance"
            ],
            'data_aggregation': [
                "Use vectorized pandas operations for better performance",
                "Consider pre-aggregating data for common timeframes",
                "Reduce the amount of data being processed",
                "Check memory usage and available RAM"
            ],
            'pattern_detection': [
                "Implement vectorized pattern detection algorithms",
                "Use parallel processing for multiple patterns",
                "Cache pattern detection results",
                "Reduce the number of patterns being detected simultaneously"
            ],
            'chart_rendering': [
                "Implement data sampling for large datasets",
                "Reduce chart complexity and number of traces",
                "Use chart caching for repeated views",
                "Consider server-side chart generation"
            ]
        }
    
    @contextmanager
    def monitor_operation(
        self, 
        operation_name: str, 
        timeout: Optional[int] = None,
        show_progress: bool = True
    ):
        """
        Context manager for monitoring operation performance.
        
        Args:
            operation_name: Name of the operation
            timeout: Timeout in seconds (uses default if None)
            show_progress: Whether to show progress indicators
            
        Yields:
            OperationContext object
        """
        timeout = timeout or self.default_timeout
        operation_id = f"{operation_name}_{int(time.time())}"
        
        # Create operation context
        context = OperationContext(
            operation_id=operation_id,
            operation_name=operation_name,
            timeout=timeout,
            monitor=self
        )
        
        # Track active operation
        self.active_operations[operation_id] = {
            'name': operation_name,
            'start_time': time.time(),
            'timeout': timeout,
            'context': context
        }
        
        # Show progress if requested
        progress_placeholder = None
        if show_progress:
            progress_placeholder = st.empty()
            progress_placeholder.info(f"🔄 Starting {operation_name}...")
        
        try:
            # Start timeout monitoring
            timeout_thread = threading.Thread(
                target=self._monitor_timeout,
                args=(operation_id, timeout),
                daemon=True
            )
            timeout_thread.start()
            
            yield context
            
            # Operation completed successfully
            self._complete_operation(operation_id, success=True)
            
        except TimeoutError as e:
            self._complete_operation(operation_id, success=False, error=str(e))
            if progress_placeholder:
                progress_placeholder.error(f"⏱️ {operation_name} timed out after {timeout}s")
            raise
            
        except Exception as e:
            self._complete_operation(operation_id, success=False, error=str(e))
            if progress_placeholder:
                progress_placeholder.error(f"❌ {operation_name} failed: {str(e)}")
            raise
            
        finally:
            # Clean up
            if operation_id in self.active_operations:
                del self.active_operations[operation_id]
            
            if progress_placeholder:
                progress_placeholder.empty()
    
    def _monitor_timeout(self, operation_id: str, timeout: int) -> None:
        """
        Monitor operation timeout in a separate thread.
        
        Args:
            operation_id: ID of the operation to monitor
            timeout: Timeout in seconds
        """
        try:
            time.sleep(timeout)
            
            # Check if operation is still active
            if operation_id in self.active_operations:
                operation = self.active_operations[operation_id]
                operation_name = operation['name']
                
                logger.warning(f"Operation {operation_name} timed out after {timeout}s")
                
                # Try to interrupt the operation
                context = operation.get('context')
                if context:
                    context.timeout_occurred = True
                
                # Show timeout warning to user
                st.warning(
                    f"⏱️ **{operation_name} is taking longer than expected**\n\n"
                    f"The operation has exceeded the {timeout}s timeout. "
                    f"This may indicate performance issues or large data processing."
                )
                
        except Exception as e:
            logger.error(f"Error in timeout monitoring: {e}")
    
    def _complete_operation(
        self, 
        operation_id: str, 
        success: bool, 
        error: Optional[str] = None
    ) -> None:
        """
        Complete an operation and record performance metrics.
        
        Args:
            operation_id: ID of the operation
            success: Whether the operation succeeded
            error: Error message if failed
        """
        try:
            if operation_id not in self.active_operations:
                return
            
            operation = self.active_operations[operation_id]
            end_time = time.time()
            duration = end_time - operation['start_time']
            
            # Record performance metrics
            performance_record = {
                'operation_name': operation['name'],
                'duration': duration,
                'success': success,
                'error': error,
                'timestamp': datetime.now(),
                'timeout': operation['timeout']
            }
            
            self.performance_history.append(performance_record)
            
            # Limit history size
            if len(self.performance_history) > self.max_history_size:
                self.performance_history.pop(0)
            
            # Log performance
            log_performance(operation['name'], duration, f"success={success}")
            
            # Check if operation exceeded threshold
            threshold = self.thresholds.get(operation['name'], self.default_timeout)
            if duration > threshold:
                self._suggest_optimizations(operation['name'], duration, threshold)
            
        except Exception as e:
            logger.error(f"Error completing operation {operation_id}: {e}")
    
    def _suggest_optimizations(
        self, 
        operation_name: str, 
        duration: float, 
        threshold: float
    ) -> None:
        """
        Suggest optimizations for slow operations.
        
        Args:
            operation_name: Name of the slow operation
            duration: Actual duration
            threshold: Expected threshold
        """
        try:
            suggestions = self.optimization_suggestions.get(operation_name, [])
            
            if suggestions:
                st.warning(
                    f"⚡ **Performance Notice**\n\n"
                    f"{operation_name} took {duration:.1f}s (expected < {threshold:.1f}s)\n\n"
                    f"**Optimization suggestions:**\n" +
                    "\n".join([f"• {suggestion}" for suggestion in suggestions[:3]])
                )
                
        except Exception as e:
            logger.error(f"Error suggesting optimizations: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get performance summary and statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not self.performance_history:
                return {'message': 'No performance data available'}
            
            # Calculate statistics
            total_operations = len(self.performance_history)
            successful_operations = sum(1 for record in self.performance_history if record['success'])
            success_rate = (successful_operations / total_operations) * 100
            
            # Calculate average durations by operation
            operation_stats = {}
            for record in self.performance_history:
                op_name = record['operation_name']
                if op_name not in operation_stats:
                    operation_stats[op_name] = {
                        'count': 0,
                        'total_duration': 0,
                        'max_duration': 0,
                        'min_duration': float('inf'),
                        'failures': 0
                    }
                
                stats = operation_stats[op_name]
                stats['count'] += 1
                stats['total_duration'] += record['duration']
                stats['max_duration'] = max(stats['max_duration'], record['duration'])
                stats['min_duration'] = min(stats['min_duration'], record['duration'])
                
                if not record['success']:
                    stats['failures'] += 1
            
            # Calculate averages
            for op_name, stats in operation_stats.items():
                stats['avg_duration'] = stats['total_duration'] / stats['count']
                stats['success_rate'] = ((stats['count'] - stats['failures']) / stats['count']) * 100
            
            # Find slowest operations
            slowest_operations = sorted(
                operation_stats.items(),
                key=lambda x: x[1]['avg_duration'],
                reverse=True
            )[:3]
            
            return {
                'total_operations': total_operations,
                'success_rate': success_rate,
                'operation_stats': operation_stats,
                'slowest_operations': slowest_operations,
                'active_operations': len(self.active_operations),
                'thresholds': self.thresholds
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def clear_history(self) -> None:
        """Clear performance history."""
        try:
            self.performance_history.clear()
            logger.info("Performance history cleared")
        except Exception as e:
            logger.error(f"Error clearing performance history: {e}")
    
    def update_thresholds(self, new_thresholds: Dict[str, float]) -> None:
        """
        Update performance thresholds.
        
        Args:
            new_thresholds: Dictionary of operation names to threshold values
        """
        try:
            self.thresholds.update(new_thresholds)
            logger.info(f"Updated performance thresholds: {new_thresholds}")
        except Exception as e:
            logger.error(f"Error updating thresholds: {e}")


class OperationContext:
    """
    Context object for monitored operations.
    
    Provides methods to check timeout status and update progress.
    """
    
    def __init__(self, operation_id: str, operation_name: str, timeout: int, monitor: PerformanceMonitor):
        """
        Initialize operation context.
        
        Args:
            operation_id: Unique operation ID
            operation_name: Name of the operation
            timeout: Timeout in seconds
            monitor: Performance monitor instance
        """
        self.operation_id = operation_id
        self.operation_name = operation_name
        self.timeout = timeout
        self.monitor = monitor
        self.start_time = time.time()
        self.timeout_occurred = False
        self.checkpoints: List[Dict[str, Any]] = []
    
    def check_timeout(self) -> None:
        """
        Check if operation has timed out and raise TimeoutError if so.
        
        Raises:
            TimeoutError: If operation has exceeded timeout
        """
        if self.timeout_occurred:
            raise TimeoutError(f"Operation {self.operation_name} timed out after {self.timeout}s")
        
        elapsed = time.time() - self.start_time
        if elapsed > self.timeout:
            self.timeout_occurred = True
            raise TimeoutError(f"Operation {self.operation_name} timed out after {elapsed:.1f}s")
    
    def add_checkpoint(self, name: str, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a performance checkpoint.
        
        Args:
            name: Checkpoint name
            details: Optional additional details
        """
        try:
            checkpoint = {
                'name': name,
                'timestamp': time.time(),
                'elapsed': time.time() - self.start_time,
                'details': details or {}
            }
            self.checkpoints.append(checkpoint)
            
            logger.debug(f"Checkpoint {name} at {checkpoint['elapsed']:.3f}s")
            
        except Exception as e:
            logger.warning(f"Error adding checkpoint: {e}")
    
    def get_elapsed_time(self) -> float:
        """Get elapsed time since operation start."""
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        """Get remaining time before timeout."""
        return max(0, self.timeout - self.get_elapsed_time())


# Global performance monitor instance
performance_monitor = PerformanceMonitor()


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    return performance_monitor


@contextmanager
def monitor_performance(
    operation_name: str, 
    timeout: Optional[int] = None,
    show_progress: bool = True
):
    """
    Context manager for performance monitoring.
    
    Args:
        operation_name: Name of the operation
        timeout: Timeout in seconds
        show_progress: Whether to show progress indicators
        
    Yields:
        OperationContext object
    """
    with performance_monitor.monitor_operation(operation_name, timeout, show_progress) as context:
        yield context


def check_performance_requirements() -> Dict[str, Any]:
    """
    Check if performance requirements are being met.
    
    Returns:
        Dictionary with performance status
    """
    try:
        summary = performance_monitor.get_performance_summary()
        
        if 'error' in summary:
            return {'status': 'error', 'message': summary['error']}
        
        if 'message' in summary:
            return {'status': 'no_data', 'message': summary['message']}
        
        # Check if any operations are consistently slow
        slow_operations = []
        for op_name, stats in summary.get('operation_stats', {}).items():
            threshold = performance_monitor.thresholds.get(op_name, 30.0)
            if stats['avg_duration'] > threshold:
                slow_operations.append({
                    'operation': op_name,
                    'avg_duration': stats['avg_duration'],
                    'threshold': threshold,
                    'count': stats['count']
                })
        
        # Determine overall status
        if not slow_operations:
            status = 'good'
            message = "All operations are meeting performance requirements"
        elif len(slow_operations) == 1:
            status = 'warning'
            message = f"1 operation is slower than expected: {slow_operations[0]['operation']}"
        else:
            status = 'poor'
            message = f"{len(slow_operations)} operations are slower than expected"
        
        return {
            'status': status,
            'message': message,
            'slow_operations': slow_operations,
            'success_rate': summary.get('success_rate', 0),
            'total_operations': summary.get('total_operations', 0)
        }
        
    except Exception as e:
        logger.error(f"Error checking performance requirements: {e}")
        return {'status': 'error', 'message': str(e)}