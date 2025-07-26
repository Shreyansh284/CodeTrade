"""
Progress indicators and user feedback utilities for the Stock Pattern Detector.

This module provides enhanced progress indicators, loading states, and user feedback
mechanisms to improve the user experience during long-running operations.
"""

import time
import streamlit as st
from typing import Optional, List, Dict, Any, Callable
from contextlib import contextmanager
from datetime import datetime, timedelta

from .logging_config import get_logger

logger = get_logger(__name__)


class ProgressTracker:
    """
    Enhanced progress tracking with multiple indicators and user feedback.
    """
    
    def __init__(self, title: str = "Processing", show_time_estimate: bool = True):
        """
        Initialize progress tracker.
        
        Args:
            title: Title for the progress operation
            show_time_estimate: Whether to show time estimates
        """
        self.title = title
        self.show_time_estimate = show_time_estimate
        self.start_time = None
        self.steps = []
        self.current_step = 0
        self.total_steps = 0
        
        # UI elements
        self.progress_bar = None
        self.status_text = None
        self.time_text = None
        self.detail_text = None
        
    def initialize(self, steps: List[str]) -> None:
        """
        Initialize progress tracker with steps.
        
        Args:
            steps: List of step descriptions
        """
        self.steps = steps
        self.total_steps = len(steps)
        self.current_step = 0
        self.start_time = time.time()
        
        # Create UI elements
        st.write(f"**{self.title}**")
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()
        
        if self.show_time_estimate:
            self.time_text = st.empty()
        
        self.detail_text = st.empty()
        
        # Show initial status
        self._update_display()
    
    def next_step(self, details: str = "") -> None:
        """
        Move to the next step.
        
        Args:
            details: Additional details about the current step
        """
        if self.current_step < self.total_steps:
            self.current_step += 1
            self._update_display(details)
    
    def update_details(self, details: str) -> None:
        """
        Update details for the current step.
        
        Args:
            details: Details to display
        """
        if self.detail_text:
            self.detail_text.info(f"ℹ️ {details}")
    
    def _update_display(self, details: str = "") -> None:
        """Update the progress display."""
        try:
            if self.current_step <= self.total_steps:
                # Update progress bar
                progress = self.current_step / self.total_steps if self.total_steps > 0 else 0
                if self.progress_bar:
                    self.progress_bar.progress(progress)
                
                # Update status text
                if self.current_step < self.total_steps:
                    current_step_desc = self.steps[self.current_step]
                    status_msg = f"Step {self.current_step + 1}/{self.total_steps}: {current_step_desc}"
                else:
                    status_msg = "✅ Complete!"
                
                if self.status_text:
                    self.status_text.text(status_msg)
                
                # Update time estimate
                if self.show_time_estimate and self.time_text and self.start_time:
                    elapsed = time.time() - self.start_time
                    
                    if self.current_step > 0 and self.current_step < self.total_steps:
                        avg_time_per_step = elapsed / self.current_step
                        remaining_steps = self.total_steps - self.current_step
                        estimated_remaining = avg_time_per_step * remaining_steps
                        
                        self.time_text.caption(
                            f"⏱️ Elapsed: {elapsed:.1f}s | "
                            f"Estimated remaining: {estimated_remaining:.1f}s"
                        )
                    elif self.current_step >= self.total_steps:
                        self.time_text.caption(f"⏱️ Total time: {elapsed:.1f}s")
                
                # Update details
                if details and self.detail_text:
                    self.detail_text.info(f"ℹ️ {details}")
                    
        except Exception as e:
            logger.error(f"Error updating progress display: {e}")
    
    def complete(self, success_message: str = "Operation completed successfully!") -> None:
        """
        Mark the operation as complete.
        
        Args:
            success_message: Message to show on completion
        """
        try:
            self.current_step = self.total_steps
            self._update_display()
            
            # Show success message
            if self.detail_text:
                self.detail_text.success(f"✅ {success_message}")
            
            # Auto-clear after a delay
            time.sleep(1)
            self.clear()
            
        except Exception as e:
            logger.error(f"Error completing progress tracker: {e}")
    
    def error(self, error_message: str) -> None:
        """
        Mark the operation as failed.
        
        Args:
            error_message: Error message to display
        """
        try:
            if self.detail_text:
                self.detail_text.error(f"❌ {error_message}")
            
            if self.status_text:
                self.status_text.text("❌ Operation failed")
                
        except Exception as e:
            logger.error(f"Error showing progress error: {e}")
    
    def clear(self) -> None:
        """Clear all progress indicators."""
        try:
            if self.progress_bar:
                self.progress_bar.empty()
            if self.status_text:
                self.status_text.empty()
            if self.time_text:
                self.time_text.empty()
            if self.detail_text:
                self.detail_text.empty()
                
        except Exception as e:
            logger.error(f"Error clearing progress indicators: {e}")


@contextmanager
def progress_context(title: str, steps: List[str], show_time_estimate: bool = True):
    """
    Context manager for progress tracking.
    
    Args:
        title: Title for the operation
        steps: List of step descriptions
        show_time_estimate: Whether to show time estimates
        
    Yields:
        ProgressTracker instance
    """
    tracker = ProgressTracker(title, show_time_estimate)
    try:
        tracker.initialize(steps)
        yield tracker
    except Exception as e:
        tracker.error(f"Operation failed: {str(e)}")
        raise
    finally:
        tracker.clear()


class LoadingSpinner:
    """
    Simple loading spinner for quick operations.
    """
    
    def __init__(self, message: str = "Loading..."):
        """
        Initialize loading spinner.
        
        Args:
            message: Message to display
        """
        self.message = message
        self.spinner = None
    
    def __enter__(self):
        """Start the spinner."""
        self.spinner = st.spinner(self.message)
        return self.spinner.__enter__()
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the spinner."""
        if self.spinner:
            return self.spinner.__exit__(exc_type, exc_val, exc_tb)


def show_operation_feedback(
    operation_name: str,
    duration: float,
    success: bool = True,
    details: Optional[Dict[str, Any]] = None
) -> None:
    """
    Show feedback about a completed operation.
    
    Args:
        operation_name: Name of the operation
        duration: Duration in seconds
        success: Whether the operation was successful
        details: Additional details to display
    """
    try:
        if success:
            st.success(f"✅ {operation_name} completed in {duration:.2f}s")
        else:
            st.error(f"❌ {operation_name} failed after {duration:.2f}s")
        
        if details:
            with st.expander("📋 Operation Details", expanded=False):
                for key, value in details.items():
                    st.write(f"**{key}:** {value}")
                    
    except Exception as e:
        logger.error(f"Error showing operation feedback: {e}")


def show_data_quality_feedback(
    data_name: str,
    total_records: int,
    valid_records: int,
    issues: Optional[List[str]] = None
) -> None:
    """
    Show feedback about data quality.
    
    Args:
        data_name: Name of the data being processed
        total_records: Total number of records
        valid_records: Number of valid records
        issues: List of data quality issues
    """
    try:
        quality_pct = (valid_records / total_records * 100) if total_records > 0 else 0
        
        if quality_pct >= 95:
            st.success(f"✅ {data_name}: {valid_records:,}/{total_records:,} records ({quality_pct:.1f}% valid)")
        elif quality_pct >= 80:
            st.warning(f"⚠️ {data_name}: {valid_records:,}/{total_records:,} records ({quality_pct:.1f}% valid)")
        else:
            st.error(f"❌ {data_name}: {valid_records:,}/{total_records:,} records ({quality_pct:.1f}% valid)")
        
        if issues:
            with st.expander("🔍 Data Quality Issues", expanded=False):
                for issue in issues[:10]:  # Show first 10 issues
                    st.write(f"• {issue}")
                if len(issues) > 10:
                    st.write(f"... and {len(issues) - 10} more issues")
                    
    except Exception as e:
        logger.error(f"Error showing data quality feedback: {e}")


def show_performance_metrics(
    operation_name: str,
    metrics: Dict[str, Any],
    thresholds: Optional[Dict[str, float]] = None
) -> None:
    """
    Show performance metrics for an operation.
    
    Args:
        operation_name: Name of the operation
        metrics: Dictionary of metrics
        thresholds: Optional thresholds for performance warnings
    """
    try:
        st.subheader(f"📊 {operation_name} Performance")
        
        cols = st.columns(len(metrics))
        
        for i, (metric_name, value) in enumerate(metrics.items()):
            with cols[i]:
                # Determine if metric is within acceptable range
                threshold = thresholds.get(metric_name) if thresholds else None
                
                if threshold and isinstance(value, (int, float)):
                    if value > threshold:
                        st.metric(metric_name, value, delta=f"⚠️ Above {threshold}")
                    else:
                        st.metric(metric_name, value, delta="✅ Good")
                else:
                    st.metric(metric_name, value)
                    
    except Exception as e:
        logger.error(f"Error showing performance metrics: {e}")


def create_status_dashboard(
    title: str,
    status_items: List[Dict[str, Any]]
) -> None:
    """
    Create a status dashboard showing multiple status items.
    
    Args:
        title: Dashboard title
        status_items: List of status items with 'name', 'status', 'details' keys
    """
    try:
        st.subheader(f"📊 {title}")
        
        for item in status_items:
            name = item.get('name', 'Unknown')
            status = item.get('status', 'unknown')
            details = item.get('details', '')
            
            col1, col2, col3 = st.columns([2, 1, 3])
            
            with col1:
                st.write(f"**{name}**")
            
            with col2:
                if status == 'success':
                    st.success("✅ OK")
                elif status == 'warning':
                    st.warning("⚠️ Warning")
                elif status == 'error':
                    st.error("❌ Error")
                else:
                    st.info("ℹ️ Unknown")
            
            with col3:
                if details:
                    st.write(details)
                    
    except Exception as e:
        logger.error(f"Error creating status dashboard: {e}")


def show_timeout_warning(operation_name: str, timeout_seconds: int) -> None:
    """
    Show a timeout warning for long-running operations.
    
    Args:
        operation_name: Name of the operation
        timeout_seconds: Timeout threshold in seconds
    """
    try:
        st.warning(
            f"⏱️ **{operation_name} is taking longer than expected**\n\n"
            f"Operations typically complete within {timeout_seconds} seconds. "
            f"This may indicate:\n"
            f"• Large dataset being processed\n"
            f"• System resource constraints\n"
            f"• Network or file access issues\n\n"
            f"Please wait or consider reducing the data size."
        )
    except Exception as e:
        logger.error(f"Error showing timeout warning: {e}")


def create_progress_callback(tracker: ProgressTracker) -> Callable[[str, float], None]:
    """
    Create a callback function for progress updates.
    
    Args:
        tracker: ProgressTracker instance
        
    Returns:
        Callback function that accepts (message, progress) parameters
    """
    def callback(message: str, progress: float = None):
        """
        Progress callback function.
        
        Args:
            message: Progress message
            progress: Progress value (0.0 to 1.0)
        """
        try:
            if progress is not None:
                # Update progress bar if specific progress provided
                if tracker.progress_bar:
                    tracker.progress_bar.progress(min(max(progress, 0.0), 1.0))
            
            # Update status message
            tracker.update_details(message)
            
        except Exception as e:
            logger.error(f"Error in progress callback: {e}")
    
    return callback