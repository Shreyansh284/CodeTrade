"""
Error handling utilities for the Stock Pattern Detector application.

This module provides comprehensive error handling, user-friendly error messages,
fallback options, and error recovery mechanisms.
"""

import functools
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import streamlit as st

from .logging_config import get_logger, log_error, log_warning, log_performance


logger = get_logger(__name__)


class ApplicationError(Exception):
    """Base exception class for application-specific errors."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        """
        Initialize application error.
        
        Args:
            message: Human-readable error message
            error_code: Unique error code for identification
            context: Additional context information
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "UNKNOWN_ERROR"
        self.context = context or {}
        self.timestamp = datetime.now()


class DataLoadingError(ApplicationError):
    """Exception raised when data loading fails."""
    pass


class DataValidationError(ApplicationError):
    """Exception raised when data validation fails."""
    pass


class PatternDetectionError(ApplicationError):
    """Exception raised when pattern detection fails."""
    pass


class VisualizationError(ApplicationError):
    """Exception raised when chart rendering fails."""
    pass


class ErrorHandler:
    """
    Centralized error handling and user feedback manager.
    
    Provides error recovery, user-friendly messages, and fallback options.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_messages = {
            # Data loading errors
            "DATA_LOAD_FAILED": {
                "title": "📥 Data Loading Failed",
                "message": "Unable to load the requested data. Please check if the data files exist and are accessible.",
                "suggestions": [
                    "Verify that CSV files are present in the 5Scripts directory",
                    "Check file permissions and ensure files are not corrupted",
                    "Try selecting a different instrument",
                    "Refresh the page and try again"
                ]
            },
            "INVALID_DATA_FORMAT": {
                "title": "📋 Invalid Data Format",
                "message": "The data format is not compatible with the expected structure.",
                "suggestions": [
                    "Ensure CSV files contain required columns: date, time, open, high, low, close, volume",
                    "Check that date format is DD-MM-YYYY and time format is HH:MM:SS",
                    "Verify that numeric values are properly formatted",
                    "Remove any extra headers or formatting from CSV files"
                ]
            },
            "INSUFFICIENT_DATA": {
                "title": "📊 Insufficient Data",
                "message": "Not enough data points available for analysis.",
                "suggestions": [
                    "Try selecting a different time period",
                    "Check if more data files are available for this instrument",
                    "Consider using a shorter timeframe for aggregation",
                    "Verify that the selected date range contains trading data"
                ]
            },
            
            # Pattern detection errors
            "PATTERN_DETECTION_FAILED": {
                "title": "🔍 Pattern Detection Failed",
                "message": "Unable to complete pattern detection analysis.",
                "suggestions": [
                    "Try reducing the number of selected patterns",
                    "Check if the data quality is sufficient for analysis",
                    "Consider using a different timeframe",
                    "Refresh and try the analysis again"
                ]
            },
            "NO_PATTERNS_FOUND": {
                "title": "🔍 No Patterns Detected",
                "message": "No patterns were found matching the selected criteria.",
                "suggestions": [
                    "Try adjusting the confidence threshold",
                    "Select additional pattern types to detect",
                    "Use a different timeframe for analysis",
                    "Check a different date range or instrument"
                ]
            },
            
            # Visualization errors
            "CHART_RENDER_FAILED": {
                "title": "📈 Chart Rendering Failed",
                "message": "Unable to create the requested chart visualization.",
                "suggestions": [
                    "Try refreshing the page",
                    "Check if the data is valid for charting",
                    "Reduce the amount of data being displayed",
                    "Try a different chart configuration"
                ]
            },
            
            # General errors
            "PROCESSING_TIMEOUT": {
                "title": "⏱️ Processing Timeout",
                "message": "The operation took too long to complete and was stopped.",
                "suggestions": [
                    "Try processing smaller amounts of data",
                    "Use a longer timeframe to reduce data points",
                    "Check system resources and try again",
                    "Consider processing data in smaller batches"
                ]
            },
            "UNKNOWN_ERROR": {
                "title": "❌ Unexpected Error",
                "message": "An unexpected error occurred during processing.",
                "suggestions": [
                    "Try refreshing the page and starting over",
                    "Check the application logs for more details",
                    "Try with different settings or data",
                    "Contact support if the problem persists"
                ]
            }
        }
    
    def handle_error(
        self, 
        error: Exception, 
        context: str = "", 
        show_user_message: bool = True,
        fallback_action: Optional[Callable] = None
    ) -> Optional[Any]:
        """
        Handle an error with logging, user feedback, and optional fallback.
        
        Args:
            error: Exception that occurred
            context: Context where the error occurred
            show_user_message: Whether to show user-friendly message
            fallback_action: Optional fallback function to execute
            
        Returns:
            Result of fallback action if provided, None otherwise
        """
        try:
            # Log the error
            log_error(error, context)
            
            # Determine error type and code
            error_code = getattr(error, 'error_code', 'UNKNOWN_ERROR')
            
            if isinstance(error, DataLoadingError):
                error_code = error.error_code or "DATA_LOAD_FAILED"
            elif isinstance(error, DataValidationError):
                error_code = error.error_code or "INVALID_DATA_FORMAT"
            elif isinstance(error, PatternDetectionError):
                error_code = error.error_code or "PATTERN_DETECTION_FAILED"
            elif isinstance(error, VisualizationError):
                error_code = error.error_code or "CHART_RENDER_FAILED"
            
            # Show user-friendly message
            if show_user_message:
                self.show_error_message(error_code, str(error))
            
            # Execute fallback action if provided
            if fallback_action:
                try:
                    logger.info(f"Executing fallback action for error: {error_code}")
                    return fallback_action()
                except Exception as fallback_error:
                    log_error(fallback_error, f"Fallback action failed for {context}")
            
            return None
            
        except Exception as handler_error:
            # Critical error in error handler itself
            logger.critical(f"Error handler failed: {handler_error}")
            if show_user_message:
                st.error("❌ A critical error occurred. Please refresh the page and try again.")
    
    def show_error_message(self, error_code: str, technical_message: str = "") -> None:
        """
        Display user-friendly error message in Streamlit interface.
        
        Args:
            error_code: Error code to look up message
            technical_message: Technical error details
        """
        try:
            error_info = self.error_messages.get(error_code, self.error_messages["UNKNOWN_ERROR"])
            
            # Show main error message
            st.error(f"**{error_info['title']}**")
            st.write(error_info['message'])
            
            # Show suggestions in an expandable section
            with st.expander("💡 Suggested Solutions", expanded=True):
                for i, suggestion in enumerate(error_info['suggestions'], 1):
                    st.write(f"{i}. {suggestion}")
            
            # Show technical details in a collapsible section
            if technical_message:
                with st.expander("🔧 Technical Details", expanded=False):
                    st.code(technical_message)
            
        except Exception as e:
            # Fallback to basic error display
            st.error("❌ An error occurred. Please try again or contact support.")
            logger.error(f"Failed to show error message: {e}")
    
    def show_warning_message(self, message: str, suggestions: List[str] = None) -> None:
        """
        Display user-friendly warning message.
        
        Args:
            message: Warning message to display
            suggestions: Optional list of suggestions
        """
        try:
            st.warning(f"⚠️ **Warning:** {message}")
            
            if suggestions:
                with st.expander("💡 Suggestions", expanded=False):
                    for i, suggestion in enumerate(suggestions, 1):
                        st.write(f"{i}. {suggestion}")
                        
        except Exception as e:
            logger.error(f"Failed to show warning message: {e}")
    
    def validate_data_integrity(self, data: pd.DataFrame, context: str = "") -> Tuple[bool, List[str]]:
        """
        Validate data integrity and return issues found.
        
        Args:
            data: DataFrame to validate
            context: Context for validation
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        try:
            if data is None:
                issues.append("Data is None")
                return False, issues
            
            if data.empty:
                issues.append("Data is empty")
                return False, issues
            
            # Check required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                issues.append(f"Missing required columns: {missing_columns}")
            
            # Check for null values in critical columns
            for col in required_columns:
                if col in data.columns:
                    null_count = data[col].isnull().sum()
                    if null_count > 0:
                        issues.append(f"Column '{col}' has {null_count} null values")
            
            # Check OHLC relationships
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = ~(
                    (data['high'] >= data['open']) &
                    (data['high'] >= data['close']) &
                    (data['high'] >= data['low']) &
                    (data['low'] <= data['open']) &
                    (data['low'] <= data['close'])
                )
                invalid_count = invalid_ohlc.sum()
                if invalid_count > 0:
                    issues.append(f"{invalid_count} rows have invalid OHLC relationships")
            
            # Check for negative prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                if col in data.columns:
                    negative_count = (data[col] <= 0).sum()
                    if negative_count > 0:
                        issues.append(f"Column '{col}' has {negative_count} non-positive values")
            
            # Check volume
            if 'volume' in data.columns:
                negative_volume = (data['volume'] < 0).sum()
                if negative_volume > 0:
                    issues.append(f"Volume has {negative_volume} negative values")
            
            is_valid = len(issues) == 0
            
            if not is_valid:
                log_warning(f"Data validation issues in {context}: {issues}")
            
            return is_valid, issues
            
        except Exception as e:
            log_error(e, f"Data validation failed in {context}")
            return False, [f"Validation error: {str(e)}"]
    
    def create_fallback_data(self, error_context: str = "") -> pd.DataFrame:
        """
        Create minimal fallback data for testing/demo purposes.
        
        Args:
            error_context: Context where fallback is needed
            
        Returns:
            DataFrame with sample data
        """
        try:
            logger.info(f"Creating fallback data for context: {error_context}")
            
            # Create minimal sample data
            dates = pd.date_range(start='2024-01-01 09:30:00', periods=100, freq='1min')
            
            # Generate realistic OHLCV data
            import numpy as np
            np.random.seed(42)  # For reproducible results
            
            base_price = 100.0
            prices = []
            volumes = []
            
            for i in range(len(dates)):
                # Simple random walk for price
                change = np.random.normal(0, 0.5)
                base_price = max(base_price + change, 50.0)  # Minimum price of 50
                
                # Generate OHLC around base price
                high = base_price + abs(np.random.normal(0, 1))
                low = base_price - abs(np.random.normal(0, 1))
                open_price = base_price + np.random.normal(0, 0.5)
                close_price = base_price + np.random.normal(0, 0.5)
                
                # Ensure OHLC relationships
                high = max(high, open_price, close_price)
                low = min(low, open_price, close_price)
                
                prices.append({
                    'open': round(open_price, 2),
                    'high': round(high, 2),
                    'low': round(low, 2),
                    'close': round(close_price, 2)
                })
                
                # Generate volume
                volume = int(np.random.normal(10000, 3000))
                volumes.append(max(volume, 1000))  # Minimum volume
            
            # Create DataFrame
            fallback_data = pd.DataFrame({
                'datetime': dates,
                'open': [p['open'] for p in prices],
                'high': [p['high'] for p in prices],
                'low': [p['low'] for p in prices],
                'close': [p['close'] for p in prices],
                'volume': volumes
            })
            
            logger.info(f"Created fallback data with {len(fallback_data)} records")
            return fallback_data
            
        except Exception as e:
            log_error(e, "Failed to create fallback data")
            # Return absolute minimal data
            return pd.DataFrame({
                'datetime': [pd.Timestamp.now()],
                'open': [100.0],
                'high': [101.0],
                'low': [99.0],
                'close': [100.5],
                'volume': [10000]
            })


# Global error handler instance
error_handler = ErrorHandler()


def with_error_handling(
    error_types: Union[Exception, Tuple[Exception, ...]] = Exception,
    context: str = "",
    show_user_message: bool = True,
    fallback_result: Any = None,
    fallback_function: Optional[Callable] = None
):
    """
    Decorator for automatic error handling in functions.
    
    Args:
        error_types: Exception types to catch
        context: Context description for logging
        show_user_message: Whether to show user-friendly error message
        fallback_result: Default result to return on error
        fallback_function: Function to call for fallback result
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log performance for slow operations
                if duration > 1.0:
                    log_performance(func.__name__, duration, context)
                
                return result
                
            except error_types as e:
                func_context = context or f"{func.__module__}.{func.__name__}"
                
                # Use fallback function if provided
                if fallback_function:
                    try:
                        return fallback_function(*args, **kwargs)
                    except Exception as fallback_error:
                        log_error(fallback_error, f"Fallback failed for {func_context}")
                
                # Handle the error
                error_handler.handle_error(e, func_context, show_user_message)
                return fallback_result
                
        return wrapper
    return decorator


def safe_execute(
    func: Callable,
    *args,
    context: str = "",
    fallback_result: Any = None,
    show_errors: bool = True,
    **kwargs
) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        context: Context for error logging
        fallback_result: Result to return on error
        show_errors: Whether to show error messages to user
        **kwargs: Function keyword arguments
        
    Returns:
        Function result or fallback_result on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        error_handler.handle_error(e, context or func.__name__, show_errors)
        return fallback_result


def validate_and_handle_data(
    data: pd.DataFrame,
    context: str = "",
    create_fallback: bool = False
) -> Tuple[pd.DataFrame, bool]:
    """
    Validate data and handle issues with optional fallback creation.
    
    Args:
        data: DataFrame to validate
        context: Context for validation
        create_fallback: Whether to create fallback data on validation failure
        
    Returns:
        Tuple of (validated_data, is_original_data)
    """
    try:
        is_valid, issues = error_handler.validate_data_integrity(data, context)
        
        if is_valid:
            return data, True
        
        # Show validation issues
        if issues:
            error_handler.show_warning_message(
                f"Data validation issues detected in {context}",
                issues
            )
        
        # Create fallback data if requested
        if create_fallback:
            fallback_data = error_handler.create_fallback_data(context)
            st.info("🔄 Using sample data for demonstration purposes.")
            return fallback_data, False
        
        return data, False
        
    except Exception as e:
        log_error(e, f"Data validation and handling failed for {context}")
        if create_fallback:
            return error_handler.create_fallback_data(context), False
        return data, False