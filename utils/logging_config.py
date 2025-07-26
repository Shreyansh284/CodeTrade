"""
Comprehensive logging configuration for the Stock Pattern Detector application.

This module provides centralized logging configuration with different levels,
formatters, and handlers for debugging and monitoring.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import traceback


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        if hasattr(record, 'levelname'):
            color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
            record.levelname = f"{color}{record.levelname}{self.COLORS['RESET']}"
        return super().format(record)


class ApplicationLogger:
    """
    Centralized logging manager for the Stock Pattern Detector application.
    
    Provides structured logging with file rotation, console output,
    and error tracking capabilities.
    """
    
    def __init__(self, app_name: str = "StockPatternDetector"):
        """
        Initialize the application logger.
        
        Args:
            app_name: Name of the application for log identification
        """
        self.app_name = app_name
        self.log_directory = Path("logs")
        self.log_directory.mkdir(exist_ok=True)
        
        # Create main logger
        self.logger = logging.getLogger(app_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
        
        # Error tracking
        self.error_count = 0
        self.warning_count = 0
        self.last_errors = []
        self.max_stored_errors = 50
    
    def _setup_handlers(self) -> None:
        """Set up logging handlers for console and file output."""
        try:
            # Console handler with colors
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = ColoredFormatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler with rotation
            log_file = self.log_directory / f"{self.app_name.lower()}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            
            # Error file handler for errors and critical issues
            error_file = self.log_directory / f"{self.app_name.lower()}_errors.log"
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=5*1024*1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            self.logger.addHandler(error_handler)
            
        except Exception as e:
            # Fallback to basic console logging if file setup fails
            basic_handler = logging.StreamHandler(sys.stdout)
            basic_handler.setLevel(logging.INFO)
            basic_formatter = logging.Formatter('%(levelname)s - %(message)s')
            basic_handler.setFormatter(basic_formatter)
            self.logger.addHandler(basic_handler)
            self.logger.error(f"Failed to setup advanced logging: {e}")
    
    def get_logger(self, module_name: str) -> logging.Logger:
        """
        Get a logger for a specific module.
        
        Args:
            module_name: Name of the module requesting the logger
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(f"{self.app_name}.{module_name}")
    
    def log_error(self, error: Exception, context: str = "", extra_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with full context and tracking.
        
        Args:
            error: Exception that occurred
            context: Additional context about where the error occurred
            extra_data: Additional data to include in the log
        """
        try:
            self.error_count += 1
            
            # Create error record
            error_record = {
                'timestamp': datetime.now(),
                'error_type': type(error).__name__,
                'error_message': str(error),
                'context': context,
                'traceback': traceback.format_exc(),
                'extra_data': extra_data or {}
            }
            
            # Store for tracking
            self.last_errors.append(error_record)
            if len(self.last_errors) > self.max_stored_errors:
                self.last_errors.pop(0)
            
            # Log the error
            error_msg = f"ERROR in {context}: {error}"
            if extra_data:
                error_msg += f" | Extra data: {extra_data}"
            
            self.logger.error(error_msg, exc_info=True)
            
        except Exception as log_error:
            # Fallback logging if error logging fails
            print(f"CRITICAL: Failed to log error: {log_error}")
            print(f"Original error: {error}")
    
    def log_warning(self, message: str, context: str = "", extra_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Log a warning with context tracking.
        
        Args:
            message: Warning message
            context: Context where warning occurred
            extra_data: Additional data to include
        """
        try:
            self.warning_count += 1
            
            warning_msg = f"WARNING in {context}: {message}"
            if extra_data:
                warning_msg += f" | Extra data: {extra_data}"
            
            self.logger.warning(warning_msg)
            
        except Exception as e:
            print(f"Failed to log warning: {e}")
    
    def log_performance(self, operation: str, duration: float, context: str = "") -> None:
        """
        Log performance metrics for operations.
        
        Args:
            operation: Name of the operation
            duration: Duration in seconds
            context: Additional context
        """
        try:
            perf_msg = f"PERFORMANCE - {operation}: {duration:.3f}s"
            if context:
                perf_msg += f" | Context: {context}"
            
            # Log as info for normal operations, warning for slow operations
            if duration > 10.0:  # More than 10 seconds
                self.logger.warning(f"SLOW {perf_msg}")
            elif duration > 5.0:  # More than 5 seconds
                self.logger.info(f"MODERATE {perf_msg}")
            else:
                self.logger.debug(perf_msg)
                
        except Exception as e:
            print(f"Failed to log performance: {e}")
    
    def get_error_summary(self) -> Dict[str, Any]:
        """
        Get summary of recent errors and warnings.
        
        Returns:
            Dictionary with error statistics and recent errors
        """
        try:
            return {
                'total_errors': self.error_count,
                'total_warnings': self.warning_count,
                'recent_errors': [
                    {
                        'timestamp': err['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                        'type': err['error_type'],
                        'message': err['error_message'],
                        'context': err['context']
                    }
                    for err in self.last_errors[-10:]  # Last 10 errors
                ],
                'log_directory': str(self.log_directory)
            }
        except Exception as e:
            return {
                'error': f"Failed to generate error summary: {e}",
                'total_errors': self.error_count,
                'total_warnings': self.warning_count
            }
    
    def clear_error_tracking(self) -> None:
        """Clear error tracking counters and stored errors."""
        try:
            self.error_count = 0
            self.warning_count = 0
            self.last_errors.clear()
            self.logger.info("Error tracking cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear error tracking: {e}")
    
    def set_log_level(self, level: str) -> None:
        """
        Set the logging level for the application.
        
        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        try:
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f'Invalid log level: {level}')
            
            self.logger.setLevel(numeric_level)
            self.logger.info(f"Log level set to {level.upper()}")
            
        except Exception as e:
            self.logger.error(f"Failed to set log level: {e}")


# Global logger instance
app_logger = ApplicationLogger()


def get_logger(module_name: str) -> logging.Logger:
    """
    Convenience function to get a logger for a module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        Configured logger instance
    """
    return app_logger.get_logger(module_name)


def log_error(error: Exception, context: str = "", extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to log errors.
    
    Args:
        error: Exception that occurred
        context: Context where error occurred
        extra_data: Additional data to include
    """
    app_logger.log_error(error, context, extra_data)


def log_warning(message: str, context: str = "", extra_data: Optional[Dict[str, Any]] = None) -> None:
    """
    Convenience function to log warnings.
    
    Args:
        message: Warning message
        context: Context where warning occurred
        extra_data: Additional data to include
    """
    app_logger.log_warning(message, context, extra_data)


def log_performance(operation: str, duration: float, context: str = "") -> None:
    """
    Convenience function to log performance metrics.
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        context: Additional context
    """
    app_logger.log_performance(operation, duration, context)


def get_error_summary() -> Dict[str, Any]:
    """
    Get summary of application errors and warnings.
    
    Returns:
        Dictionary with error statistics
    """
    return app_logger.get_error_summary()