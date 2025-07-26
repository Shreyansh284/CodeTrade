"""
Data aggregation module for stock pattern detector.

This module provides functionality to aggregate minute-level stock data
into different timeframes using OHLCV aggregation rules with comprehensive error handling.
"""

import time
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

from utils.logging_config import get_logger
from utils.error_handler import (
    DataValidationError, with_error_handling, error_handler, safe_execute
)

logger = get_logger(__name__)


class DataAggregator:
    """
    Handles aggregation of minute-level stock data into different timeframes.
    
    Supports aggregation to: 1min, 5min, 15min, 1hour, 2hour, 5hour, 1day
    Uses OHLCV aggregation rules: first open, max high, min low, last close, sum volume
    """
    
    # Supported timeframes and their pandas frequency strings
    TIMEFRAMES = {
        '1min': '1min',    # 1 minute
        '5min': '5min',    # 5 minutes
        '15min': '15min',  # 15 minutes
        '1hour': '1h',     # 1 hour
        '2hour': '2h',     # 2 hours
        '5hour': '5h',     # 5 hours
        '1day': '1D'       # 1 day
    }
    
    def __init__(self):
        """Initialize the data aggregator."""
        pass
    
    def get_supported_timeframes(self) -> list:
        """
        Get list of supported timeframes.
        
        Returns:
            List of supported timeframe strings
        """
        return list(self.TIMEFRAMES.keys())
    
    def aggregate_data(self, data: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Aggregate minute-level data to specified timeframe.
        
        Args:
            data: DataFrame with minute-level OHLCV data
            timeframe: Target timeframe ('1min', '5min', '15min', '1hour', '2hour', '5hour', '1day')
            
        Returns:
            Aggregated DataFrame or None if error
        """
        start_time = time.time()
        
        try:
            # Validate timeframe
            if timeframe not in self.TIMEFRAMES:
                raise DataValidationError(
                    f"Unsupported timeframe: {timeframe}",
                    error_code="INVALID_TIMEFRAME",
                    context={
                        'requested_timeframe': timeframe,
                        'supported_timeframes': list(self.TIMEFRAMES.keys())
                    }
                )
            
            # Validate input data
            if data is None or data.empty:
                raise DataValidationError(
                    "Empty or None data provided for aggregation",
                    error_code="EMPTY_DATA_FOR_AGGREGATION",
                    context={'timeframe': timeframe}
                )
            
            # Show progress for longer operations
            progress_placeholder = st.empty()
            if timeframe != '1min':
                progress_placeholder.info(f"⚙️ Aggregating data to {timeframe} timeframe...")
            
            # Validate required columns
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns for aggregation: {missing_columns}",
                    error_code="MISSING_AGGREGATION_COLUMNS",
                    context={
                        'missing_columns': missing_columns,
                        'available_columns': list(data.columns),
                        'required_columns': required_columns
                    }
                )
            
            # For 1min, return original data (no aggregation needed)
            if timeframe == '1min':
                progress_placeholder.empty()
                logger.info("No aggregation needed for 1min timeframe")
                return data.copy()
            
            # Validate data integrity before aggregation
            is_valid, issues = error_handler.validate_data_integrity(
                data, f"aggregate_data({timeframe})"
            )
            
            if not is_valid:
                error_handler.show_warning_message(
                    f"Data quality issues detected before aggregation to {timeframe}",
                    issues[:3]  # Show first 3 issues
                )
            
            # Prepare data for aggregation
            df = data.copy()
            
            # Ensure datetime is properly formatted
            if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                except Exception as e:
                    raise DataValidationError(
                        f"Could not convert datetime column for aggregation: {str(e)}",
                        error_code="DATETIME_CONVERSION_ERROR",
                        context={'timeframe': timeframe}
                    )
            
            # Sort by datetime to ensure proper aggregation
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Set datetime as index for resampling
            df = df.set_index('datetime')
            
            # Get pandas frequency string
            freq = self.TIMEFRAMES[timeframe]
            
            # Perform aggregation using OHLCV rules
            aggregated = self._perform_ohlcv_aggregation(df, freq, timeframe)
            
            if aggregated is None or aggregated.empty:
                raise DataValidationError(
                    f"No data produced after aggregation to {timeframe}",
                    error_code="NO_AGGREGATED_DATA",
                    context={
                        'timeframe': timeframe,
                        'input_records': len(data),
                        'frequency': freq
                    }
                )
            
            # Reset index to get datetime back as column
            aggregated = aggregated.reset_index()
            
            # Clear progress indicator
            progress_placeholder.empty()
            
            # Log performance and results
            duration = time.time() - start_time
            compression_ratio = len(data) / len(aggregated) if len(aggregated) > 0 else 0
            
            logger.info(
                f"Successfully aggregated {len(data)} records to {len(aggregated)} {timeframe} candles "
                f"(compression: {compression_ratio:.1f}x) in {duration:.2f}s"
            )
            
            return aggregated
            
        except DataValidationError:
            # Re-raise validation errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise DataValidationError(
                f"Unexpected error during aggregation to {timeframe}: {str(e)}",
                error_code="AGGREGATION_UNEXPECTED_ERROR",
                context={'timeframe': timeframe, 'input_records': len(data) if data is not None else 0}
            ) from e
        finally:
            # Ensure progress indicator is cleared
            try:
                progress_placeholder.empty()
            except:
                pass
    
    def _perform_ohlcv_aggregation(self, df: pd.DataFrame, freq: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Perform OHLCV aggregation using pandas resample.
        
        Aggregation rules:
        - Open: First value in period
        - High: Maximum value in period  
        - Low: Minimum value in period
        - Close: Last value in period
        - Volume: Sum of all volumes in period
        
        Args:
            df: DataFrame with datetime index
            freq: Pandas frequency string
            timeframe: Human-readable timeframe for error reporting
            
        Returns:
            Aggregated DataFrame or None if error
        """
        try:
            # Define aggregation rules
            agg_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }
            
            # Add other columns with appropriate aggregation
            other_columns = [col for col in df.columns if col not in agg_rules.keys()]
            for col in other_columns:
                try:
                    if col in ['oi', 'exchangecode', 'symbolcode']:
                        agg_rules[col] = 'last'  # Take last value for metadata columns
                    elif col == 'expiry':
                        agg_rules[col] = 'first'  # Take first value for expiry
                    else:
                        # For unknown columns, try to determine appropriate aggregation
                        if df[col].dtype in ['object', 'string']:
                            agg_rules[col] = 'last'
                        else:
                            agg_rules[col] = 'mean'  # Numeric columns get averaged
                except Exception as e:
                    logger.warning(f"Could not determine aggregation for column {col}: {e}")
                    continue
            
            # Perform resampling and aggregation
            try:
                resampled = df.resample(freq).agg(agg_rules)
            except Exception as e:
                raise DataValidationError(
                    f"Failed to resample data to {timeframe}: {str(e)}",
                    error_code="RESAMPLING_FAILED",
                    context={
                        'timeframe': timeframe,
                        'frequency': freq,
                        'input_shape': df.shape,
                        'aggregation_rules': list(agg_rules.keys())
                    }
                ) from e
            
            # Check if resampling produced any data
            if resampled.empty:
                raise DataValidationError(
                    f"Resampling to {timeframe} produced no data",
                    error_code="EMPTY_RESAMPLED_DATA",
                    context={
                        'timeframe': timeframe,
                        'frequency': freq,
                        'input_records': len(df)
                    }
                )
            
            # Remove rows where all OHLCV values are NaN (no data in that period)
            before_dropna = len(resampled)
            resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
            after_dropna = len(resampled)
            
            if before_dropna != after_dropna:
                dropped = before_dropna - after_dropna
                logger.info(f"Dropped {dropped} empty periods during {timeframe} aggregation")
            
            if resampled.empty:
                raise DataValidationError(
                    f"No valid data remaining after removing empty periods for {timeframe}",
                    error_code="NO_VALID_PERIODS",
                    context={'timeframe': timeframe, 'empty_periods_dropped': dropped}
                )
            
            # Validate aggregated data integrity
            resampled = self._validate_aggregated_data(resampled, timeframe)
            
            return resampled
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"Unexpected error performing OHLCV aggregation to {timeframe}: {str(e)}",
                error_code="AGGREGATION_UNEXPECTED_ERROR",
                context={'timeframe': timeframe, 'frequency': freq}
            ) from e
    
    def _validate_aggregated_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Validate aggregated data integrity and fix any issues.
        
        Args:
            df: Aggregated DataFrame
            timeframe: Timeframe for error reporting
            
        Returns:
            Validated DataFrame
        """
        try:
            initial_count = len(df)
            validation_issues = []
            
            # Check for NaN values in critical columns
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in critical_columns:
                if col in df.columns:
                    nan_count = df[col].isnull().sum()
                    if nan_count > 0:
                        validation_issues.append(f"{col} has {nan_count} NaN values")
            
            # Remove rows with NaN in critical columns
            df = df.dropna(subset=[col for col in critical_columns if col in df.columns])
            
            # Ensure OHLC relationships are maintained
            if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
                valid_ohlc_mask = (
                    (df['high'] >= df['open']) &
                    (df['high'] >= df['close']) &
                    (df['high'] >= df['low']) &
                    (df['low'] <= df['open']) &
                    (df['low'] <= df['close']) &
                    (df['low'] <= df['high'])
                )
                
                invalid_ohlc_count = (~valid_ohlc_mask).sum()
                if invalid_ohlc_count > 0:
                    validation_issues.append(f"{invalid_ohlc_count} candles have invalid OHLC relationships")
                    df = df[valid_ohlc_mask]
            
            # Ensure volume is non-negative
            if 'volume' in df.columns:
                negative_volume_mask = df['volume'] < 0
                negative_volume_count = negative_volume_mask.sum()
                if negative_volume_count > 0:
                    validation_issues.append(f"{negative_volume_count} candles have negative volume")
                    df = df[~negative_volume_mask]
            
            # Check for zero-range candles (high == low)
            if all(col in df.columns for col in ['high', 'low']):
                zero_range_mask = df['high'] == df['low']
                zero_range_count = zero_range_mask.sum()
                if zero_range_count > 0:
                    validation_issues.append(f"{zero_range_count} candles have zero price range")
                    # Don't remove these as they can be valid (e.g., no trading activity)
            
            # Log validation results
            final_count = len(df)
            if final_count < initial_count:
                lost_count = initial_count - final_count
                logger.warning(
                    f"Removed {lost_count} invalid aggregated candles for {timeframe}: "
                    f"{'; '.join(validation_issues)}"
                )
            
            if validation_issues:
                logger.info(f"Validation issues found in {timeframe} aggregation: {'; '.join(validation_issues)}")
            
            # Check if we have any data left
            if df.empty:
                raise DataValidationError(
                    f"No valid data remaining after validation for {timeframe}",
                    error_code="NO_VALID_AGGREGATED_DATA",
                    context={
                        'timeframe': timeframe,
                        'initial_count': initial_count,
                        'validation_issues': validation_issues
                    }
                )
            
            return df
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating aggregated data for {timeframe}: {e}")
            # Return original data if validation fails
            return df
    
    def get_aggregation_info(self, original_data: pd.DataFrame, timeframe: str) -> Optional[Dict]:
        """
        Get information about what aggregation would produce.
        
        Args:
            original_data: Original minute-level data
            timeframe: Target timeframe
            
        Returns:
            Dictionary with aggregation information
        """
        try:
            if original_data is None or original_data.empty:
                return None
            
            aggregated = self.aggregate_data(original_data, timeframe)
            if aggregated is None:
                return None
            
            return {
                'original_records': len(original_data),
                'aggregated_records': len(aggregated),
                'compression_ratio': len(original_data) / len(aggregated),
                'timeframe': timeframe,
                'date_range': {
                    'start': aggregated['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': aggregated['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting aggregation info: {e}")
            return None
    
    def aggregate_multiple_timeframes(self, data: pd.DataFrame, timeframes: list) -> Dict[str, pd.DataFrame]:
        """
        Aggregate data to multiple timeframes at once.
        
        Args:
            data: Original minute-level data
            timeframes: List of timeframes to aggregate to
            
        Returns:
            Dictionary mapping timeframe to aggregated DataFrame
        """
        results = {}
        
        for timeframe in timeframes:
            try:
                aggregated = self.aggregate_data(data, timeframe)
                if aggregated is not None:
                    results[timeframe] = aggregated
                else:
                    logger.warning(f"Failed to aggregate to {timeframe}")
            except Exception as e:
                logger.error(f"Error aggregating to {timeframe}: {e}")
        
        return results