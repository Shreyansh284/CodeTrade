"""
Data aggregation module for stock pattern detector.

This module provides functionality to aggregate minute-level stock data
into different timeframes using OHLCV aggregation rules.
"""

import logging
from typing import Dict, Optional
import pandas as pd
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        try:
            if timeframe not in self.TIMEFRAMES:
                logger.error(f"Unsupported timeframe: {timeframe}. Supported: {list(self.TIMEFRAMES.keys())}")
                return None
            
            if data is None or data.empty:
                logger.warning("Empty data provided for aggregation")
                return None
            
            # Validate required columns
            required_columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                logger.error(f"Missing required columns for aggregation: {missing_columns}")
                return None
            
            # For 1min, return original data (no aggregation needed)
            if timeframe == '1min':
                logger.info("No aggregation needed for 1min timeframe")
                return data.copy()
            
            # Set datetime as index for resampling
            df = data.copy()
            df = df.set_index('datetime')
            
            # Get pandas frequency string
            freq = self.TIMEFRAMES[timeframe]
            
            # Perform aggregation using OHLCV rules
            aggregated = self._perform_ohlcv_aggregation(df, freq)
            
            if aggregated is None or aggregated.empty:
                logger.warning(f"No data after aggregation to {timeframe}")
                return None
            
            # Reset index to get datetime back as column
            aggregated = aggregated.reset_index()
            
            logger.info(f"Successfully aggregated {len(data)} records to {len(aggregated)} {timeframe} candles")
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating data to {timeframe}: {e}")
            return None
    
    def _perform_ohlcv_aggregation(self, df: pd.DataFrame, freq: str) -> Optional[pd.DataFrame]:
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
                if col in ['oi', 'exchangecode', 'symbolcode']:
                    agg_rules[col] = 'last'  # Take last value for metadata columns
                elif col == 'expiry':
                    agg_rules[col] = 'first'  # Take first value for expiry
            
            # Perform resampling and aggregation
            resampled = df.resample(freq).agg(agg_rules)
            
            # Remove rows where all OHLCV values are NaN (no data in that period)
            resampled = resampled.dropna(subset=['open', 'high', 'low', 'close'])
            
            # Validate aggregated data integrity
            resampled = self._validate_aggregated_data(resampled)
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error performing OHLCV aggregation: {e}")
            return None
    
    def _validate_aggregated_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate aggregated data integrity and fix any issues.
        
        Args:
            df: Aggregated DataFrame
            
        Returns:
            Validated DataFrame
        """
        try:
            initial_count = len(df)
            
            # Ensure OHLC relationships are maintained
            # High should be >= Open, Close, Low
            # Low should be <= Open, Close, High
            valid_mask = (
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['low']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close']) &
                (df['low'] <= df['high']) &
                (df['volume'] >= 0)
            )
            
            df = df[valid_mask]
            
            # Log any data loss
            final_count = len(df)
            if final_count < initial_count:
                lost_count = initial_count - final_count
                logger.warning(f"Removed {lost_count} invalid aggregated candles")
            
            return df
            
        except Exception as e:
            logger.error(f"Error validating aggregated data: {e}")
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