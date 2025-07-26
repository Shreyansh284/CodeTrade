"""
CSV data loader module for stock pattern detector.

This module provides functionality to load and validate stock data from CSV files
in the 5Scripts directory structure.
"""

import os
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CSVLoader:
    """
    Handles loading and validation of stock data from CSV files.
    
    The loader scans the 5Scripts directory for available instruments and
    loads their historical data with proper date/time parsing and validation.
    """
    
    def __init__(self, data_directory: str = "5Scripts"):
        """
        Initialize the CSV loader.
        
        Args:
            data_directory: Path to the directory containing instrument folders
        """
        self.data_directory = Path(data_directory)
        self.required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        
    def get_available_instruments(self) -> List[str]:
        """
        Scan the data directory and return list of available instruments.
        
        Returns:
            List of instrument names (folder names in 5Scripts directory)
        """
        try:
            if not self.data_directory.exists():
                logger.error(f"Data directory {self.data_directory} does not exist")
                return []
            
            instruments = []
            for item in self.data_directory.iterdir():
                if item.is_dir():
                    instruments.append(item.name)
            
            logger.info(f"Found {len(instruments)} instruments: {instruments}")
            return sorted(instruments)
            
        except Exception as e:
            logger.error(f"Error scanning data directory: {e}")
            return []
    
    def load_instrument_data(self, instrument: str) -> Optional[pd.DataFrame]:
        """
        Load all CSV files for a specific instrument and combine into single DataFrame.
        
        Args:
            instrument: Name of the instrument (folder name)
            
        Returns:
            Combined DataFrame with all data for the instrument, or None if error
        """
        try:
            instrument_path = self.data_directory / instrument
            
            if not instrument_path.exists():
                logger.error(f"Instrument directory {instrument_path} does not exist")
                return None
            
            csv_files = list(instrument_path.glob("*.csv"))
            if not csv_files:
                logger.warning(f"No CSV files found for instrument {instrument}")
                return None
            
            logger.info(f"Loading {len(csv_files)} CSV files for {instrument}")
            
            dataframes = []
            for csv_file in csv_files:
                df = self._load_single_csv(csv_file)
                if df is not None:
                    dataframes.append(df)
            
            if not dataframes:
                logger.error(f"No valid CSV files loaded for {instrument}")
                return None
            
            # Combine all dataframes
            combined_df = pd.concat(dataframes, ignore_index=True)
            
            # Sort by datetime
            combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            
            logger.info(f"Successfully loaded {len(combined_df)} records for {instrument}")
            return combined_df
            
        except Exception as e:
            logger.error(f"Error loading data for instrument {instrument}: {e}")
            return None
    
    def _load_single_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate a single CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            Validated DataFrame or None if error
        """
        try:
            # Load CSV file
            df = pd.read_csv(csv_file)
            
            # Validate required columns
            if not self._validate_columns(df, csv_file):
                return None
            
            # Parse datetime
            df = self._parse_datetime(df, csv_file)
            if df is None:
                return None
            
            # Validate and clean data
            df = self._validate_and_clean_data(df, csv_file)
            if df is None:
                return None
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            return None
    
    def _validate_columns(self, df: pd.DataFrame, csv_file: Path) -> bool:
        """
        Validate that all required columns exist in the DataFrame.
        
        Args:
            df: DataFrame to validate
            csv_file: Path to the CSV file (for logging)
            
        Returns:
            True if all required columns exist, False otherwise
        """
        missing_columns = [col for col in self.required_columns if col not in df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns in {csv_file}: {missing_columns}")
            return False
        
        return True
    
    def _parse_datetime(self, df: pd.DataFrame, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Parse date and time columns into a single datetime column.
        
        Expected formats:
        - Date: DD-MM-YYYY
        - Time: HH:MM:SS
        
        Args:
            df: DataFrame with date and time columns
            csv_file: Path to the CSV file (for logging)
            
        Returns:
            DataFrame with datetime column or None if parsing fails
        """
        try:
            # Combine date and time columns
            datetime_str = df['date'] + ' ' + df['time']
            
            # Parse datetime with DD-MM-YYYY HH:MM:SS format
            df['datetime'] = pd.to_datetime(datetime_str, format='%d-%m-%Y %H:%M:%S')
            
            # Drop original date and time columns
            df = df.drop(['date', 'time'], axis=1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error parsing datetime in {csv_file}: {e}")
            return None
    
    def _validate_and_clean_data(self, df: pd.DataFrame, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Validate OHLCV data integrity and clean invalid records.
        
        Args:
            df: DataFrame to validate
            csv_file: Path to the CSV file (for logging)
            
        Returns:
            Cleaned DataFrame or None if too much invalid data
        """
        try:
            initial_count = len(df)
            
            # Remove rows with missing OHLCV data
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            
            # Validate OHLCV relationships
            # High should be >= Open, Close, Low
            # Low should be <= Open, Close, High
            valid_ohlc = (
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['low']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close']) &
                (df['low'] <= df['high'])
            )
            
            df = df[valid_ohlc]
            
            # Remove rows with negative or zero prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df = df[df[col] > 0]
            
            # Remove rows with negative volume
            df = df[df['volume'] >= 0]
            
            # Check if we lost too much data
            final_count = len(df)
            data_loss_pct = (initial_count - final_count) / initial_count * 100
            
            if data_loss_pct > 50:
                logger.error(f"Too much invalid data in {csv_file}: {data_loss_pct:.1f}% loss")
                return None
            
            if data_loss_pct > 0:
                logger.warning(f"Cleaned {data_loss_pct:.1f}% invalid records from {csv_file}")
            
            return df.reset_index(drop=True)
            
        except Exception as e:
            logger.error(f"Error validating data in {csv_file}: {e}")
            return None
    
    def get_data_info(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about an instrument's data.
        
        Args:
            instrument: Name of the instrument
            
        Returns:
            Dictionary with data summary information
        """
        try:
            df = self.load_instrument_data(instrument)
            if df is None:
                return None
            
            return {
                'instrument': instrument,
                'total_records': len(df),
                'date_range': {
                    'start': df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end': df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
                },
                'price_range': {
                    'min': float(df['low'].min()),
                    'max': float(df['high'].max())
                },
                'total_volume': int(df['volume'].sum())
            }
            
        except Exception as e:
            logger.error(f"Error getting data info for {instrument}: {e}")
            return None