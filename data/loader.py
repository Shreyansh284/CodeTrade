"""
CSV data loader module for stock pattern detector.

This module provides functionality to load and validate stock data from CSV files
in the 5Scripts directory structure with comprehensive error handling.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import streamlit as st

from utils.logging_config import get_logger
from utils.error_handler import (
    DataLoadingError, DataValidationError, with_error_handling,
    error_handler, safe_execute
)

logger = get_logger(__name__)


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
        
    @with_error_handling(
        error_types=(OSError, PermissionError, Exception),
        context="get_available_instruments",
        fallback_result=[]
    )
    def get_available_instruments(self) -> List[str]:
        """
        Scan the data directory and return list of available instruments.
        
        Returns:
            List of instrument names (folder names in 5Scripts directory)
        """
        if not self.data_directory.exists():
            raise DataLoadingError(
                f"Data directory {self.data_directory} does not exist",
                error_code="DATA_DIRECTORY_NOT_FOUND",
                context={'directory': str(self.data_directory)}
            )
        
        instruments = []
        failed_directories = []
        
        for item in self.data_directory.iterdir():
            try:
                if item.is_dir():
                    # Verify directory is accessible and contains CSV files
                    csv_files = list(item.glob("*.csv"))
                    if csv_files:
                        instruments.append(item.name)
                    else:
                        failed_directories.append(f"{item.name} (no CSV files)")
            except (OSError, PermissionError) as e:
                failed_directories.append(f"{item.name} (access denied)")
                logger.warning(f"Cannot access directory {item.name}: {e}")
        
        if failed_directories:
            error_handler.show_warning_message(
                f"Some instrument directories could not be accessed",
                [f"Skipped: {dir_name}" for dir_name in failed_directories]
            )
        
        if not instruments:
            raise DataLoadingError(
                "No valid instrument directories found with CSV files",
                error_code="NO_INSTRUMENTS_FOUND",
                context={'directory': str(self.data_directory), 'failed_dirs': failed_directories}
            )
        
        logger.info(f"Found {len(instruments)} valid instruments: {instruments}")
        return sorted(instruments)
    
    def load_instrument_data(self, instrument: str) -> Optional[pd.DataFrame]:
        """
        Load all CSV files for a specific instrument and combine into single DataFrame.
        
        Args:
            instrument: Name of the instrument (folder name)
            
        Returns:
            Combined DataFrame with all data for the instrument, or None if error
        """
        start_time = time.time()
        
        try:
            # Validate input
            if not instrument or not isinstance(instrument, str):
                raise DataLoadingError(
                    "Invalid instrument name provided",
                    error_code="INVALID_INSTRUMENT_NAME",
                    context={'instrument': instrument}
                )
            
            instrument_path = self.data_directory / instrument
            
            if not instrument_path.exists():
                raise DataLoadingError(
                    f"Instrument directory '{instrument}' does not exist",
                    error_code="INSTRUMENT_NOT_FOUND",
                    context={'instrument': instrument, 'path': str(instrument_path)}
                )
            
            # Show progress indicator
            progress_placeholder = st.empty()
            progress_placeholder.info(f"📥 Loading data for {instrument}...")
            
            csv_files = list(instrument_path.glob("*.csv"))
            if not csv_files:
                raise DataLoadingError(
                    f"No CSV files found for instrument '{instrument}'",
                    error_code="NO_CSV_FILES",
                    context={'instrument': instrument, 'path': str(instrument_path)}
                )
            
            logger.info(f"Loading {len(csv_files)} CSV files for {instrument}")
            
            # Load files with progress tracking
            dataframes = []
            failed_files = []
            
            for i, csv_file in enumerate(csv_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(csv_files)
                    progress_placeholder.info(
                        f"📥 Loading {instrument}: {i+1}/{len(csv_files)} files ({progress:.0%})"
                    )
                    
                    df = self._load_single_csv(csv_file)
                    if df is not None and not df.empty:
                        dataframes.append(df)
                    else:
                        failed_files.append(csv_file.name)
                        
                except Exception as e:
                    failed_files.append(f"{csv_file.name} ({str(e)[:50]}...)")
                    logger.warning(f"Failed to load {csv_file.name}: {e}")
            
            # Clear progress indicator
            progress_placeholder.empty()
            
            if not dataframes:
                raise DataLoadingError(
                    f"No valid CSV files could be loaded for instrument '{instrument}'",
                    error_code="ALL_FILES_FAILED",
                    context={
                        'instrument': instrument,
                        'total_files': len(csv_files),
                        'failed_files': failed_files
                    }
                )
            
            # Show warnings for failed files
            if failed_files:
                error_handler.show_warning_message(
                    f"Some files could not be loaded for {instrument}",
                    [f"Failed: {file}" for file in failed_files[:5]]  # Show first 5
                )
            
            # Combine all dataframes
            try:
                combined_df = pd.concat(dataframes, ignore_index=True)
            except Exception as e:
                raise DataLoadingError(
                    f"Failed to combine data files for '{instrument}'",
                    error_code="DATA_COMBINATION_FAILED",
                    context={'instrument': instrument, 'dataframe_count': len(dataframes)}
                )
            
            # Sort by datetime
            try:
                combined_df = combined_df.sort_values('datetime').reset_index(drop=True)
            except Exception as e:
                logger.warning(f"Could not sort data by datetime: {e}")
                # Continue without sorting
            
            # Validate final data
            is_valid, issues = error_handler.validate_data_integrity(
                combined_df, f"load_instrument_data({instrument})"
            )
            
            if not is_valid:
                error_handler.show_warning_message(
                    f"Data quality issues detected for {instrument}",
                    issues[:5]  # Show first 5 issues
                )
            
            # Log performance
            duration = time.time() - start_time
            logger.info(
                f"Successfully loaded {len(combined_df)} records for {instrument} "
                f"from {len(dataframes)}/{len(csv_files)} files in {duration:.2f}s"
            )
            
            return combined_df
            
        except DataLoadingError:
            # Re-raise data loading errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise DataLoadingError(
                f"Unexpected error loading data for '{instrument}': {str(e)}",
                error_code="UNEXPECTED_LOAD_ERROR",
                context={'instrument': instrument}
            ) from e
    
    def _load_single_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """
        Load and validate a single CSV file.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            Validated DataFrame or None if error
        """
        try:
            # Check file accessibility
            if not csv_file.exists():
                raise DataLoadingError(
                    f"CSV file does not exist: {csv_file.name}",
                    error_code="FILE_NOT_FOUND"
                )
            
            if csv_file.stat().st_size == 0:
                logger.warning(f"Empty CSV file: {csv_file.name}")
                return None
            
            # Load CSV file with error handling
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
            except UnicodeDecodeError:
                # Try different encodings
                for encoding in ['latin-1', 'cp1252', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(csv_file, encoding=encoding)
                        logger.info(f"Loaded {csv_file.name} with {encoding} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise DataLoadingError(
                        f"Could not decode CSV file with any supported encoding: {csv_file.name}",
                        error_code="ENCODING_ERROR"
                    )
            
            if df.empty:
                logger.warning(f"CSV file contains no data: {csv_file.name}")
                return None
            
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
            
        except DataLoadingError:
            # Re-raise data loading errors
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading CSV file {csv_file.name}: {e}")
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
        try:
            # Check for required columns (case-insensitive)
            df_columns_lower = [col.lower().strip() for col in df.columns]
            required_columns_lower = [col.lower() for col in self.required_columns]
            
            missing_columns = []
            column_mapping = {}
            
            for req_col in self.required_columns:
                req_col_lower = req_col.lower()
                found = False
                
                for i, df_col_lower in enumerate(df_columns_lower):
                    if df_col_lower == req_col_lower:
                        column_mapping[req_col] = df.columns[i]
                        found = True
                        break
                
                if not found:
                    missing_columns.append(req_col)
            
            if missing_columns:
                raise DataValidationError(
                    f"Missing required columns in {csv_file.name}: {missing_columns}",
                    error_code="MISSING_COLUMNS",
                    context={
                        'file': csv_file.name,
                        'missing_columns': missing_columns,
                        'available_columns': list(df.columns),
                        'required_columns': self.required_columns
                    }
                )
            
            # Rename columns to standard names if needed
            if column_mapping:
                rename_map = {v: k for k, v in column_mapping.items() if v != k}
                if rename_map:
                    df.rename(columns=rename_map, inplace=True)
                    logger.debug(f"Renamed columns in {csv_file.name}: {rename_map}")
            
            return True
            
        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Error validating columns in {csv_file.name}: {e}")
            return False
    
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
            # Check if datetime column already exists
            if 'datetime' in df.columns:
                try:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    return df
                except Exception as e:
                    logger.warning(f"Could not parse existing datetime column in {csv_file.name}: {e}")
            
            # Combine date and time columns
            if 'date' not in df.columns or 'time' not in df.columns:
                raise DataValidationError(
                    f"Missing date or time columns in {csv_file.name}",
                    error_code="MISSING_DATETIME_COLUMNS",
                    context={'file': csv_file.name, 'columns': list(df.columns)}
                )
            
            # Clean date and time strings
            df['date'] = df['date'].astype(str).str.strip()
            df['time'] = df['time'].astype(str).str.strip()
            
            # Combine date and time
            datetime_str = df['date'] + ' ' + df['time']
            
            # Try multiple datetime formats
            datetime_formats = [
                '%d-%m-%Y %H:%M:%S',    # DD-MM-YYYY HH:MM:SS
                '%d/%m/%Y %H:%M:%S',    # DD/MM/YYYY HH:MM:SS
                '%Y-%m-%d %H:%M:%S',    # YYYY-MM-DD HH:MM:SS
                '%d-%m-%Y %H:%M',       # DD-MM-YYYY HH:MM
                '%d/%m/%Y %H:%M',       # DD/MM/YYYY HH:MM
            ]
            
            parsed_datetime = None
            successful_format = None
            
            for fmt in datetime_formats:
                try:
                    parsed_datetime = pd.to_datetime(datetime_str, format=fmt)
                    successful_format = fmt
                    break
                except (ValueError, TypeError):
                    continue
            
            if parsed_datetime is None:
                # Try pandas' automatic parsing as last resort
                try:
                    parsed_datetime = pd.to_datetime(datetime_str, infer_datetime_format=True)
                    successful_format = "auto-detected"
                except Exception:
                    raise DataValidationError(
                        f"Could not parse datetime in {csv_file.name}",
                        error_code="DATETIME_PARSE_ERROR",
                        context={
                            'file': csv_file.name,
                            'sample_datetime': datetime_str.iloc[0] if len(datetime_str) > 0 else "N/A",
                            'tried_formats': datetime_formats
                        }
                    )
            
            # Check for invalid dates
            if parsed_datetime.isnull().any():
                null_count = parsed_datetime.isnull().sum()
                logger.warning(f"Found {null_count} invalid datetime entries in {csv_file.name}")
                
                # Remove rows with invalid dates
                valid_mask = parsed_datetime.notnull()
                df = df[valid_mask].copy()
                parsed_datetime = parsed_datetime[valid_mask]
            
            df['datetime'] = parsed_datetime
            
            # Drop original date and time columns
            df = df.drop(['date', 'time'], axis=1)
            
            logger.debug(f"Successfully parsed datetime in {csv_file.name} using format: {successful_format}")
            return df
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"Unexpected error parsing datetime in {csv_file.name}: {str(e)}",
                error_code="DATETIME_PARSE_UNEXPECTED_ERROR",
                context={'file': csv_file.name}
            ) from e
    
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
            cleaning_log = []
            
            # Remove rows with missing OHLCV data
            before_na = len(df)
            df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
            after_na = len(df)
            if before_na != after_na:
                removed = before_na - after_na
                cleaning_log.append(f"Removed {removed} rows with missing OHLCV data")
            
            # Convert numeric columns to proper types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    logger.warning(f"Could not convert {col} to numeric in {csv_file.name}: {e}")
            
            # Remove rows where numeric conversion failed
            before_numeric = len(df)
            df = df.dropna(subset=numeric_columns)
            after_numeric = len(df)
            if before_numeric != after_numeric:
                removed = before_numeric - after_numeric
                cleaning_log.append(f"Removed {removed} rows with non-numeric OHLCV data")
            
            # Validate OHLCV relationships
            before_ohlc = len(df)
            valid_ohlc = (
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['high'] >= df['low']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close']) &
                (df['low'] <= df['high'])
            )
            df = df[valid_ohlc]
            after_ohlc = len(df)
            if before_ohlc != after_ohlc:
                removed = before_ohlc - after_ohlc
                cleaning_log.append(f"Removed {removed} rows with invalid OHLC relationships")
            
            # Remove rows with negative or zero prices
            before_prices = len(df)
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                df = df[df[col] > 0]
            after_prices = len(df)
            if before_prices != after_prices:
                removed = before_prices - after_prices
                cleaning_log.append(f"Removed {removed} rows with non-positive prices")
            
            # Remove rows with negative volume (allow zero volume)
            before_volume = len(df)
            df = df[df['volume'] >= 0]
            after_volume = len(df)
            if before_volume != after_volume:
                removed = before_volume - after_volume
                cleaning_log.append(f"Removed {removed} rows with negative volume")
            
            # Check for duplicate timestamps
            before_duplicates = len(df)
            df = df.drop_duplicates(subset=['datetime'], keep='first')
            after_duplicates = len(df)
            if before_duplicates != after_duplicates:
                removed = before_duplicates - after_duplicates
                cleaning_log.append(f"Removed {removed} duplicate timestamp rows")
            
            # Calculate data loss percentage
            final_count = len(df)
            data_loss_pct = (initial_count - final_count) / initial_count * 100 if initial_count > 0 else 0
            
            # Log cleaning results
            if cleaning_log:
                logger.info(f"Data cleaning for {csv_file.name}: {'; '.join(cleaning_log)}")
            
            # Check if we lost too much data
            if data_loss_pct > 75:
                raise DataValidationError(
                    f"Excessive data loss in {csv_file.name}: {data_loss_pct:.1f}% of data was invalid",
                    error_code="EXCESSIVE_DATA_LOSS",
                    context={
                        'file': csv_file.name,
                        'initial_count': initial_count,
                        'final_count': final_count,
                        'loss_percentage': data_loss_pct,
                        'cleaning_log': cleaning_log
                    }
                )
            
            if data_loss_pct > 25:
                logger.warning(f"Significant data loss in {csv_file.name}: {data_loss_pct:.1f}% of data was cleaned")
            elif data_loss_pct > 0:
                logger.info(f"Cleaned {data_loss_pct:.1f}% invalid records from {csv_file.name}")
            
            if final_count == 0:
                raise DataValidationError(
                    f"No valid data remaining after cleaning {csv_file.name}",
                    error_code="NO_VALID_DATA",
                    context={'file': csv_file.name, 'cleaning_log': cleaning_log}
                )
            
            return df.reset_index(drop=True)
            
        except DataValidationError:
            raise
        except Exception as e:
            raise DataValidationError(
                f"Unexpected error validating data in {csv_file.name}: {str(e)}",
                error_code="DATA_VALIDATION_UNEXPECTED_ERROR",
                context={'file': csv_file.name}
            ) from e
    
    @with_error_handling(
        error_types=(DataLoadingError, DataValidationError, Exception),
        context="get_data_info",
        fallback_result=None
    )
    def get_data_info(self, instrument: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about an instrument's data.
        
        Args:
            instrument: Name of the instrument
            
        Returns:
            Dictionary with data summary information
        """
        # Use cached data if available to avoid reloading
        df = safe_execute(
            self.load_instrument_data,
            instrument,
            context=f"get_data_info({instrument})",
            show_errors=False  # Don't show errors in info gathering
        )
        
        if df is None or df.empty:
            return {
                'instrument': instrument,
                'total_records': 0,
                'date_range': {'start': 'N/A', 'end': 'N/A'},
                'price_range': {'min': 0.0, 'max': 0.0},
                'total_volume': 0,
                'status': 'No data available'
            }
        
        try:
            # Calculate statistics safely
            total_records = len(df)
            
            # Date range
            date_range = {
                'start': df['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                'end': df['datetime'].max().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Price range
            price_range = {
                'min': float(df['low'].min()),
                'max': float(df['high'].max())
            }
            
            # Volume statistics
            total_volume = int(df['volume'].sum())
            avg_volume = int(df['volume'].mean())
            
            # Additional statistics
            price_change = float(df['close'].iloc[-1] - df['close'].iloc[0]) if len(df) > 1 else 0.0
            price_change_pct = (price_change / df['close'].iloc[0] * 100) if len(df) > 1 and df['close'].iloc[0] != 0 else 0.0
            
            return {
                'instrument': instrument,
                'total_records': total_records,
                'date_range': date_range,
                'price_range': price_range,
                'total_volume': total_volume,
                'avg_volume': avg_volume,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'status': 'Data loaded successfully'
            }
            
        except Exception as e:
            logger.warning(f"Error calculating statistics for {instrument}: {e}")
            return {
                'instrument': instrument,
                'total_records': len(df) if df is not None else 0,
                'date_range': {'start': 'Error', 'end': 'Error'},
                'price_range': {'min': 0.0, 'max': 0.0},
                'total_volume': 0,
                'status': f'Error calculating statistics: {str(e)[:50]}...'
            }