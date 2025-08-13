"""
CSV data loader module for stock pattern detector.

This module provides functionality to load and validate stock data from CSV files
in the 5Scripts directory structure with comprehensive error handling.
"""

import os
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import pandas as pd

# Optional Streamlit import for progress messages; fall back to no-op when unavailable
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # type: ignore

class _NoOpProgress:
    def info(self, *args, **kwargs):
        return None
    def empty(self):
        return None

from utils.logging_config import get_logger
from utils.error_handler import (
    DataLoadingError, DataValidationError, with_error_handling,
    error_handler, safe_execute
)
from utils.cache_manager import cache_manager, cached_function

logger = get_logger(__name__)


class CSVLoader:
    """CSV Loader supporting two directory paradigms.

    Modes:
    1. Legacy intraday mode ("5Scripts")
       - Directory structure: <root>/<INSTRUMENT>/<DD-MM-YYYY>.csv (per-day, per-instrument, intraday rows containing date+time columns)
    2. Flat daily mode ("StockData")
       - Directory structure: <root>/<INSTRUMENT>.csv (one file per instrument with daily OHLCV rows, single Date column)

    The loader auto-detects which mode to use based on the presence of CSV files
    directly under the root versus instrument subdirectories. Backwards compatible:
    If the requested directory does not exist but the legacy directory does, it
    falls back automatically.
    """

    def __init__(self, data_directory: str = "StockData"):
        """Initialize the CSV loader with auto mode detection.

        Args:
            data_directory: Preferred data directory (defaults to new StockData flat structure)
        """
        # Allow environment variable override if caller used default
        env_dir = os.getenv("DATA_DIR")
        if data_directory == "StockData" and env_dir:
            env_path = Path(env_dir)
            if env_path.exists():
                data_directory = env_dir
                logger.info(f"Using DATA_DIR environment override: {env_dir}")

        preferred = Path(data_directory)
        # Fallback to legacy directory if preferred doesn't exist
        if not preferred.exists() and data_directory == "StockData":
            legacy = Path("5Scripts")
            if legacy.exists():
                preferred = legacy
        self.data_directory = preferred

        # Mode detection: flat daily if CSVs at root level and (a) no subdirs with CSVs or (b) majority of entries are CSVs
        root_csvs = list(self.data_directory.glob("*.csv")) if self.data_directory.exists() else []
        subdir_with_csv = False
        for item in self.data_directory.iterdir() if self.data_directory.exists() else []:
            if item.is_dir() and list(item.glob("*.csv")):
                subdir_with_csv = True
                break
        self.flat_mode = len(root_csvs) > 0 and not subdir_with_csv

        # Standard required columns for legacy intraday mode
        self.required_columns = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        # Required columns for flat daily mode (after normalization to lowercase)
        self.required_daily_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        logger.info(f"CSVLoader initialized in {'FLAT DAILY' if self.flat_mode else 'LEGACY'} mode using directory: {self.data_directory}")
        
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

        if self.flat_mode:
            csv_files = list(self.data_directory.glob("*.csv"))
            if not csv_files:
                raise DataLoadingError(
                    "No CSV files found in flat daily data directory",
                    error_code="NO_DAILY_FILES",
                    context={'directory': str(self.data_directory)}
                )
            instruments = sorted([f.stem for f in csv_files])
            logger.info(f"Flat daily mode: {len(instruments)} instruments detected")
            return instruments

        # Legacy mode behaviour
        instruments = []
        failed_directories = []
        for item in self.data_directory.iterdir():
            try:
                if item.is_dir():
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
                "Some instrument directories could not be accessed",
                [f"Skipped: {dir_name}" for dir_name in failed_directories]
            )

        if not instruments:
            raise DataLoadingError(
                "No valid instrument directories found with CSV files",
                error_code="NO_INSTRUMENTS_FOUND",
                context={'directory': str(self.data_directory), 'failed_dirs': failed_directories}
            )

        logger.info(f"Legacy mode: Found {len(instruments)} valid instruments: {instruments}")
        return sorted(instruments)
    
    @with_error_handling(
        error_types=(OSError, PermissionError, Exception),
        context="get_available_dates",
        fallback_result=[]
    )
    def get_available_dates(self, instrument: str) -> List[datetime]:
        """
        Get available dates for a specific instrument by scanning CSV filenames.
        
        Args:
            instrument: Name of the instrument (folder name)
            
        Returns:
            List of datetime objects representing available dates
        """
        if not instrument or not isinstance(instrument, str):
            raise DataLoadingError(
                "Invalid instrument name provided",
                error_code="INVALID_INSTRUMENT_NAME",
                context={'instrument': instrument}
            )
        
        if self.flat_mode:
            file_path = self.data_directory / f"{instrument}.csv"
            if not file_path.exists():
                raise DataLoadingError(
                    f"Instrument file '{instrument}.csv' does not exist",
                    error_code="INSTRUMENT_FILE_NOT_FOUND",
                    context={'instrument': instrument, 'path': str(file_path)}
                )
            try:
                # Read only Date column for efficiency
                df_dates = pd.read_csv(file_path, usecols=['Date'])
            except ValueError:
                # Column may be lowercase or different; load full file then check
                df_full = pd.read_csv(file_path)
                date_col = None
                for cand in ['Date', 'date']:
                    if cand in df_full.columns:
                        date_col = cand
                        break
                if not date_col:
                    raise DataLoadingError(
                        f"No Date column found in {file_path.name}",
                        error_code="MISSING_DATE_COLUMN",
                        context={'columns': list(df_full.columns)}
                    )
                df_dates = df_full[[date_col]].rename(columns={date_col: 'Date'})
            # Parse dates (expecting ISO YYYY-MM-DD, fallback to DD-MM-YYYY)
            parsed = None
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']:
                try:
                    parsed = pd.to_datetime(df_dates['Date'], format=fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                parsed = pd.to_datetime(df_dates['Date'], errors='coerce', infer_datetime_format=True)
            parsed = parsed.dropna()
            dates = sorted(set(d.date() for d in parsed))
            if not dates:
                raise DataLoadingError(
                    f"Could not parse any valid dates for instrument '{instrument}'",
                    error_code="NO_VALID_DATES",
                    context={'instrument': instrument, 'file': str(file_path)}
                )
            logger.info(f"Flat mode: {instrument} has {len(dates)} trading days")
            return dates

        # Legacy behaviour
        instrument_path = self.data_directory / instrument
        if not instrument_path.exists():
            raise DataLoadingError(
                f"Instrument directory '{instrument}' does not exist",
                error_code="INSTRUMENT_NOT_FOUND",
                context={'instrument': instrument, 'path': str(instrument_path)}
            )
        csv_files = list(instrument_path.glob("*.csv"))
        if not csv_files:
            raise DataLoadingError(
                f"No CSV files found for instrument '{instrument}'",
                error_code="NO_CSV_FILES",
                context={'instrument': instrument, 'path': str(instrument_path)}
            )
        dates = []
        for csv_file in csv_files:
            try:
                filename = csv_file.stem
                date_obj = datetime.strptime(filename, '%d-%m-%Y')
                dates.append(date_obj.date())
            except ValueError as e:
                logger.warning(f"Could not parse date from filename {csv_file.name}: {e}")
        if not dates:
            raise DataLoadingError(
                f"No valid date files found for instrument '{instrument}'",
                error_code="NO_VALID_DATES",
                context={'instrument': instrument, 'path': str(instrument_path)}
            )
        logger.info(f"Legacy mode: Found {len(dates)} available dates for {instrument}")
        return sorted(dates)
    
    def load_instrument_data(self, instrument: str, start_date: Optional[datetime] = None, 
                            end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """
        Load all CSV files for a specific instrument and combine into single DataFrame.
        Uses caching to avoid reloading unchanged data.
        
        Args:
            instrument: Name of the instrument (folder name)
            start_date: Optional start date for filtering data (inclusive)
            end_date: Optional end date for filtering data (inclusive)
            
        Returns:
            Combined DataFrame with all data for the instrument, or None if error
        """
        start_time = time.time()
        # If in flat daily mode, delegate to flat file loader
        if self.flat_mode:
            return self._load_flat_daily_file(instrument, start_date, end_date)
        
        # Generate cache key including date range for proper caching
        date_suffix = ""
        if start_date or end_date:
            start_str = start_date.strftime('%Y%m%d') if start_date else "start"
            end_str = end_date.strftime('%Y%m%d') if end_date else "end"
            date_suffix = f"_{start_str}_{end_str}"
        
        cache_key = f"instrument_data_{instrument}_{self.data_directory.name}{date_suffix}"
        
        # Get list of CSV files for cache invalidation and date filtering
        instrument_path = self.data_directory / instrument
        
        # Filter CSV files based on date range if provided
        all_csv_files = []
        if instrument_path.exists():
            all_csv_files = list(instrument_path.glob("*.csv"))
        
        # Filter files by date range if specified
        csv_files = []
        if start_date or end_date:
            for csv_file in all_csv_files:
                try:
                    # Extract date from filename (format: DD-MM-YYYY.csv)
                    filename = csv_file.stem  # Remove .csv extension
                    file_date = datetime.strptime(filename, '%d-%m-%Y').date()
                    
                    # Check if file date is within range
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    csv_files.append(csv_file)
                except ValueError:
                    # Skip files with invalid date formats
                    logger.warning(f"Skipping file with invalid date format: {csv_file.name}")
                    continue
        else:
            csv_files = all_csv_files
        
        source_files = [str(f) for f in csv_files]
        
        # Try to get from cache first (only if no date filtering for now - full data cache)
        if not start_date and not end_date:
            cached_data = cache_manager.get(cache_key)
            if cached_data is not None:
                # Verify cache is still valid by checking file modification times
                cache_valid = True
                try:
                    cache_path = cache_manager._get_file_cache_path(cache_key)
                    if cache_path.exists():
                        cache_mtime = cache_path.stat().st_mtime
                        for source_file in source_files:
                            if Path(source_file).stat().st_mtime > cache_mtime:
                                cache_valid = False
                                break
                except Exception:
                    cache_valid = False
                
                if cache_valid:
                    cache_time = time.time() - start_time
                    logger.info(f"Loaded {instrument} from cache in {cache_time:.3f}s ({len(cached_data)} records)")
                    return cached_data
                else:
                    # Invalidate stale cache
                    cache_manager.invalidate(cache_key)
        
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
            date_info = ""
            if start_date and end_date:
                date_info = f" ({start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')})"
            elif start_date:
                date_info = f" (from {start_date.strftime('%d-%m-%Y')})"
            elif end_date:
                date_info = f" (until {end_date.strftime('%d-%m-%Y')})"
            
            # Progress UI (Streamlit) is optional
            progress_placeholder = st.empty() if st else _NoOpProgress()
            progress_placeholder.info(f"📥 Loading data for {instrument}{date_info}...")
            
            if not csv_files:
                if start_date or end_date:
                    raise DataLoadingError(
                        f"No CSV files found for instrument '{instrument}' in the specified date range",
                        error_code="NO_CSV_FILES_IN_RANGE",
                        context={
                            'instrument': instrument, 
                            'path': str(instrument_path),
                            'start_date': start_date,
                            'end_date': end_date
                        }
                    )
                else:
                    raise DataLoadingError(
                        f"No CSV files found for instrument '{instrument}'",
                        error_code="NO_CSV_FILES",
                        context={'instrument': instrument, 'path': str(instrument_path)}
                    )
            
            logger.info(f"Loading {len(csv_files)} CSV files for {instrument}{date_info}")
            
            # Load files with progress tracking
            dataframes = []
            failed_files = []
            
            for i, csv_file in enumerate(csv_files):
                try:
                    # Update progress
                    progress = (i + 1) / len(csv_files)
                    if st:
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
            if st:
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
            
            # Cache the result for future use
            cache_manager.set(
                cache_key, 
                combined_df, 
                ttl=1800,  # 30 minutes cache
                source_files=source_files
            )
            
            # Log performance
            duration = time.time() - start_time
            logger.info(
                f"Successfully loaded {len(combined_df)} records for {instrument} "
                f"from {len(dataframes)}/{len(csv_files)} files in {duration:.2f}s (cached)"
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

    # Backward-compatibility alias for older UI code paths
    def load_data(self, instrument: str, start_date: Optional[datetime] = None,
                  end_date: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Compat shim: forwards to load_instrument_data.

        Accepts either datetime or date for start/end and normalizes.
        """
        # Normalize start/end to datetime at midnight when given as date
        if isinstance(start_date, date) and not isinstance(start_date, datetime):
            start_date = datetime.combine(start_date, datetime.min.time())
        if isinstance(end_date, date) and not isinstance(end_date, datetime):
            end_date = datetime.combine(end_date, datetime.min.time())
        return self.load_instrument_data(instrument, start_date, end_date)
    
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

    # --------------------------- Flat Daily Mode Helpers ---------------------------- #
    def _load_flat_daily_file(self, instrument: str, start_date: Optional[datetime], end_date: Optional[datetime]) -> Optional[pd.DataFrame]:
        """Load a flat daily CSV file (<instrument>.csv) and normalize schema.

        Creates a datetime column at midnight for each trading day. Applies date
        range filtering if provided.
        """
        try:
            file_path = self.data_directory / f"{instrument}.csv"
            if not file_path.exists():
                raise DataLoadingError(
                    f"Instrument file not found: {instrument}.csv",
                    error_code="INSTRUMENT_FILE_NOT_FOUND",
                    context={'instrument': instrument, 'path': str(file_path)}
                )

            cache_key = f"daily_file_{instrument}_{self.data_directory.name}"
            cached = cache_manager.get(cache_key)
            if cached is not None:
                # Filter cached data if date range specified
                df_cached = cached
                if start_date or end_date:
                    # Compare on date objects to avoid datetime vs date mismatches
                    start_d = start_date.date() if isinstance(start_date, datetime) else start_date
                    end_d = end_date.date() if isinstance(end_date, datetime) else end_date
                    mask = True
                    if start_d:
                        mask &= df_cached['datetime'].dt.date >= start_d
                    if end_d:
                        mask &= df_cached['datetime'].dt.date <= end_d
                    filtered = df_cached[mask].reset_index(drop=True)
                    return filtered
                return df_cached

            # Read full file
            try:
                df = pd.read_csv(file_path)
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')

            # Map columns (case-insensitive)
            col_map = {c.lower(): c for c in df.columns}
            required_raw = {
                'date': None,
                'open': None,
                'high': None,
                'low': None,
                'close': None,
                'volume': None
            }
            for logical in required_raw.keys():
                for actual_lower, actual in col_map.items():
                    if actual_lower == logical:
                        required_raw[logical] = actual
                        break
                # Special cases (source uses capitalized names)
                if required_raw[logical] is None:
                    fallback = logical.capitalize() if logical != 'volume' else 'Volume'
                    if fallback in df.columns:
                        required_raw[logical] = fallback
            missing = [k for k, v in required_raw.items() if v is None]
            if missing:
                raise DataValidationError(
                    f"Missing required columns in daily file {file_path.name}: {missing}",
                    error_code="MISSING_DAILY_COLUMNS",
                    context={'file': file_path.name, 'columns': list(df.columns)}
                )

            # Build normalized DataFrame
            norm = pd.DataFrame({
                'date': df[required_raw['date']].astype(str).str.strip(),
                'open': df[required_raw['open']],
                'high': df[required_raw['high']],
                'low': df[required_raw['low']],
                'close': df[required_raw['close']],
                'volume': df[required_raw['volume']]
            })

            # Parse date to datetime (try ISO first)
            parsed = None
            for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']:
                try:
                    parsed = pd.to_datetime(norm['date'], format=fmt)
                    break
                except Exception:
                    continue
            if parsed is None:
                parsed = pd.to_datetime(norm['date'], errors='coerce', infer_datetime_format=True)
            valid_mask = parsed.notna()
            norm = norm[valid_mask].copy()
            norm['datetime'] = parsed[valid_mask].dt.floor('D')  # midnight

            # Convert numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                norm[col] = pd.to_numeric(norm[col], errors='coerce')
            norm = norm.dropna(subset=['open', 'high', 'low', 'close', 'volume'])

            # Sort
            norm = norm.sort_values('datetime').reset_index(drop=True)

            # Optional date range filtering
            if start_date or end_date:
                # Normalize comparisons on date objects
                start_d = start_date.date() if isinstance(start_date, datetime) else start_date
                end_d = end_date.date() if isinstance(end_date, datetime) else end_date
                mask = True
                if start_d:
                    mask &= norm['datetime'].dt.date >= start_d
                if end_d:
                    mask &= norm['datetime'].dt.date <= end_d
                norm = norm[mask].reset_index(drop=True)
                if norm.empty:
                    return None

            # Basic validation (reuse integrity checker)
            is_valid, issues = error_handler.validate_data_integrity(norm, f"flat_daily({instrument})")
            if not is_valid and issues:
                error_handler.show_warning_message(
                    f"Data quality issues detected for {instrument}",
                    issues[:3]
                )

            cache_manager.set(cache_key, norm, ttl=1800, source_files=[str(file_path)])
            logger.info(f"Loaded daily data for {instrument}: {len(norm)} records (cached)")
            return norm
        except (DataLoadingError, DataValidationError):
            raise
        except Exception as e:
            raise DataLoadingError(
                f"Unexpected error loading daily file for '{instrument}': {e}",
                error_code="UNEXPECTED_DAILY_LOAD_ERROR",
                context={'instrument': instrument}
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