"""
CSV export functionality for stock pattern detection results.
"""

import os
import sys
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patterns.base import PatternResult
from utils.logging_config import get_logger
from utils.error_handler import ExportError, safe_execute

logger = get_logger(__name__)


class CSVExporter:
    """Handles CSV export operations for pattern detection results."""
    
    def __init__(self, export_directory: str = "exports"):
        """
        Initialize CSV exporter.
        
        Args:
            export_directory: Directory to save exported CSV files
        """
        self.export_directory = export_directory
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Create export directory if it doesn't exist
        self._ensure_export_directory()
    
    def _ensure_export_directory(self) -> None:
        """Create export directory if it doesn't exist."""
        try:
            if not os.path.exists(self.export_directory):
                os.makedirs(self.export_directory)
                self.logger.info(f"Created export directory: {self.export_directory}")
        except Exception as e:
            self.logger.error(f"Failed to create export directory {self.export_directory}: {e}")
            raise ExportError(
                f"Cannot create export directory: {e}",
                error_code="EXPORT_DIRECTORY_CREATION_FAILED",
                context={'directory': self.export_directory}
            )
    
    def export_patterns(
        self, 
        patterns: List[PatternResult], 
        dataset_name: str,
        timeframe: str,
        filename: Optional[str] = None
    ) -> str:
        """
        Export pattern detection results to CSV file.
        
        Args:
            patterns: List of detected patterns
            dataset_name: Name of the dataset/instrument
            timeframe: Time period used for analysis
            filename: Optional custom filename (auto-generated if None)
            
        Returns:
            Path to the exported CSV file
            
        Raises:
            ExportError: If export operation fails
        """
        try:
            # Generate filename if not provided
            if filename is None:
                filename = self.generate_filename(dataset_name, timeframe)
            
            # Create export DataFrame
            export_df = self.create_export_dataframe(patterns, dataset_name)
            
            # Validate export data
            self._validate_export_data(export_df)
            
            # Full path for the export file
            export_path = os.path.join(self.export_directory, filename)
            
            # Export to CSV
            export_df.to_csv(export_path, index=False)
            
            self.logger.info(
                f"Successfully exported {len(patterns)} patterns to {export_path}"
            )
            
            return export_path
            
        except Exception as e:
            error_msg = f"Failed to export patterns for {dataset_name}: {str(e)}"
            self.logger.error(error_msg)
            raise ExportError(
                error_msg,
                error_code="PATTERN_EXPORT_FAILED",
                context={
                    'dataset_name': dataset_name,
                    'timeframe': timeframe,
                    'pattern_count': len(patterns) if patterns else 0,
                    'filename': filename
                }
            ) from e
    
    def generate_filename(self, dataset_name: str, timeframe: str) -> str:
        """
        Generate descriptive filename with timestamp and dataset information.
        
        Args:
            dataset_name: Name of the dataset/instrument
            timeframe: Time period used for analysis
            
        Returns:
            Generated filename string
        """
        try:
            # Clean dataset name for filename
            clean_dataset = dataset_name.replace('-', '_').replace(' ', '_')
            
            # Generate timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"patterns_{clean_dataset}_{timeframe}_{timestamp}.csv"
            
            self.logger.debug(f"Generated filename: {filename}")
            return filename
            
        except Exception as e:
            # Fallback to simple filename
            fallback_filename = f"patterns_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            self.logger.warning(f"Error generating filename, using fallback: {fallback_filename}")
            return fallback_filename
    
    def create_export_dataframe(
        self, 
        patterns: List[PatternResult], 
        dataset_name: str
    ) -> pd.DataFrame:
        """
        Create DataFrame with required export columns from pattern results.
        
        Args:
            patterns: List of detected patterns
            dataset_name: Name of the dataset/instrument
            
        Returns:
            DataFrame formatted for CSV export
        """
        try:
            if not patterns:
                # Create empty DataFrame with required headers
                return pd.DataFrame(columns=[
                    'dataset_name',
                    'datetime', 
                    'timeframe',
                    'pattern',
                    'confidence_score'
                ])
            
            # Convert patterns to export format
            export_data = []
            for pattern in patterns:
                try:
                    # Format datetime consistently
                    if hasattr(pattern.datetime, 'strftime'):
                        datetime_str = pattern.datetime.strftime('%Y-%m-%d %H:%M:%S')
                    else:
                        datetime_str = str(pattern.datetime)
                    
                    # Format pattern name
                    pattern_name = pattern.pattern_type.replace('_', ' ').title()
                    
                    export_row = {
                        'dataset_name': dataset_name,
                        'datetime': datetime_str,
                        'timeframe': pattern.timeframe,
                        'pattern': pattern_name,
                        'confidence_score': round(pattern.confidence, 3)
                    }
                    
                    export_data.append(export_row)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing pattern for export: {e}")
                    continue
            
            # Create DataFrame
            export_df = pd.DataFrame(export_data)
            
            # Sort by datetime (most recent first)
            if not export_df.empty and 'datetime' in export_df.columns:
                export_df = export_df.sort_values('datetime', ascending=False)
            
            self.logger.debug(f"Created export DataFrame with {len(export_df)} rows")
            return export_df
            
        except Exception as e:
            self.logger.error(f"Error creating export DataFrame: {e}")
            # Return empty DataFrame with headers as fallback
            return pd.DataFrame(columns=[
                'dataset_name',
                'datetime', 
                'timeframe',
                'pattern',
                'confidence_score'
            ])
    
    def _validate_export_data(self, export_df: pd.DataFrame) -> None:
        """
        Validate export data integrity.
        
        Args:
            export_df: DataFrame to validate
            
        Raises:
            ExportError: If validation fails
        """
        try:
            required_columns = [
                'dataset_name',
                'datetime', 
                'timeframe',
                'pattern',
                'confidence_score'
            ]
            
            # Check required columns exist
            missing_columns = [col for col in required_columns if col not in export_df.columns]
            if missing_columns:
                raise ExportError(
                    f"Missing required columns in export data: {missing_columns}",
                    error_code="EXPORT_VALIDATION_MISSING_COLUMNS",
                    context={'missing_columns': missing_columns}
                )
            
            # Check data types and values
            if not export_df.empty:
                # Validate confidence scores are numeric and in valid range
                if 'confidence_score' in export_df.columns:
                    invalid_confidence = export_df[
                        (export_df['confidence_score'] < 0) | 
                        (export_df['confidence_score'] > 1)
                    ]
                    if not invalid_confidence.empty:
                        self.logger.warning(
                            f"Found {len(invalid_confidence)} rows with invalid confidence scores"
                        )
                
                # Check for null values in critical columns
                critical_columns = ['dataset_name', 'datetime', 'pattern']
                for col in critical_columns:
                    null_count = export_df[col].isnull().sum()
                    if null_count > 0:
                        self.logger.warning(f"Found {null_count} null values in {col} column")
            
            self.logger.debug("Export data validation passed")
            
        except ExportError:
            raise
        except Exception as e:
            raise ExportError(
                f"Export data validation failed: {str(e)}",
                error_code="EXPORT_VALIDATION_ERROR",
                context={'error_details': str(e)}
            ) from e
    
    def export_empty_results(
        self, 
        dataset_name: str, 
        timeframe: str,
        message: str = "No patterns detected"
    ) -> str:
        """
        Export empty CSV file when no patterns are detected.
        
        Args:
            dataset_name: Name of the dataset/instrument
            timeframe: Time period used for analysis
            message: Message to log about empty results
            
        Returns:
            Path to the exported CSV file
        """
        try:
            self.logger.info(f"{message} for {dataset_name} ({timeframe})")
            
            # Create empty DataFrame with headers
            empty_df = pd.DataFrame(columns=[
                'dataset_name',
                'datetime', 
                'timeframe',
                'pattern',
                'confidence_score'
            ])
            
            # Generate filename
            filename = self.generate_filename(dataset_name, timeframe)
            export_path = os.path.join(self.export_directory, filename)
            
            # Export empty CSV
            empty_df.to_csv(export_path, index=False)
            
            self.logger.info(f"Created empty export file: {export_path}")
            return export_path
            
        except Exception as e:
            error_msg = f"Failed to create empty export file: {str(e)}"
            self.logger.error(error_msg)
            raise ExportError(
                error_msg,
                error_code="EMPTY_EXPORT_FAILED",
                context={
                    'dataset_name': dataset_name,
                    'timeframe': timeframe
                }
            ) from e
    
    def get_export_summary(self, export_path: str) -> Dict[str, Any]:
        """
        Get summary information about an exported file.
        
        Args:
            export_path: Path to the exported CSV file
            
        Returns:
            Dictionary with export summary information
        """
        try:
            if not os.path.exists(export_path):
                return {'error': 'Export file not found'}
            
            # Get file stats
            file_stats = os.stat(export_path)
            file_size = file_stats.st_size
            
            # Read CSV to get row count
            try:
                df = pd.read_csv(export_path)
                row_count = len(df)
                has_data = row_count > 0
            except Exception:
                row_count = 0
                has_data = False
            
            summary = {
                'file_path': export_path,
                'file_size_bytes': file_size,
                'row_count': row_count,
                'has_data': has_data,
                'created_at': datetime.fromtimestamp(file_stats.st_ctime).isoformat()
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting export summary: {e}")
            return {'error': str(e)}