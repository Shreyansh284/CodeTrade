"""
Legacy export patterns module for backward compatibility.
This module provides the export_patterns_to_csv function that was referenced in main.py.
"""

import os
import sys
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from patterns.base import PatternResult
from export.csv_exporter import CSVExporter
from utils.logging_config import get_logger

logger = get_logger(__name__)


def export_patterns_to_csv(patterns: List[PatternResult], filename: str) -> str:
    """
    Export patterns to CSV file (legacy function for backward compatibility).
    
    Args:
        patterns: List of pattern results to export
        filename: Name of the output CSV file
        
    Returns:
        Path to the exported file or None if failed
    """
    try:
        # Extract dataset name from filename if possible
        dataset_name = filename.split('_')[0] if '_' in filename else 'Unknown'
        
        # Extract timeframe from filename if possible
        timeframe = '5min'  # Default
        if '_' in filename:
            parts = filename.split('_')
            for part in parts:
                if part in ['1min', '5min', '15min', '1hour', '2hour', '5hour', '1day']:
                    timeframe = part
                    break
        
        # Use the new CSV exporter
        exporter = CSVExporter()
        export_path = exporter.export_patterns(patterns, dataset_name, timeframe, filename)
        
        logger.info(f"Legacy export completed: {export_path}")
        return export_path
        
    except Exception as e:
        logger.error(f"Legacy export failed: {e}")
        return None