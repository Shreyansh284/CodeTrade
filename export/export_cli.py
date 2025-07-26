#!/usr/bin/env python3
"""
Command-line export script for Stock Pattern Detector.

This script provides a command-line interface for running pattern detection
and exporting results to CSV files without using the Streamlit UI.

Usage:
    python export_cli.py                    # Interactive mode
    python export_cli.py --batch            # Batch mode with all instruments
    python export_cli.py --help             # Show help
"""

import argparse
import sys
import os
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from export.cli_interface import CommandLineInterface
from utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Pattern Detector - CSV Export Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python export_cli.py                           # Interactive mode
    python export_cli.py --batch                   # Batch export all instruments
    python export_cli.py --instrument BAJAJ-AUTO  # Export specific instrument
    python export_cli.py --timeframe 5min          # Use specific timeframe
    python export_cli.py --patterns hammer,doji    # Detect specific patterns
        """
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run in batch mode (export all available instruments)'
    )
    
    parser.add_argument(
        '--instrument',
        type=str,
        help='Specific instrument to analyze (e.g., BAJAJ-AUTO)'
    )
    
    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['1min', '5min', '15min', '1hour', '2hour', '5hour', '1day', 'all'],
        default='5min',
        help='Timeframe for analysis or "all" for all timeframes (default: 5min)'
    )
    
    parser.add_argument(
        '--all-timeframes',
        action='store_true',
        help='Process all available timeframes (same as --timeframe all)'
    )
    
    parser.add_argument(
        '--patterns',
        type=str,
        help='Comma-separated list of patterns to detect (e.g., hammer,doji,rising_window)'
    )
    
    parser.add_argument(
        '--all-instruments',
        action='store_true',
        help='Process all available instruments (same as --batch)'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel processing for batch operations (experimental)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='exports',
        help='Output directory for CSV files (default: exports)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='5Scripts',
        help='Data directory containing CSV files (default: 5Scripts)'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Stock Pattern Detector CLI v1.0'
    )
    
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace, cli: CommandLineInterface) -> bool:
    """
    Validate command-line arguments.
    
    Args:
        args: Parsed arguments
        cli: CLI interface instance
        
    Returns:
        True if arguments are valid, False otherwise
    """
    try:
        # Validate instrument if specified
        if args.instrument:
            available_instruments = cli.get_available_instruments()
            if args.instrument not in available_instruments:
                if not args.quiet:
                    print(f"❌ Instrument '{args.instrument}' not found.")
                    print(f"Available instruments: {', '.join(available_instruments)}")
                return False
        
        # Validate patterns if specified
        if args.patterns:
            available_patterns = list(cli.pattern_detectors.keys())
            requested_patterns = [p.strip() for p in args.patterns.split(',')]
            
            invalid_patterns = [p for p in requested_patterns if p not in available_patterns]
            if invalid_patterns:
                if not args.quiet:
                    print(f"❌ Invalid patterns: {', '.join(invalid_patterns)}")
                    print(f"Available patterns: {', '.join(available_patterns)}")
                return False
        
        # Validate data directory
        if not os.path.exists(args.data_dir):
            if not args.quiet:
                print(f"❌ Data directory not found: {args.data_dir}")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating arguments: {e}")
        if not args.quiet:
            print(f"❌ Argument validation error: {e}")
        return False


def run_single_export(
    cli: CommandLineInterface, 
    instrument: str, 
    timeframe: str, 
    patterns: List[str],
    quiet: bool = False
) -> bool:
    """
    Run export for a single instrument configuration.
    
    Args:
        cli: CLI interface instance
        instrument: Instrument to analyze
        timeframe: Timeframe for analysis
        patterns: List of patterns to detect
        quiet: Suppress output
        
    Returns:
        True if export succeeded, False otherwise
    """
    try:
        if not quiet:
            print(f"\n🔄 Exporting {instrument} ({timeframe})...")
        
        export_path = cli.run_analysis_and_export(instrument, timeframe, patterns)
        
        if export_path:
            if not quiet:
                print(f"✅ Export completed: {export_path}")
            return True
        else:
            if not quiet:
                print(f"❌ Export failed for {instrument}")
            return False
            
    except Exception as e:
        logger.error(f"Error in single export: {e}")
        if not quiet:
            print(f"❌ Export error: {e}")
        return False


def run_batch_export(
    cli: CommandLineInterface, 
    timeframe: str, 
    patterns: List[str],
    quiet: bool = False
) -> List[str]:
    """
    Run batch export for all available instruments.
    
    Args:
        cli: CLI interface instance
        timeframe: Timeframe for analysis
        patterns: List of patterns to detect
        quiet: Suppress output
        
    Returns:
        List of successfully exported file paths
    """
    try:
        instruments = cli.get_available_instruments()
        
        if not instruments:
            if not quiet:
                print("❌ No instruments found for batch export")
            return []
        
        if not quiet:
            print(f"\n🔄 Starting batch export for {len(instruments)} instruments...")
        
        batch_config = {
            'instruments': instruments,
            'timeframes': [timeframe],
            'patterns': patterns
        }
        
        return cli.run_batch_mode(batch_config)
        
    except Exception as e:
        logger.error(f"Error in batch export: {e}")
        if not quiet:
            print(f"❌ Batch export error: {e}")
        return []


def main():
    """Main entry point for the CLI export script."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Initialize CLI interface
        cli = CommandLineInterface(data_directory=args.data_dir)
        cli.csv_exporter.export_directory = args.output_dir
        
        # Validate arguments
        if not validate_arguments(args, cli):
            sys.exit(1)
        
        # Determine patterns to detect
        if args.patterns:
            patterns = [p.strip() for p in args.patterns.split(',')]
        else:
            patterns = list(cli.pattern_detectors.keys())
        
        # Determine timeframes to process
        all_timeframes = ['1min', '5min', '15min', '1hour', '2hour', '5hour', '1day']
        if args.timeframe == 'all' or args.all_timeframes:
            timeframes = all_timeframes
        else:
            timeframes = [args.timeframe]
        
        # Determine instruments to process
        if args.batch or args.all_instruments:
            instruments = cli.get_available_instruments()
            if not instruments:
                if not args.quiet:
                    print("❌ No instruments found for batch processing")
                sys.exit(1)
        elif args.instrument:
            instruments = [args.instrument]
        else:
            # Interactive mode
            if not args.quiet:
                print("🚀 Starting interactive mode...")
            cli.run_interactive_mode()
            return
        
        # Run batch processing
        if not args.quiet:
            print(f"\n� Starting batch processing...")
            print(f"📊 Instruments: {len(instruments)}")
            print(f"⏰ Timeframes: {len(timeframes)}")
            print(f"🔍 Patterns: {len(patterns)}")
        
        exported_files = []
        total_jobs = len(instruments) * len(timeframes)
        current_job = 0
        
        for instrument in instruments:
            for timeframe in timeframes:
                current_job += 1
                if not args.quiet:
                    print(f"\n[{current_job}/{total_jobs}] Processing {instrument} ({timeframe})...")
                
                success = run_single_export(cli, instrument, timeframe, patterns, args.quiet)
                if success:
                    exported_files.append(f"{instrument}_{timeframe}")
        
        if not args.quiet:
            print(f"\n🎉 Batch processing completed!")
            print(f"✅ Successfully processed {len(exported_files)} combinations")
            print(f"❌ Failed: {total_jobs - len(exported_files)} combinations")
        
        sys.exit(0 if exported_files else 1)
            
    except KeyboardInterrupt:
        if not args.quiet:
            print("\n\n👋 Export cancelled by user.")
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"CLI main error: {e}")
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()