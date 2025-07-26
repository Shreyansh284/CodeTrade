#!/usr/bin/env python3
"""
Stock Pattern Detector - Main Entry Point

This is the main entry point for the Stock Pattern Detector application.
It provides command-line interface to access different modes:
- GUI mode: Launches the Streamlit web interface
- Export mode: Runs the command-line export tool

Usage:
    python3 main.py -gui        # Launch Streamlit GUI
    python3 main.py -export     # Launch export CLI
    python3 main.py --help      # Show help
"""

import sys
import os
import argparse
import subprocess
from typing import List, Optional

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Pattern Detector - Main Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 main.py -gui                    # Launch web interface
    python3 main.py -export                 # Launch export CLI
    python3 main.py -export --batch         # Run batch export
    python3 main.py -export --help          # Show export help
        """
    )
    
    # Main mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '-gui', '--gui',
        action='store_true',
        help='Launch Streamlit GUI interface'
    )
    mode_group.add_argument(
        '-export', '--export',
        action='store_true',
        help='Launch command-line export tool'
    )
    
    # Export-specific arguments (passed through to export CLI)
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run export in batch mode (export all instruments)'
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
        help='Timeframe for analysis or "all" for all timeframes'
    )
    
    parser.add_argument(
        '--all-timeframes',
        action='store_true',
        help='Process all available timeframes (same as --timeframe all)'
    )
    
    parser.add_argument(
        '--patterns',
        type=str,
        help='Comma-separated list of patterns to detect'
    )
    
    parser.add_argument(
        '--all-instruments',
        action='store_true',
        help='Process all available instruments (same as --batch)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory for CSV files'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        help='Data directory containing CSV files'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error output (export mode only)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Stock Pattern Detector v1.0'
    )
    
    return parser.parse_args()


def launch_gui() -> int:
    """
    Launch the Streamlit GUI interface.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        print("🚀 Launching Stock Pattern Detector GUI...")
        print("📱 Opening web interface in your default browser...")
        print("🔗 URL: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        # Launch Streamlit app
        streamlit_cmd = [
            sys.executable, '-m', 'streamlit', 'run',
            'ui/streamlit_app.py',
            '--server.port=8501',
            '--server.address=localhost',
            '--browser.gatherUsageStats=false'
        ]
        
        result = subprocess.run(streamlit_cmd, check=False)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n👋 GUI stopped by user.")
        return 0
    except FileNotFoundError:
        print("❌ Error: Streamlit not found. Please install it with:")
        print("   pip install streamlit")
        return 1
    except Exception as e:
        logger.error(f"Error launching GUI: {e}")
        print(f"❌ Failed to launch GUI: {e}")
        return 1


def launch_export(export_args: List[str]) -> int:
    """
    Launch the export CLI tool.
    
    Args:
        export_args: List of arguments to pass to export CLI
        
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        print("🚀 Launching Stock Pattern Detector Export CLI...")
        
        # Build command for export CLI
        export_cmd = [sys.executable, 'export/export_cli.py'] + export_args
        
        # Execute export CLI
        result = subprocess.run(export_cmd, check=False)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n👋 Export cancelled by user.")
        return 130  # Standard exit code for SIGINT
    except FileNotFoundError:
        print("❌ Error: Export CLI script not found.")
        return 1
    except Exception as e:
        logger.error(f"Error launching export CLI: {e}")
        print(f"❌ Failed to launch export CLI: {e}")
        return 1


def build_export_args(args: argparse.Namespace) -> List[str]:
    """
    Build argument list for export CLI from parsed arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        List of arguments for export CLI
    """
    export_args = []
    
    # Add export-specific arguments
    if args.batch:
        export_args.append('--batch')
    
    if args.all_instruments:
        export_args.append('--all-instruments')
    
    if args.instrument:
        export_args.extend(['--instrument', args.instrument])
    
    if args.timeframe:
        export_args.extend(['--timeframe', args.timeframe])
    
    if args.all_timeframes:
        export_args.append('--all-timeframes')
    
    if args.patterns:
        export_args.extend(['--patterns', args.patterns])
    
    if args.output_dir:
        export_args.extend(['--output-dir', args.output_dir])
    
    if args.data_dir:
        export_args.extend(['--data-dir', args.data_dir])
    
    if args.quiet:
        export_args.append('--quiet')
    
    return export_args


def validate_environment() -> bool:
    """
    Validate that the environment is properly set up.
    
    Returns:
        True if environment is valid, False otherwise
    """
    try:
        # Check if required directories exist
        required_dirs = ['data', 'patterns', 'export', 'ui', 'utils']
        missing_dirs = []
        
        for dir_name in required_dirs:
            if not os.path.exists(dir_name):
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            print(f"❌ Missing required directories: {', '.join(missing_dirs)}")
            return False
        
        # Check if data directory has content
        data_dir = '5Scripts'
        if not os.path.exists(data_dir):
            print(f"⚠️  Warning: Data directory '{data_dir}' not found.")
            print("   You may need to add your CSV data files to this directory.")
        
        return True
        
    except Exception as e:
        logger.error(f"Environment validation error: {e}")
        print(f"❌ Environment validation failed: {e}")
        return False


def show_welcome_message():
    """Display welcome message and basic information."""
    print("=" * 60)
    print("📈 Stock Pattern Detector")
    print("=" * 60)
    print("Professional candlestick pattern detection and analysis tool")
    print("")
    print("Available modes:")
    print("  🖥️  GUI Mode:    Interactive web interface")
    print("  📤 Export Mode: Command-line export tool")
    print("")


def main() -> int:
    """
    Main entry point for the application.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    try:
        # Parse command-line arguments
        args = parse_arguments()
        
        # Show welcome message
        if not args.quiet:
            show_welcome_message()
        
        # Validate environment
        if not validate_environment():
            return 1
        
        # Route to appropriate mode
        if args.gui:
            return launch_gui()
        elif args.export:
            export_args = build_export_args(args)
            return launch_export(export_args)
        else:
            # This shouldn't happen due to mutually_exclusive_group
            print("❌ No mode specified. Use -gui or -export.")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n👋 Application stopped by user.")
        return 0
    except Exception as e:
        logger.error(f"Main application error: {e}")
        print(f"❌ Application error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)