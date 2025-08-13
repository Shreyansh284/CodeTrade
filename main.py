#!/usr/bin/env python3
"""
Stock Pattern Detector - Main Entry Point

Simplified application focused on:
- Head & Shoulders pattern detection
- Double Top pattern detection  
- Double Bottom pattern detection

Usage:
    python3 main.py        # Launch Streamlit GUI (default)
    python3 main.py --help # Show help
"""

import sys
import os
import subprocess
import argparse

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logging_config import get_logger

logger = get_logger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Stock Pattern Detector - Focused on structural patterns",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 main.py                    # Launch web interface (default)
    python3 main.py --help            # Show this help
        """
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Stock Pattern Detector 2.0'
    )
    
    return parser.parse_args()


def launch_gui():
    """Launch Streamlit GUI interface."""
    try:
        logger.info("Starting Streamlit GUI...")
        streamlit_script = os.path.join(os.path.dirname(__file__), 'ui', 'streamlit_app.py')
        
        if not os.path.exists(streamlit_script):
            logger.error(f"Streamlit app not found at: {streamlit_script}")
            return 1
        
        # Launch Streamlit
        result = subprocess.run([
            'python3', '-m', 'streamlit', 'run', streamlit_script,
            '--server.port', '8501',
            '--server.address', 'localhost',
            '--browser.gatherUsageStats', 'false'
        ], check=False)
        
        return result.returncode
        
    except KeyboardInterrupt:
        logger.info("GUI shutdown requested by user")
        return 0
    except Exception as e:
        logger.error(f"Failed to launch GUI: {e}")
        return 1


def main():
    """Main entry point."""
    try:
        args = parse_arguments()
        
        # Default behavior is to launch GUI
        logger.info("Stock Pattern Detector - Starting...")
        return launch_gui()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Application error: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())