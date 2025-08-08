"""
Command-line interface for CSV export operations.
"""

import os
import sys
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import CSVLoader
from data.aggregator import DataAggregator
from patterns.dragonfly_doji import DragonflyDojiDetector
from patterns.hammer import HammerDetector
from patterns.rising_window import RisingWindowDetector
from patterns.evening_star import EveningStarDetector
from patterns.three_white_soldiers import ThreeWhiteSoldiersDetector
from patterns.double_top import DoubleTopDetector
from patterns.double_bottom import DoubleBottomDetector
from export.csv_exporter import CSVExporter
from utils.logging_config import get_logger
from utils.error_handler import ExportError, safe_execute

logger = get_logger(__name__)


class CommandLineInterface:
    """Handles command-line export operations for pattern detection."""
    
    def __init__(self, data_directory: str = "5Scripts"):
        """
        Initialize CLI interface.
        
        Args:
            data_directory: Directory containing CSV data files
        """
        self.data_directory = data_directory
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self.csv_loader = CSVLoader(data_directory)
        self.data_aggregator = DataAggregator()
        self.csv_exporter = CSVExporter()
        
        # Initialize pattern detectors
        self.pattern_detectors = {
            'dragonfly_doji': DragonflyDojiDetector(),
            'hammer': HammerDetector(),
            'rising_window': RisingWindowDetector(),
            'evening_star': EveningStarDetector(),
            'three_white_soldiers': ThreeWhiteSoldiersDetector(),
            'double_top': DoubleTopDetector(),
            'double_bottom': DoubleBottomDetector()
        }
        
        # Available timeframes
        self.timeframes = ['1min', '5min', '15min', '1hour', '2hour', '5hour', '1day']
    
    def get_available_instruments(self) -> List[str]:
        """
        Get list of available instruments from data directory.
        
        Returns:
            List of available instrument names
        """
        try:
            instruments = self.csv_loader.get_available_instruments()
            self.logger.info(f"Found {len(instruments)} available instruments")
            return instruments
        except Exception as e:
            self.logger.error(f"Error getting available instruments: {e}")
            return []
    
    def prompt_dataset_selection(self) -> Optional[str]:
        """
        Prompt user to select dataset from available instruments.
        
        Returns:
            Selected instrument name or None if cancelled
        """
        try:
            instruments = self.get_available_instruments()
            
            if not instruments:
                print("❌ No instruments found in the data directory.")
                return None
            
            print("\n📊 Available Instruments:")
            print("=" * 40)
            
            for i, instrument in enumerate(instruments, 1):
                print(f"{i}. {instrument}")
            
            print("0. Cancel")
            print("=" * 40)
            
            while True:
                try:
                    choice = input("\nSelect instrument (1-{}): ".format(len(instruments)))
                    
                    if choice == '0':
                        print("Operation cancelled.")
                        return None
                    
                    choice_num = int(choice)
                    if 1 <= choice_num <= len(instruments):
                        selected = instruments[choice_num - 1]
                        print(f"✅ Selected: {selected}")
                        return selected
                    else:
                        print(f"❌ Please enter a number between 1 and {len(instruments)}")
                        
                except ValueError:
                    print("❌ Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled by user.")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in dataset selection: {e}")
            print(f"❌ Error during selection: {e}")
            return None
    
    def prompt_timeframe_selection(self) -> Optional[str]:
        """
        Prompt user to select timeframe for analysis.
        
        Returns:
            Selected timeframe or None if cancelled
        """
        try:
            print("\n⏰ Available Timeframes:")
            print("=" * 30)
            
            # Add "All Timeframes" option
            print("1. All Timeframes (process all)")
            
            for i, timeframe in enumerate(self.timeframes, 2):
                print(f"{i}. {timeframe}")
            
            print("0. Cancel")
            print("=" * 30)
            
            while True:
                try:
                    choice = input(f"\nSelect timeframe (1-{len(self.timeframes) + 1}): ")
                    
                    if choice == '0':
                        print("Operation cancelled.")
                        return None
                    
                    choice_num = int(choice)
                    if choice_num == 1:
                        print("✅ Selected: All Timeframes")
                        return "all"
                    elif 2 <= choice_num <= len(self.timeframes) + 1:
                        selected = self.timeframes[choice_num - 2]
                        print(f"✅ Selected: {selected}")
                        return selected
                    else:
                        print(f"❌ Please enter a number between 1 and {len(self.timeframes) + 1}")
                        
                except ValueError:
                    print("❌ Please enter a valid number")
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled by user.")
                    return None
                    
        except Exception as e:
            self.logger.error(f"Error in timeframe selection: {e}")
            print(f"❌ Error during selection: {e}")
            return None
    
    def prompt_pattern_selection(self) -> List[str]:
        """
        Prompt user to select patterns to detect.
        
        Returns:
            List of selected pattern names
        """
        try:
            pattern_names = list(self.pattern_detectors.keys())
            
            print("\n🔍 Available Patterns:")
            print("=" * 35)
            
            for i, pattern in enumerate(pattern_names, 1):
                display_name = pattern.replace('_', ' ').title()
                print(f"{i}. {display_name}")
            
            print(f"{len(pattern_names) + 1}. All patterns")
            print("0. Cancel")
            print("=" * 35)
            
            while True:
                try:
                    choice = input(f"\nSelect patterns (1-{len(pattern_names) + 1}, or comma-separated): ")
                    
                    if choice == '0':
                        print("Operation cancelled.")
                        return []
                    
                    # Handle "all patterns" option
                    if choice == str(len(pattern_names) + 1):
                        print("✅ Selected: All patterns")
                        return pattern_names
                    
                    # Handle comma-separated choices
                    selected_patterns = []
                    choices = [c.strip() for c in choice.split(',')]
                    
                    for c in choices:
                        choice_num = int(c)
                        if 1 <= choice_num <= len(pattern_names):
                            selected_patterns.append(pattern_names[choice_num - 1])
                        else:
                            print(f"❌ Invalid choice: {c}")
                            break
                    else:
                        if selected_patterns:
                            display_names = [p.replace('_', ' ').title() for p in selected_patterns]
                            print(f"✅ Selected: {', '.join(display_names)}")
                            return selected_patterns
                    
                    print(f"❌ Please enter valid numbers between 1 and {len(pattern_names)}")
                        
                except ValueError:
                    print("❌ Please enter valid numbers")
                except KeyboardInterrupt:
                    print("\n\nOperation cancelled by user.")
                    return []
                    
        except Exception as e:
            self.logger.error(f"Error in pattern selection: {e}")
            print(f"❌ Error during selection: {e}")
            return []
    
    def run_analysis_and_export(
        self, 
        instrument: str, 
        timeframe: str,
        selected_patterns: List[str] = None
    ) -> Optional[str]:
        """
        Run pattern analysis and export results to CSV.
        
        Args:
            instrument: Instrument name to analyze
            timeframe: Timeframe for analysis
            selected_patterns: List of pattern names to detect (all if None)
            
        Returns:
            Path to exported CSV file or None if failed
        """
        try:
            print(f"\n🔄 Starting analysis for {instrument} ({timeframe})...")
            
            # Use all patterns if none specified
            if selected_patterns is None:
                selected_patterns = list(self.pattern_detectors.keys())
            
            # Load data
            print("📥 Loading data...")
            data = safe_execute(
                self.csv_loader.load_instrument_data,
                instrument,
                context=f"load_data_{instrument}",
                show_errors=True
            )
            
            if data is None or data.empty:
                print(f"❌ No data available for {instrument}")
                return None
            
            print(f"✅ Loaded {len(data)} data points")
            
            # Aggregate data if needed
            if timeframe != '1min':
                print(f"🔄 Aggregating data to {timeframe}...")
                data = safe_execute(
                    self.data_aggregator.aggregate_data,
                    data, timeframe,
                    context=f"aggregate_data_{timeframe}",
                    show_errors=True
                )
                
                if data is None or data.empty:
                    print(f"❌ Failed to aggregate data to {timeframe}")
                    return None
                
                print(f"✅ Aggregated to {len(data)} data points")
            
            # Detect patterns
            print("🔍 Detecting patterns...")
            all_patterns = []
            
            for pattern_name in selected_patterns:
                if pattern_name in self.pattern_detectors:
                    detector = self.pattern_detectors[pattern_name]
                    
                    print(f"  - Detecting {pattern_name.replace('_', ' ').title()}...")
                    patterns = safe_execute(
                        detector.detect,
                        data, timeframe,
                        context=f"detect_{pattern_name}",
                        fallback_result=[],
                        show_errors=True
                    )
                    
                    if patterns:
                        all_patterns.extend(patterns)
                        print(f"    ✅ Found {len(patterns)} patterns")
                    else:
                        print(f"    ℹ️  No patterns found")
            
            print(f"\n📊 Total patterns detected: {len(all_patterns)}")
            
            # Export results
            print("📤 Exporting results...")
            
            if all_patterns:
                export_path = safe_execute(
                    self.csv_exporter.export_patterns,
                    all_patterns, instrument, timeframe,
                    context=f"export_{instrument}_{timeframe}",
                    show_errors=True
                )
            else:
                export_path = safe_execute(
                    self.csv_exporter.export_empty_results,
                    instrument, timeframe, "No patterns detected",
                    context=f"export_empty_{instrument}_{timeframe}",
                    show_errors=True
                )
            
            if export_path:
                print(f"✅ Results exported to: {export_path}")
                
                # Show export summary
                summary = self.csv_exporter.get_export_summary(export_path)
                if 'error' not in summary:
                    print(f"📋 Export Summary:")
                    print(f"   - File size: {summary.get('file_size_bytes', 0)} bytes")
                    print(f"   - Rows: {summary.get('row_count', 0)}")
                    print(f"   - Created: {summary.get('created_at', 'Unknown')}")
                
                return export_path
            else:
                print("❌ Export failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Error in analysis and export: {e}")
            print(f"❌ Analysis failed: {e}")
            return None
    
    def run_interactive_mode(self) -> None:
        """Run interactive command-line mode for pattern detection and export."""
        try:
            print("\n" + "=" * 60)
            print("📈 Stock Pattern Detector - CLI Export Tool")
            print("=" * 60)
            
            # Select instrument
            instrument = self.prompt_dataset_selection()
            if not instrument:
                return
            
            # Select timeframe
            timeframe_selection = self.prompt_timeframe_selection()
            if not timeframe_selection:
                return
            
            # Select patterns
            selected_patterns = self.prompt_pattern_selection()
            if not selected_patterns:
                return
            
            # Determine timeframes to process
            if timeframe_selection == "all":
                timeframes_to_process = self.timeframes
                print(f"\n🔄 Processing all timeframes: {', '.join(timeframes_to_process)}")
            else:
                timeframes_to_process = [timeframe_selection]
            
            # Run analysis for each timeframe
            export_paths = []
            for i, timeframe in enumerate(timeframes_to_process, 1):
                print(f"\n[{i}/{len(timeframes_to_process)}] Processing {timeframe}...")
                export_path = self.run_analysis_and_export(instrument, timeframe, selected_patterns)
                if export_path:
                    export_paths.append(export_path)
            
            # Summary
            if export_paths:
                print(f"\n🎉 Analysis completed successfully!")
                print(f"📁 Generated {len(export_paths)} export files:")
                for path in export_paths:
                    print(f"   • {path}")
            else:
                print(f"\n❌ Analysis failed. Check the logs for details.")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            self.logger.error(f"Error in interactive mode: {e}")
            print(f"❌ Unexpected error: {e}")
    
    def run_batch_mode(self, config: Dict[str, Any]) -> List[str]:
        """
        Run batch export for multiple configurations.
        
        Args:
            config: Dictionary with batch configuration
            
        Returns:
            List of exported file paths
        """
        try:
            exported_files = []
            
            instruments = config.get('instruments', [])
            timeframes = config.get('timeframes', ['5min'])
            patterns = config.get('patterns', list(self.pattern_detectors.keys()))
            
            print(f"\n🔄 Starting batch export...")
            print(f"📊 Instruments: {len(instruments)}")
            print(f"⏰ Timeframes: {len(timeframes)}")
            print(f"🔍 Patterns: {len(patterns)}")
            
            total_jobs = len(instruments) * len(timeframes)
            current_job = 0
            
            for instrument in instruments:
                for timeframe in timeframes:
                    current_job += 1
                    print(f"\n[{current_job}/{total_jobs}] Processing {instrument} ({timeframe})...")
                    
                    export_path = self.run_analysis_and_export(instrument, timeframe, patterns)
                    if export_path:
                        exported_files.append(export_path)
            
            print(f"\n🎉 Batch export completed!")
            print(f"📁 Exported {len(exported_files)} files")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error in batch mode: {e}")
            print(f"❌ Batch export failed: {e}")
            return []


def main():
    """Main entry point for CLI export tool."""
    try:
        cli = CommandLineInterface()
        
        # Check if running in batch mode (with arguments)
        if len(sys.argv) > 1:
            # Simple argument parsing for batch mode
            if sys.argv[1] == '--batch':
                # Example batch configuration
                batch_config = {
                    'instruments': cli.get_available_instruments(),
                    'timeframes': ['5min', '15min', '1hour'],
                    'patterns': ['dragonfly_doji', 'hammer', 'rising_window']
                }
                cli.run_batch_mode(batch_config)
            else:
                print("Usage: python export_cli.py [--batch]")
        else:
            # Run interactive mode
            cli.run_interactive_mode()
            
    except Exception as e:
        logger.error(f"CLI main function error: {e}")
        print(f"❌ Application error: {e}")


if __name__ == "__main__":
    main()