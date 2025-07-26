# Stock Pattern Detector - Main Streamlit Application
"""
Main Streamlit application for the Stock Pattern Detector.

This application provides an interactive dashboard for detecting and visualizing
candlestick patterns in stock market data from local CSV files.
"""

import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

# Import application modules
from data.loader import CSVLoader
from data.aggregator import DataAggregator
from patterns.dragonfly_doji import DragonflyDojiDetector
from patterns.hammer import HammerDetector
from patterns.rising_window import RisingWindowDetector
from patterns.evening_star import EveningStarDetector
from patterns.three_white_soldiers import ThreeWhiteSoldiersDetector
from patterns.base import PatternResult
from visualization.charts import ChartRenderer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Pattern Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data_loader' not in st.session_state:
    st.session_state.data_loader = CSVLoader()
if 'data_aggregator' not in st.session_state:
    st.session_state.data_aggregator = DataAggregator()
if 'chart_renderer' not in st.session_state:
    st.session_state.chart_renderer = ChartRenderer()
if 'pattern_results' not in st.session_state:
    st.session_state.pattern_results = []
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False

# Pattern detector mapping
PATTERN_DETECTORS = {
    'Dragonfly Doji': DragonflyDojiDetector(),
    'Hammer': HammerDetector(),
    'Rising Window': RisingWindowDetector(),
    'Evening Star': EveningStarDetector(),
    'Three White Soldiers': ThreeWhiteSoldiersDetector()
}


def create_sidebar() -> Dict[str, Any]:
    """
    Create the sidebar with instrument selection, time period controls,
    and pattern selection checkboxes with comprehensive error handling.
    
    Returns:
        Dictionary with user selections
    """
    from utils.error_handler import error_handler, safe_execute
    
    st.sidebar.header("📊 Analysis Configuration")
    
    # Instrument selection
    st.sidebar.subheader("🏢 Select Instrument")
    
    # Get available instruments with error handling
    try:
        with st.spinner("Loading available instruments..."):
            available_instruments = safe_execute(
                st.session_state.data_loader.get_available_instruments,
                context="sidebar_get_instruments",
                fallback_result=[],
                show_errors=True
            )
        
        if not available_instruments:
            st.sidebar.error("❌ No instruments found in 5Scripts directory")
            st.sidebar.info("Please ensure CSV files are available in the 5Scripts folder")
            
            # Show troubleshooting suggestions
            with st.sidebar.expander("🔧 Troubleshooting", expanded=True):
                st.write("**Possible solutions:**")
                st.write("1. Check if the 5Scripts directory exists")
                st.write("2. Verify CSV files are present in instrument folders")
                st.write("3. Ensure file permissions allow reading")
                st.write("4. Try refreshing the page")
            
            return None
            
    except Exception as e:
        st.sidebar.error("❌ Error loading instruments")
        error_handler.handle_error(e, "sidebar_instrument_loading", show_user_message=True)
        return None
    
    selected_instrument = st.sidebar.selectbox(
        "Choose an instrument:",
        available_instruments,
        help="Select the stock instrument to analyze"
    )
    
    # Time period selection
    st.sidebar.subheader("⏰ Time Period")
    
    supported_timeframes = st.session_state.data_aggregator.get_supported_timeframes()
    selected_timeframe = st.sidebar.selectbox(
        "Select timeframe:",
        supported_timeframes,
        index=0,  # Default to 1min
        help="Choose the time period for candlestick aggregation"
    )
    
    # Pattern type selection
    st.sidebar.subheader("🔍 Pattern Selection")
    st.sidebar.write("Select patterns to detect:")
    
    selected_patterns = {}
    for pattern_name in PATTERN_DETECTORS.keys():
        selected_patterns[pattern_name] = st.sidebar.checkbox(
            pattern_name,
            value=True,  # Default all patterns selected
            help=f"Detect {pattern_name} patterns"
        )
    
    # Analysis trigger section
    st.sidebar.subheader("🚀 Analysis")
    
    # Show data info if instrument is selected
    if selected_instrument:
        with st.sidebar.expander("📋 Data Information", expanded=False):
            try:
                # Use a placeholder for loading state
                info_placeholder = st.empty()
                info_placeholder.info("Loading data information...")
                
                data_info = safe_execute(
                    st.session_state.data_loader.get_data_info,
                    selected_instrument,
                    context=f"sidebar_data_info_{selected_instrument}",
                    fallback_result=None,
                    show_errors=False
                )
                
                info_placeholder.empty()
                
                if data_info and data_info.get('total_records', 0) > 0:
                    st.write(f"**Records:** {data_info['total_records']:,}")
                    st.write(f"**Date Range:**")
                    st.write(f"  From: {data_info['date_range']['start']}")
                    st.write(f"  To: {data_info['date_range']['end']}")
                    
                    if data_info.get('price_range'):
                        st.write(f"**Price Range:** ${data_info['price_range']['min']:.2f} - ${data_info['price_range']['max']:.2f}")
                    
                    if data_info.get('total_volume'):
                        st.write(f"**Total Volume:** {data_info['total_volume']:,}")
                    
                    if data_info.get('avg_volume'):
                        st.write(f"**Avg Volume:** {data_info['avg_volume']:,}")
                    
                    # Show data quality status
                    status = data_info.get('status', 'Unknown')
                    if 'successfully' in status.lower():
                        st.success(f"✅ {status}")
                    elif 'error' in status.lower():
                        st.warning(f"⚠️ {status}")
                    else:
                        st.info(f"ℹ️ {status}")
                        
                else:
                    st.warning("⚠️ Unable to load data information")
                    st.write("This may indicate data loading issues.")
                    
            except Exception as e:
                st.error("❌ Error loading data information")
                logger.error(f"Error in sidebar data info for {selected_instrument}: {e}")
    
    return {
        'instrument': selected_instrument,
        'timeframe': selected_timeframe,
        'patterns': selected_patterns,
        'available_instruments': available_instruments
    }


def create_analysis_button(config: Dict[str, Any]) -> bool:
    """
    Create the analysis trigger button with validation and loading states.
    
    Args:
        config: Configuration dictionary from sidebar
        
    Returns:
        True if analysis should be triggered, False otherwise
    """
    if not config:
        return False
    
    # Validate configuration
    selected_pattern_count = sum(1 for selected in config['patterns'].values() if selected)
    
    if selected_pattern_count == 0:
        st.sidebar.warning("⚠️ Please select at least one pattern to detect")
        analysis_disabled = True
    else:
        analysis_disabled = False
    
    # Analysis button
    if st.sidebar.button(
        "🔍 Start Analysis",
        disabled=analysis_disabled,
        help="Begin pattern detection analysis",
        type="primary"
    ):
        return True
    
    # Show analysis summary
    if not analysis_disabled:
        st.sidebar.info(
            f"Ready to analyze **{config['instrument']}** "
            f"on **{config['timeframe']}** timeframe for "
            f"**{selected_pattern_count}** pattern types"
        )
    
    return False


def perform_pattern_analysis(config: Dict[str, Any]) -> None:
    """
    Perform the pattern detection analysis with comprehensive error handling and progress indicators.
    
    Args:
        config: Configuration dictionary with user selections
    """
    from utils.error_handler import (
        error_handler, DataLoadingError, DataValidationError, 
        PatternDetectionError, safe_execute
    )
    from utils.logging_config import log_performance
    
    analysis_start_time = time.time()
    
    # Create progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    error_container = st.empty()
    
    try:
        # Validate configuration
        if not config or not config.get('instrument'):
            raise ValueError("Invalid configuration: missing instrument")
        
        selected_patterns = [name for name, selected in config['patterns'].items() if selected]
        if not selected_patterns:
            raise ValueError("No patterns selected for detection")
        
        # Step 1: Load data with error handling
        status_text.text("📥 Loading instrument data...")
        progress_bar.progress(10)
        
        try:
            raw_data = st.session_state.data_loader.load_instrument_data(config['instrument'])
            
            if raw_data is None or raw_data.empty:
                raise DataLoadingError(
                    f"No data available for instrument '{config['instrument']}'",
                    error_code="NO_INSTRUMENT_DATA",
                    context={'instrument': config['instrument']}
                )
            
            logger.info(f"Loaded {len(raw_data)} records for {config['instrument']}")
            
        except (DataLoadingError, DataValidationError) as e:
            error_handler.handle_error(e, "data_loading", show_user_message=True)
            return
        except Exception as e:
            error_handler.handle_error(
                DataLoadingError(f"Unexpected error loading data: {str(e)}", "UNEXPECTED_LOAD_ERROR"),
                "data_loading", show_user_message=True
            )
            return
        
        # Step 2: Aggregate data with error handling
        status_text.text(f"⚙️ Aggregating data to {config['timeframe']} timeframe...")
        progress_bar.progress(30)
        
        try:
            aggregated_data = st.session_state.data_aggregator.aggregate_data(raw_data, config['timeframe'])
            
            if aggregated_data is None or aggregated_data.empty:
                raise DataValidationError(
                    f"Data aggregation to {config['timeframe']} produced no results",
                    error_code="AGGREGATION_NO_RESULTS",
                    context={
                        'timeframe': config['timeframe'],
                        'input_records': len(raw_data)
                    }
                )
            
            logger.info(f"Aggregated to {len(aggregated_data)} {config['timeframe']} candles")
            
        except DataValidationError as e:
            error_handler.handle_error(e, "data_aggregation", show_user_message=True)
            return
        except Exception as e:
            error_handler.handle_error(
                DataValidationError(f"Unexpected error during aggregation: {str(e)}", "UNEXPECTED_AGGREGATION_ERROR"),
                "data_aggregation", show_user_message=True
            )
            return
        
        # Prepare data for pattern detection
        try:
            if 'datetime' in aggregated_data.columns:
                pattern_data = aggregated_data.set_index('datetime')
            else:
                pattern_data = aggregated_data
                
            # Validate pattern data
            is_valid, issues = error_handler.validate_data_integrity(
                pattern_data, "pattern_detection_input"
            )
            
            if not is_valid:
                error_handler.show_warning_message(
                    "Data quality issues detected before pattern detection",
                    issues[:3]
                )
                
        except Exception as e:
            error_handler.handle_error(
                DataValidationError(f"Error preparing data for pattern detection: {str(e)}", "DATA_PREP_ERROR"),
                "data_preparation", show_user_message=True
            )
            return
        
        # Step 3: Detect patterns with comprehensive error handling
        status_text.text("🔍 Detecting patterns...")
        progress_bar.progress(50)
        
        all_patterns = []
        pattern_errors = []
        selected_detectors = [
            (name, detector) for name, detector in PATTERN_DETECTORS.items()
            if config['patterns'].get(name, False)
        ]
        
        for i, (pattern_name, detector) in enumerate(selected_detectors):
            try:
                # Update progress
                detection_progress = 50 + (40 * i / len(selected_detectors))
                progress_bar.progress(int(detection_progress))
                status_text.text(f"🔍 Detecting {pattern_name} patterns...")
                
                # Detect patterns with timeout protection
                pattern_start_time = time.time()
                patterns = safe_execute(
                    detector.detect,
                    pattern_data,
                    config['timeframe'],
                    context=f"detect_{pattern_name}",
                    fallback_result=[],
                    show_errors=False
                )
                
                pattern_duration = time.time() - pattern_start_time
                
                if patterns is None:
                    patterns = []
                    pattern_errors.append(f"{pattern_name}: Detection returned None")
                elif pattern_duration > 10.0:  # Log slow pattern detection
                    log_performance(f"detect_{pattern_name}", pattern_duration, config['timeframe'])
                
                all_patterns.extend(patterns)
                
                # Update status
                status_text.text(f"🔍 Found {len(patterns)} {pattern_name} patterns...")
                logger.info(f"Detected {len(patterns)} {pattern_name} patterns in {pattern_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Error detecting {pattern_name} patterns: {str(e)}"
                pattern_errors.append(f"{pattern_name}: {str(e)[:100]}...")
                logger.error(error_msg)
                
                # Show warning but continue with other patterns
                error_container.warning(f"⚠️ {error_msg}")
        
        # Step 4: Validate and complete analysis
        status_text.text("✅ Finalizing analysis...")
        progress_bar.progress(90)
        
        # Show pattern detection errors if any
        if pattern_errors:
            error_handler.show_warning_message(
                f"Some pattern detectors encountered issues",
                pattern_errors[:5]  # Show first 5 errors
            )
        
        # Validate pattern results
        valid_patterns = []
        for pattern in all_patterns:
            try:
                # Basic validation of pattern result
                if (hasattr(pattern, 'datetime') and hasattr(pattern, 'pattern_type') and 
                    hasattr(pattern, 'confidence') and 0 <= pattern.confidence <= 1):
                    valid_patterns.append(pattern)
                else:
                    logger.warning(f"Invalid pattern result: {pattern}")
            except Exception as e:
                logger.warning(f"Error validating pattern result: {e}")
        
        # Complete analysis
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
        # Store results in session state
        st.session_state.pattern_results = valid_patterns
        st.session_state.current_data = aggregated_data
        st.session_state.analysis_complete = True
        
        # Calculate and log total analysis time
        total_duration = time.time() - analysis_start_time
        log_performance("complete_pattern_analysis", total_duration, 
                       f"{config['instrument']}_{config['timeframe']}")
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        error_container.empty()
        
        # Show success message with details
        success_msg = f"🎉 Analysis complete! Found {len(valid_patterns)} patterns in {config['instrument']}"
        if pattern_errors:
            success_msg += f" (with {len(pattern_errors)} detector warnings)"
        
        st.success(success_msg)
        
        # Show analysis summary
        if valid_patterns:
            pattern_summary = {}
            for pattern in valid_patterns:
                pattern_type = pattern.pattern_type.replace('_', ' ').title()
                pattern_summary[pattern_type] = pattern_summary.get(pattern_type, 0) + 1
            
            summary_text = ", ".join([f"{count} {pattern}" for pattern, count in pattern_summary.items()])
            st.info(f"📊 Pattern breakdown: {summary_text}")
        
    except Exception as e:
        # Handle unexpected errors in the analysis process
        error_handler.handle_error(
            PatternDetectionError(f"Unexpected error during pattern analysis: {str(e)}", "ANALYSIS_UNEXPECTED_ERROR"),
            "pattern_analysis", show_user_message=True
        )
        
        # Clear progress indicators on error
        try:
            progress_bar.empty()
            status_text.empty()
            error_container.empty()
        except:
            pass


def display_main_content(config: Dict[str, Any]) -> None:
    """
    Display the main content area with charts and results.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    # Main title and description
    st.title("📈 Stock Pattern Detector Dashboard")
    st.markdown("""
    Analyze historical stock data to detect candlestick patterns. Select an instrument, 
    timeframe, and patterns from the sidebar, then click **Start Analysis** to begin.
    """)
    
    if not config:
        st.info("👈 Please configure analysis settings in the sidebar to get started.")
        return
    
    # Check if analysis button was clicked
    if create_analysis_button(config):
        # Clear previous results
        st.session_state.pattern_results = []
        st.session_state.current_data = None
        st.session_state.analysis_complete = False
        
        # Perform analysis
        perform_pattern_analysis(config)
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.current_data is not None:
        display_analysis_results(config)
    elif not st.session_state.analysis_complete:
        # Show placeholder content
        display_placeholder_content()


def display_analysis_results(config: Dict[str, Any]) -> None:
    """
    Display the analysis results with charts and pattern information.
    
    Args:
        config: Configuration dictionary from sidebar
    """
    try:
        data = st.session_state.current_data
        patterns = st.session_state.pattern_results
        
        # Results summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Data Points", f"{len(data):,}")
        
        with col2:
            st.metric("🔍 Patterns Found", len(patterns))
        
        with col3:
            if patterns:
                avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
                st.metric("📈 Avg Confidence", f"{avg_confidence:.1%}")
            else:
                st.metric("📈 Avg Confidence", "N/A")
        
        with col4:
            st.metric("⏰ Timeframe", config['timeframe'])
        
        # Main chart
        st.subheader(f"📈 {config['instrument']} - {config['timeframe']} Chart")
        
        if not data.empty:
            # Create candlestick chart
            chart = st.session_state.chart_renderer.create_candlestick_chart(
                data,
                title=f"{config['instrument']} - {config['timeframe']} Candlestick Chart",
                height=600,
                show_volume=True
            )
            
            # Highlight patterns if any found
            if patterns:
                chart = st.session_state.chart_renderer.highlight_patterns(
                    chart, patterns, data, advanced_highlighting=True
                )
            
            # Add interactivity
            chart = st.session_state.chart_renderer.add_interactivity(chart)
            
            # Display chart
            st.plotly_chart(chart, use_container_width=True)
            
        else:
            st.warning("⚠️ No data available for charting")
        
        # Pattern results table and export functionality
        if patterns:
            display_pattern_results_table(patterns, data, config)
        
        # Pattern summary if patterns found
        if patterns:
            st.subheader("📋 Pattern Analysis Summary")
            
            # Create two columns for summary charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Pattern distribution chart
                summary_chart = st.session_state.chart_renderer.create_pattern_summary_chart(patterns)
                st.plotly_chart(summary_chart, use_container_width=True)
            
            with col2:
                # Pattern statistics
                st.write("**Pattern Statistics:**")
                
                pattern_stats = {}
                for pattern in patterns:
                    pattern_type = pattern.pattern_type.replace('_', ' ').title()
                    if pattern_type not in pattern_stats:
                        pattern_stats[pattern_type] = {
                            'count': 0,
                            'total_confidence': 0.0,
                            'timeframes': set()
                        }
                    pattern_stats[pattern_type]['count'] += 1
                    pattern_stats[pattern_type]['total_confidence'] += pattern.confidence
                    pattern_stats[pattern_type]['timeframes'].add(pattern.timeframe)
                
                for pattern_type, stats in pattern_stats.items():
                    avg_confidence = stats['total_confidence'] / stats['count']
                    st.write(f"**{pattern_type}:** {stats['count']} patterns (avg confidence: {avg_confidence:.1%})")
        
        else:
            st.info("🔍 No patterns detected in the selected timeframe and criteria.")
            st.write("Try adjusting the timeframe or selecting different pattern types.")
    
    except Exception as e:
        logger.error(f"Error displaying analysis results: {e}")
        st.error(f"❌ Error displaying results: {str(e)}")


def display_pattern_results_table(patterns: List[PatternResult], data: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Display pattern results table with sorting, navigation, and export functionality.
    
    Args:
        patterns: List of detected patterns
        data: OHLCV data for reference
        config: Configuration dictionary
    """
    try:
        st.subheader("🔍 Pattern Detection Results")
        
        if not patterns:
            st.info("No patterns detected.")
            return
        
        # Sort patterns by datetime in descending order (most recent first)
        sorted_patterns = sorted(patterns, key=lambda p: p.datetime, reverse=True)
        
        # Create results DataFrame for display
        results_data = []
        for i, pattern in enumerate(sorted_patterns):
            results_data.append({
                'Index': i,
                'DateTime': pattern.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'Pattern Type': pattern.pattern_type.replace('_', ' ').title(),
                'Timeframe': pattern.timeframe,
                'Confidence': f"{pattern.confidence:.1%}",
                'Confidence_Raw': pattern.confidence,  # For sorting
                'Description': pattern.description if pattern.description else "Candlestick pattern detected"
            })
        
        results_df = pd.DataFrame(results_data)
        
        # Display controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{len(patterns)} patterns found** (sorted by date, newest first)")
        
        with col2:
            # Sort options
            sort_options = {
                'DateTime (Newest First)': ('DateTime', False),
                'DateTime (Oldest First)': ('DateTime', True),
                'Confidence (Highest First)': ('Confidence_Raw', False),
                'Confidence (Lowest First)': ('Confidence_Raw', True),
                'Pattern Type': ('Pattern Type', True)
            }
            
            selected_sort = st.selectbox(
                "Sort by:",
                list(sort_options.keys()),
                index=0,
                help="Choose how to sort the results"
            )
            
            sort_column, ascending = sort_options[selected_sort]
            if sort_column == 'Confidence_Raw':
                results_df_sorted = results_df.sort_values(sort_column, ascending=ascending)
            else:
                results_df_sorted = results_df.sort_values(sort_column, ascending=ascending)
        
        with col3:
            # Export functionality
            if st.button("📥 Export CSV", help="Download results as CSV file"):
                export_csv = create_export_csv(sorted_patterns, config)
                st.download_button(
                    label="Download CSV",
                    data=export_csv,
                    file_name=f"pattern_results_{config['instrument']}_{config['timeframe']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    help="Download the pattern detection results"
                )
        
        # Display the results table with clickable rows
        st.write("**Click on a row to navigate to that pattern on the chart:**")
        
        # Create interactive table
        display_df = results_df_sorted[['DateTime', 'Pattern Type', 'Timeframe', 'Confidence', 'Description']].copy()
        
        # Use st.dataframe with selection mode
        selected_rows = st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row"
        )
        
        # Handle row selection for chart navigation
        if selected_rows and 'selection' in selected_rows and selected_rows['selection']['rows']:
            selected_row_idx = selected_rows['selection']['rows'][0]
            selected_pattern_idx = results_df_sorted.iloc[selected_row_idx]['Index']
            selected_pattern = sorted_patterns[selected_pattern_idx]
            
            # Display selected pattern details
            display_selected_pattern_details(selected_pattern, data)
            
            # Create focused chart view
            create_pattern_focused_chart(selected_pattern, data, config)
        
        # Pattern type filter
        st.subheader("🔧 Filter Results")
        
        # Get unique pattern types
        unique_patterns = list(set(p.pattern_type.replace('_', ' ').title() for p in patterns))
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_pattern_types = st.multiselect(
                "Filter by Pattern Type:",
                unique_patterns,
                default=unique_patterns,
                help="Select which pattern types to display"
            )
        
        with col2:
            confidence_range = st.slider(
                "Confidence Range:",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                format="%.0%%",
                help="Filter patterns by confidence level"
            )
        
        # Apply filters
        if selected_pattern_types != unique_patterns or confidence_range != (0.0, 1.0):
            filtered_patterns = [
                p for p in patterns
                if (p.pattern_type.replace('_', ' ').title() in selected_pattern_types and
                    confidence_range[0] <= p.confidence <= confidence_range[1])
            ]
            
            if filtered_patterns:
                st.write(f"**Filtered Results: {len(filtered_patterns)} patterns**")
                display_filtered_results_summary(filtered_patterns)
            else:
                st.info("No patterns match the selected filters.")
    
    except Exception as e:
        logger.error(f"Error displaying pattern results table: {e}")
        st.error(f"❌ Error displaying results table: {str(e)}")


def display_selected_pattern_details(pattern: PatternResult, data: pd.DataFrame) -> None:
    """
    Display detailed information about the selected pattern.
    
    Args:
        pattern: Selected pattern result
        data: OHLCV data for context
    """
    try:
        st.subheader("🎯 Selected Pattern Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**Pattern:** {pattern.pattern_type.replace('_', ' ').title()}")
            st.write(f"**DateTime:** {pattern.datetime.strftime('%Y-%m-%d %H:%M:%S')}")
            st.write(f"**Confidence:** {pattern.confidence:.1%}")
        
        with col2:
            st.write(f"**Timeframe:** {pattern.timeframe}")
            st.write(f"**Candle Index:** {pattern.candle_index}")
            if pattern.description:
                st.write(f"**Description:** {pattern.description}")
        
        with col3:
            # Find the corresponding candle data
            try:
                if 'datetime' in data.columns:
                    mask = data['datetime'] == pattern.datetime
                    if mask.any():
                        candle = data[mask].iloc[0]
                        st.write("**OHLCV Data:**")
                        st.write(f"Open: ${candle['open']:.2f}")
                        st.write(f"High: ${candle['high']:.2f}")
                        st.write(f"Low: ${candle['low']:.2f}")
                        st.write(f"Close: ${candle['close']:.2f}")
                        if 'volume' in candle:
                            st.write(f"Volume: {candle['volume']:,.0f}")
            except Exception as e:
                logger.warning(f"Could not display candle data: {e}")
    
    except Exception as e:
        logger.error(f"Error displaying selected pattern details: {e}")


def create_pattern_focused_chart(pattern: PatternResult, data: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Create a focused chart view centered on the selected pattern.
    
    Args:
        pattern: Selected pattern to focus on
        data: OHLCV data
        config: Configuration dictionary
    """
    try:
        st.subheader("📈 Pattern Focus View")
        
        # Create detailed pattern view
        focused_chart = st.session_state.chart_renderer.create_pattern_detail_view(
            data, 
            pattern,
            context_candles=20,
            title=f"{pattern.pattern_type.replace('_', ' ').title()} Pattern - {config['instrument']}"
        )
        
        # Display the focused chart
        st.plotly_chart(focused_chart, use_container_width=True)
        
        # Add navigation buttons for multiple patterns
        if len(st.session_state.pattern_results) > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col1:
                if st.button("⬅️ Previous Pattern"):
                    navigate_to_adjacent_pattern(pattern, -1)
            
            with col2:
                st.write(f"Pattern {get_pattern_index(pattern) + 1} of {len(st.session_state.pattern_results)}")
            
            with col3:
                if st.button("Next Pattern ➡️"):
                    navigate_to_adjacent_pattern(pattern, 1)
    
    except Exception as e:
        logger.error(f"Error creating pattern focused chart: {e}")
        st.error(f"❌ Error creating focused chart: {str(e)}")


def create_export_csv(patterns: List[PatternResult], config: Dict[str, Any]) -> str:
    """
    Create CSV export data for pattern results.
    
    Args:
        patterns: List of patterns to export
        config: Configuration dictionary
        
    Returns:
        CSV data as string
    """
    try:
        # Create export data
        export_data = []
        for pattern in patterns:
            export_data.append({
                'DateTime': pattern.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                'Pattern_Type': pattern.pattern_type,
                'Pattern_Name': pattern.pattern_type.replace('_', ' ').title(),
                'Timeframe': pattern.timeframe,
                'Confidence': pattern.confidence,
                'Confidence_Percent': f"{pattern.confidence:.1%}",
                'Candle_Index': pattern.candle_index,
                'Description': pattern.description if pattern.description else "",
                'Instrument': config['instrument'],
                'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Convert to DataFrame and then CSV
        export_df = pd.DataFrame(export_data)
        return export_df.to_csv(index=False)
    
    except Exception as e:
        logger.error(f"Error creating export CSV: {e}")
        return "Error creating export data"


def display_filtered_results_summary(filtered_patterns: List[PatternResult]) -> None:
    """
    Display summary of filtered results.
    
    Args:
        filtered_patterns: List of patterns after filtering
    """
    try:
        # Create summary statistics
        pattern_counts = {}
        total_confidence = 0
        
        for pattern in filtered_patterns:
            pattern_type = pattern.pattern_type.replace('_', ' ').title()
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            total_confidence += pattern.confidence
        
        avg_confidence = total_confidence / len(filtered_patterns) if filtered_patterns else 0
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Pattern Distribution:**")
            for pattern_type, count in pattern_counts.items():
                st.write(f"• {pattern_type}: {count}")
        
        with col2:
            st.write(f"**Average Confidence:** {avg_confidence:.1%}")
            st.write(f"**Date Range:** {min(p.datetime for p in filtered_patterns).strftime('%Y-%m-%d')} to {max(p.datetime for p in filtered_patterns).strftime('%Y-%m-%d')}")
    
    except Exception as e:
        logger.error(f"Error displaying filtered results summary: {e}")


def navigate_to_adjacent_pattern(current_pattern: PatternResult, direction: int) -> None:
    """
    Navigate to the next or previous pattern.
    
    Args:
        current_pattern: Currently selected pattern
        direction: -1 for previous, 1 for next
    """
    try:
        patterns = st.session_state.pattern_results
        current_index = get_pattern_index(current_pattern)
        
        new_index = current_index + direction
        if 0 <= new_index < len(patterns):
            # This would trigger a rerun with the new pattern selected
            # In a full implementation, this would update the selection state
            st.info(f"Navigation to pattern {new_index + 1} (implementation requires state management)")
    
    except Exception as e:
        logger.error(f"Error navigating to adjacent pattern: {e}")


def get_pattern_index(pattern: PatternResult) -> int:
    """
    Get the index of a pattern in the results list.
    
    Args:
        pattern: Pattern to find
        
    Returns:
        Index of the pattern, or 0 if not found
    """
    try:
        patterns = st.session_state.pattern_results
        for i, p in enumerate(patterns):
            if (p.datetime == pattern.datetime and 
                p.pattern_type == pattern.pattern_type and
                p.confidence == pattern.confidence):
                return i
        return 0
    except Exception as e:
        logger.error(f"Error getting pattern index: {e}")
        return 0


def display_placeholder_content() -> None:
    """Display placeholder content when no analysis has been performed."""
    st.info("👈 Configure your analysis settings in the sidebar and click **Start Analysis** to begin.")
    
    # Show feature overview
    st.subheader("🌟 Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **📊 Data Analysis**
        - Load data from local CSV files
        - Multiple timeframe aggregation
        - Real-time data validation
        """)
    
    with col2:
        st.markdown("""
        **🔍 Pattern Detection**
        - Dragonfly Doji patterns
        - Hammer patterns
        - Rising Window patterns
        - Evening Star patterns
        - Three White Soldiers patterns
        """)
    
    with col3:
        st.markdown("""
        **📈 Visualization**
        - Interactive candlestick charts
        - Pattern highlighting
        - Volume analysis
        - Responsive design
        """)
    
    # Show sample chart placeholder
    st.subheader("📈 Sample Chart")
    st.info("Your interactive candlestick chart with pattern highlights will appear here after analysis.")


def main():
    """Main application entry point with comprehensive error handling."""
    from utils.error_handler import error_handler, get_error_summary
    from utils.logging_config import get_logger
    
    logger = get_logger(__name__)
    
    try:
        # Initialize error tracking
        if 'error_summary_shown' not in st.session_state:
            st.session_state.error_summary_shown = False
        
        # Create sidebar configuration
        config = create_sidebar()
        
        # Display main content
        display_main_content(config)
        
        # Show error summary in sidebar if there are errors
        error_summary = get_error_summary()
        if (error_summary.get('total_errors', 0) > 0 or error_summary.get('total_warnings', 0) > 0):
            with st.sidebar.expander("⚠️ Error Summary", expanded=False):
                st.write(f"**Errors:** {error_summary.get('total_errors', 0)}")
                st.write(f"**Warnings:** {error_summary.get('total_warnings', 0)}")
                
                if error_summary.get('recent_errors'):
                    st.write("**Recent Issues:**")
                    for error in error_summary['recent_errors'][-3:]:  # Show last 3
                        st.write(f"• {error['type']}: {error['message'][:50]}...")
        
        # Footer
        st.markdown("---")
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(
                "**Stock Pattern Detector** | Built with Streamlit | "
                "Analyze candlestick patterns in your local stock data"
            )
        
        with col2:
            if st.button("🔄 Clear Cache"):
                st.cache_data.clear()
                st.success("Cache cleared!")
                st.rerun()
        
        with col3:
            if st.button("📊 Show Logs"):
                error_summary = get_error_summary()
                st.json(error_summary)
        
    except Exception as e:
        # Critical error in main application
        logger.critical(f"Critical error in main application: {e}")
        
        st.error("❌ **Critical Application Error**")
        st.write("The application encountered a critical error and cannot continue.")
        
        with st.expander("🔧 Error Details", expanded=True):
            st.code(str(e))
            st.write("**Suggested Actions:**")
            st.write("1. Refresh the page (F5)")
            st.write("2. Clear browser cache")
            st.write("3. Check the application logs")
            st.write("4. Restart the application")
        
        # Try to show basic error recovery options
        if st.button("🔄 Restart Application"):
            st.cache_data.clear()
            st.rerun()on as e:
        logger.error(f"Application error: {e}")
        st.error(f"❌ Application error: {str(e)}")
        st.info("Please refresh the page and try again.")


if __name__ == "__main__":
    main()