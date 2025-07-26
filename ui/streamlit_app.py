#!/usr/bin/env python3
"""
Stock Pattern Detector - Simplified Streamlit Application

Clean, focused interface for detecting and visualizing candlestick patterns
with improved usability and clear chart displays.
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Stock Pattern Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Pattern detector mapping
PATTERN_DETECTORS = {
    'Dragonfly Doji': DragonflyDojiDetector(),
    'Hammer': HammerDetector(),
    'Rising Window': RisingWindowDetector(),
    'Evening Star': EveningStarDetector(),
    'Three White Soldiers': ThreeWhiteSoldiersDetector()
}


def main():
    """Main application entry point"""
    try:
        # Header
        st.title("📈 Stock Pattern Detector")
        st.markdown("### Simple, focused pattern analysis for trading decisions")
        
        # Initialize session state
        initialize_session_state()
        
        # Create sidebar controls
        config = create_sidebar()
        if not config:
            st.error("❌ Unable to load configuration")
            return
        
        # Main content
        if config['run_analysis']:
            run_analysis(config)
        else:
            show_welcome_screen()
            
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        logger.error(f"Main application error: {e}", exc_info=True)


def initialize_session_state():
    """Initialize session state variables"""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = CSVLoader()
    if 'data_aggregator' not in st.session_state:
        st.session_state.data_aggregator = DataAggregator()
    if 'chart_renderer' not in st.session_state:
        st.session_state.chart_renderer = ChartRenderer()


def create_sidebar():
    """Create clean sidebar configuration"""
    st.sidebar.header("📊 Configuration")
    
    # Get available instruments
    try:
        instruments = st.session_state.data_loader.get_available_instruments()
        if not instruments:
            st.sidebar.error("No instruments found")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading instruments: {e}")
        return None
    
    # Instrument selection
    selected_instrument = st.sidebar.selectbox(
        "📈 Select Stock",
        instruments,
        help="Choose the stock to analyze"
    )
    
    # Timeframe selection
    timeframes = st.session_state.data_aggregator.get_supported_timeframes()
    selected_timeframe = st.sidebar.selectbox(
        "⏱️ Timeframe",
        timeframes,
        index=1,  # Default to 5min
        help="Select analysis timeframe"
    )
    
    # Pattern selection
    st.sidebar.subheader("🔍 Patterns to Detect")
    selected_patterns = {}
    for name in PATTERN_DETECTORS.keys():
        selected_patterns[name] = st.sidebar.checkbox(
            name.replace('_', ' '),
            value=True,
            help=f"Detect {name} patterns"
        )
    
    # Quick settings
    st.sidebar.subheader("⚙️ Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_candles = st.number_input(
            "Max Candles",
            min_value=50,
            max_value=1000,
            value=300,
            step=50,
            help="Maximum candles to display"
        )
    with col2:
        chart_height = st.number_input(
            "Chart Height",
            min_value=300,
            max_value=800,
            value=500,
            step=50,
            help="Chart height in pixels"
        )
    
    # Analysis button
    run_analysis = st.sidebar.button(
        "🚀 Analyze Patterns",
        type="primary",
        use_container_width=True
    )
    
    return {
        'instrument': selected_instrument,
        'timeframe': selected_timeframe,
        'patterns': selected_patterns,
        'max_candles': max_candles,
        'chart_height': chart_height,
        'run_analysis': run_analysis
    }


def show_welcome_screen():
    """Show welcome screen when no analysis is running"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 👋 Welcome to Stock Pattern Detector
        
        **Get Started:**
        1. Select a stock instrument from the sidebar
        2. Choose your preferred timeframe
        3. Select patterns to detect
        4. Click "Analyze Patterns"
        
        **Features:**
        - 📊 Clean, readable candlestick charts
        - 🎯 Clear pattern markers with confidence levels
        - ⚡ Fast, responsive interface
        - 📈 Professional trading visualization
        """)
        
        st.info("👈 Configure your analysis in the sidebar to begin")
    
    # Show quick preview of a random instrument
    show_quick_preview()


def show_quick_preview():
    """Show a quick preview chart of sample data"""
    try:
        st.markdown("### 📊 Quick Preview")
        st.markdown("*Sample chart showing candlestick visualization*")
        
        # Try to load sample data for preview
        instruments = st.session_state.data_loader.get_available_instruments()
        if instruments:
            # Use first available instrument for preview
            preview_instrument = instruments[0]
            data = st.session_state.data_loader.load_instrument_data(preview_instrument)
            
            if data is not None and not data.empty:
                # Get last 50 candles for preview
                preview_data = data.tail(50)
                
                # Create compact preview chart
                preview_chart = st.session_state.chart_renderer.create_compact_chart(
                    data=preview_data,
                    patterns=None,  # No patterns for preview
                    title=f"{preview_instrument} - Sample Chart",
                    height=300
                )
                
                st.plotly_chart(preview_chart, use_container_width=True, key="preview_chart")
                st.caption(f"Sample data from {preview_instrument} (last 50 candles)")
            else:
                st.info("Preview chart will appear here once data is available")
        else:
            st.info("No sample data available for preview")
            
    except Exception as e:
        # Don't show errors for preview - just skip it
        st.info("Preview chart will appear here once you run analysis")


def run_analysis(config: Dict[str, Any]):
    """Run pattern analysis with improved UI"""
    instrument = config['instrument']
    timeframe = config['timeframe']
    patterns = config['patterns']
    
    # Check if any patterns are selected
    selected_pattern_names = [name for name, selected in patterns.items() if selected]
    if not selected_pattern_names:
        st.warning("⚠️ Please select at least one pattern to detect")
        return
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        st.markdown(f"### 📊 Analyzing {instrument} - {timeframe}")
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Load data
        status_text.info("📥 Loading data...")
        progress_bar.progress(20)
        
        data = st.session_state.data_loader.load_instrument_data(instrument)
        if data is None or data.empty:
            st.error(f"❌ No data available for {instrument}")
            return
        
        # Aggregate data
        status_text.info("🔄 Processing timeframe...")
        progress_bar.progress(40)
        
        agg_data = st.session_state.data_aggregator.aggregate_data(data, timeframe)
        if agg_data is None or agg_data.empty:
            st.error(f"❌ Failed to process {timeframe} data")
            return
        
        # Detect patterns
        status_text.info("🔍 Detecting patterns...")
        progress_bar.progress(60)
        
        all_patterns = []
        for pattern_name in selected_pattern_names:
            detector = PATTERN_DETECTORS[pattern_name]
            try:
                found_patterns = detector.detect(agg_data, timeframe)
                all_patterns.extend(found_patterns)
            except Exception as e:
                st.warning(f"⚠️ Error detecting {pattern_name}: {e}")
        
        # Create visualization
        status_text.info("📈 Creating chart...")
        progress_bar.progress(80)
        
        # Clear progress
        progress_bar.progress(100)
        time.sleep(0.5)
        progress_container.empty()
        
        # Display results
        display_results(instrument, timeframe, agg_data, all_patterns, config)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        logger.error(f"Analysis error: {e}", exc_info=True)


def display_results(instrument: str, timeframe: str, data: pd.DataFrame, 
                   patterns: List[PatternResult], config: Dict[str, Any]):
    """Display analysis results with clean layout"""
    
    # Summary metrics
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
        if patterns:
            high_conf_patterns = [p for p in patterns if p.confidence >= 0.7]
            st.metric("🎯 High Confidence", len(high_conf_patterns))
        else:
            st.metric("🎯 High Confidence", "0")
    
    # Main chart
    st.markdown("### 📈 Price Chart with Pattern Indicators")
    
    try:
        chart = st.session_state.chart_renderer.create_simple_chart(
            data=data,
            patterns=patterns,
            title=f"{instrument} - {timeframe}",
            height=config['chart_height'],
            max_candles=config['max_candles']
        )
        st.plotly_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {e}")
    
    # Pattern details
    if patterns:
        display_pattern_details(patterns, data, config)
    else:
        st.info("ℹ️ No patterns detected in the selected timeframe")


def display_pattern_details(patterns: List[PatternResult], data: pd.DataFrame, config: Dict[str, Any]):
    """Display pattern details in a clean format"""
    
    st.markdown("### 📋 Pattern Details")
    
    # Sort patterns by confidence
    sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
    
    # Display top patterns
    for i, pattern in enumerate(sorted_patterns[:10]):  # Show top 10
        with st.expander(
            f"#{i+1} {pattern.pattern_type.replace('_', ' ').title()} - {pattern.confidence:.1%} confidence",
            expanded=i < 3  # Expand first 3
        ):
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown(f"""
                **Pattern:** {pattern.pattern_type.replace('_', ' ').title()}  
                **Confidence:** {pattern.confidence:.1%}  
                **Date:** {pattern.datetime.strftime('%Y-%m-%d')}  
                **Time:** {pattern.datetime.strftime('%H:%M')}
                """)
                
                # Pattern strength indicator
                if pattern.confidence >= 0.8:
                    st.success("🟢 High Confidence")
                elif pattern.confidence >= 0.6:
                    st.warning("🟡 Medium Confidence") 
                else:
                    st.info("🔵 Low Confidence")
            
            with col2:
                # Create focused chart for this pattern
                try:
                    detail_chart = st.session_state.chart_renderer.create_pattern_detail_chart(
                        data=data,
                        pattern=pattern,
                        context_candles=15
                    )
                    st.plotly_chart(detail_chart, use_container_width=True, key=f"pattern_{i}")
                except Exception as e:
                    st.error(f"Error creating detail chart: {e}")
    
    # Pattern summary table
    if len(patterns) > 10:
        st.markdown("### 📊 All Patterns Summary")
        
        # Create summary DataFrame
        pattern_data = []
        for pattern in sorted_patterns:
            pattern_data.append({
                'Pattern': pattern.pattern_type.replace('_', ' ').title(),
                'Confidence': f"{pattern.confidence:.1%}",
                'Date': pattern.datetime.strftime('%Y-%m-%d'),
                'Time': pattern.datetime.strftime('%H:%M')
            })
        
        pattern_df = pd.DataFrame(pattern_data)
        st.dataframe(pattern_df, use_container_width=True)


if __name__ == "__main__":
    main()
