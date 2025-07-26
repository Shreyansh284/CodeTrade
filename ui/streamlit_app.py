#!/usr/bin/env python3
"""
Stock Pattern Detector - Main Streamlit Application

Professional trading interface for detecting and visualizing candlestick patterns
in stock market data from local CSV files with advanced navigation and theming.
"""

import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime
from typing import List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    """Main application entry point with enhanced error handling"""
    try:
        st.title("📈 Stock Pattern Detector")
        st.markdown("Professional trading interface for candlestick pattern analysis")
        
        # Initialize session state
        initialize_session_state()
        
        # Create sidebar controls
        config = create_sidebar()
        if not config:
            st.error("❌ Failed to initialize application configuration")
            return
        
        # Main content area
        display_main_content(config)
        
        # Footer with additional information
        with st.expander("ℹ️ About this Application", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Features:**
                - Multi-timeframe analysis
                - Real-time pattern detection
                - Interactive charts with zoom
                - Detailed pattern analysis
                - CSV export functionality
                """)
            
            with col2:
                st.markdown("""
                **Supported Patterns:**
                - Dragonfly Doji
                - Hammer
                - Rising Window
                - Evening Star
                - Three White Soldiers
                """)
            
            st.markdown("---")
            st.markdown("💡 **Tip:** Use 'All Timeframes' to get a comprehensive view across different time periods!")
            
    except Exception as e:
        st.error(f"❌ Application error: {e}")
        st.error("Please refresh the page and try again.")
        logger.error(f"Main application error: {e}", exc_info=True)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = CSVLoader()
    if 'data_aggregator' not in st.session_state:
        st.session_state.data_aggregator = DataAggregator()
    if 'selected_pattern' not in st.session_state:
        st.session_state.selected_pattern = None
    if 'show_pattern_details' not in st.session_state:
        st.session_state.show_pattern_details = False
    if 'fast_mode' not in st.session_state:
        st.session_state.fast_mode = True
    if 'show_zoom_view' not in st.session_state:
        st.session_state.show_zoom_view = False
    if 'zoom_pattern_index' not in st.session_state:
        st.session_state.zoom_pattern_index = None
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1


def create_sidebar():
    """Create sidebar with instrument selection and analysis controls"""
    st.sidebar.header("📊 Analysis Configuration")
    
    # Get available instruments
    try:
        available_instruments = st.session_state.data_loader.get_available_instruments()
        if not available_instruments:
            st.sidebar.error("❌ No instruments found in 5Scripts directory")
            return None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading instruments: {e}")
        return None
    
    # Instrument selection
    st.sidebar.subheader("🏢 Select Instrument")
    selected_instrument = st.sidebar.selectbox(
        "Choose an instrument:",
        available_instruments,
        help="Select the stock instrument to analyze"
    )
    
    # Timeframe selection
    st.sidebar.subheader("⏰ Time Period")
    timeframes = st.session_state.data_aggregator.get_supported_timeframes()
    
    # Add "All Timeframes" option
    timeframe_options = ["All Timeframes"] + timeframes
    
    selected_timeframe_index = st.sidebar.selectbox(
        "Select timeframe:",
        range(len(timeframe_options)),
        format_func=lambda x: timeframe_options[x],
        index=2,  # Default to 5min (index 2 because "All Timeframes" is at 0)
        help="Choose the time period for analysis or select 'All Timeframes' to analyze all available timeframes"
    )
    
    selected_timeframe = timeframe_options[selected_timeframe_index]
    
    # Pattern selection
    st.sidebar.subheader("🔍 Pattern Selection")
    st.sidebar.write("Select patterns to detect:")
    
    selected_patterns = {}
    for pattern_name in PATTERN_DETECTORS.keys():
        selected_patterns[pattern_name] = st.sidebar.checkbox(
            pattern_name,
            value=True,
            help=f"Detect {pattern_name} patterns"
        )
    
    # Analysis button
    st.sidebar.subheader("🚀 Analysis")
    
    # Quick settings
    col1, col2 = st.sidebar.columns(2)
    with col1:
        fast_mode = st.checkbox(
            "🚀 Fast Mode",
            value=st.session_state.get('fast_mode', True),
            help="Faster charts"
        )
    with col2:
        show_details = st.checkbox(
            "📊 Details",
            value=False,
            help="Show detailed info"
        )
    
    st.session_state.fast_mode = fast_mode
    
    # Single analysis button
    analyze_button = st.sidebar.button(
        "🔍 Start Analysis", 
        type="primary", 
        use_container_width=True,
        help="Begin pattern detection analysis"
    )
    
    if analyze_button:
        return {
            'instrument': selected_instrument,
            'timeframe': selected_timeframe,
            'patterns': selected_patterns,
            'trigger_analysis': True,
            'show_details': show_details
        }
    
    return {
        'instrument': selected_instrument,
        'timeframe': selected_timeframe,
        'patterns': selected_patterns,
        'trigger_analysis': False,
        'show_details': show_details
    }


def display_main_content(config: Dict[str, Any]):
    """Display main content area with analysis results"""
    if config['trigger_analysis']:
        if config['timeframe'] == "All Timeframes":
            run_all_timeframes_analysis(config['instrument'], config['patterns'])
        else:
            run_analysis(config['instrument'], config['timeframe'], config['patterns'])
    else:
        st.info("👈 Configure your analysis settings in the sidebar and click **Start Analysis** to begin.")


def run_all_timeframes_analysis(instrument: str, patterns: Dict[str, bool]):
    """Execute pattern analysis for all timeframes"""
    timeframes = st.session_state.data_aggregator.get_supported_timeframes()
    
    st.header(f"📊 Multi-Timeframe Analysis: {instrument}")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data once
    status_text.text("📥 Loading instrument data...")
    progress_bar.progress(10)
    
    data = st.session_state.data_loader.load_instrument_data(instrument)
    if data is None or data.empty:
        st.error(f"❌ No data available for {instrument}")
        return
    
    st.success(f"✅ Loaded {len(data):,} records for {instrument}")
    
    # Create tabs for each timeframe
    tabs = st.tabs([f"📈 {tf}" for tf in timeframes])
    
    for i, (tab, timeframe) in enumerate(zip(tabs, timeframes)):
        with tab:
            progress = 10 + (80 * (i + 1) / len(timeframes))
            progress_bar.progress(int(progress))
            status_text.text(f"🔍 Processing {timeframe}...")
            
            # Run analysis for this timeframe
            result = run_single_timeframe_analysis(instrument, timeframe, patterns, data)
            
            if result:
                agg_data, detected_patterns = result
                display_timeframe_results(instrument, timeframe, agg_data, detected_patterns)
            else:
                st.error(f"❌ Failed to analyze {timeframe}")
    
    # Clear progress indicators
    progress_bar.progress(100)
    status_text.text("✅ Multi-timeframe analysis complete!")
    
    # Clear progress after a moment
    import time
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()


def run_single_timeframe_analysis(instrument: str, timeframe: str, patterns: Dict[str, bool], data: pd.DataFrame):
    """Run analysis for a single timeframe"""
    try:
        # Aggregate data
        agg_data = st.session_state.data_aggregator.aggregate_data(data, timeframe)
        if agg_data is None or agg_data.empty:
            st.error(f"❌ Failed to aggregate data to {timeframe}")
            return None
        
        # Detect patterns
        all_patterns = []
        selected_pattern_names = [name for name, selected in patterns.items() if selected]
        
        for pattern_name in selected_pattern_names:
            detector = PATTERN_DETECTORS[pattern_name]
            try:
                patterns_found = detector.detect(agg_data, timeframe)
                all_patterns.extend(patterns_found)
            except Exception as e:
                st.warning(f"⚠️ Error detecting {pattern_name} in {timeframe}: {e}")
        
        return agg_data, all_patterns
        
    except Exception as e:
        st.error(f"❌ Analysis failed for {timeframe}: {e}")
        return None


def display_timeframe_results(instrument: str, timeframe: str, data: pd.DataFrame, patterns: List[PatternResult]):
    """Display results for a specific timeframe"""
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📊 Data Points", f"{len(data):,}")
    with col2:
        st.metric("🔍 Patterns", len(patterns))
    with col3:
        if patterns:
            avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
            st.metric("📈 Avg Confidence", f"{avg_confidence:.1%}")
        else:
            st.metric("📈 Avg Confidence", "N/A")
    
    # Chart
    if not data.empty:
        try:
            if st.session_state.fast_mode:
                chart = create_fast_candlestick_chart(data, patterns, instrument, timeframe)
            else:
                chart = create_themed_candlestick_chart(
                    data, patterns, instrument, timeframe, 
                    chart_type="timeframe_specific"
                )
            st.plotly_chart(chart, use_container_width=True, key=f"chart_{instrument}_{timeframe}")
        except Exception as e:
            st.error(f"❌ Error creating chart for {timeframe}: {e}")
    
    # Pattern details
    if patterns:
        with st.expander(f"📋 Pattern Details ({len(patterns)} found)", expanded=False):
            for pattern in patterns[:10]:  # Show first 10 patterns
                col1, col2, col3 = st.columns([2, 1, 1])
                with col1:
                    st.write(f"**{pattern.pattern_type.replace('_', ' ').title()}**")
                with col2:
                    st.write(f"{pattern.confidence:.1%}")
                with col3:
                    st.write(pattern.datetime.strftime('%Y-%m-%d %H:%M'))
            
            if len(patterns) > 10:
                st.write(f"... and {len(patterns) - 10} more patterns")
    else:
        st.info("🔍 No patterns detected in this timeframe")


def run_analysis(instrument: str, timeframe: str, patterns: Dict[str, bool]):
    """Execute pattern analysis workflow"""
    # Progress indicators
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Load data
        status_text.text("📥 Loading instrument data...")
        progress_bar.progress(20)
        
        data = st.session_state.data_loader.load_instrument_data(instrument)
        if data is None or data.empty:
            st.error(f"❌ No data available for {instrument}")
            return
        
        st.success(f"✅ Loaded {len(data):,} records for {instrument}")
        
        # Step 2: Aggregate data
        status_text.text(f"⚙️ Aggregating to {timeframe}...")
        progress_bar.progress(40)
        
        agg_data = st.session_state.data_aggregator.aggregate_data(data, timeframe)
        if agg_data is None or agg_data.empty:
            st.error(f"❌ Failed to aggregate data to {timeframe}")
            return
        
        st.success(f"✅ Aggregated to {len(agg_data):,} {timeframe} candles")
        
        # Step 3: Detect patterns
        status_text.text("🔍 Detecting patterns...")
        progress_bar.progress(60)
        
        all_patterns = []
        selected_pattern_names = [name for name, selected in patterns.items() if selected]
        
        for i, pattern_name in enumerate(selected_pattern_names):
            detector = PATTERN_DETECTORS[pattern_name]
            try:
                patterns_found = detector.detect(agg_data, timeframe)
                all_patterns.extend(patterns_found)
                
                progress = 60 + (30 * (i + 1) / len(selected_pattern_names))
                progress_bar.progress(int(progress))
                status_text.text(f"🔍 Found {len(patterns_found)} {pattern_name} patterns")
                
            except Exception as e:
                st.warning(f"⚠️ Error detecting {pattern_name}: {e}")
        
        # Step 4: Display results
        status_text.text("✅ Analysis complete!")
        progress_bar.progress(100)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        display_results(instrument, timeframe, agg_data, all_patterns)
        
    except Exception as e:
        st.error(f"❌ Analysis failed: {e}")
        progress_bar.empty()
        status_text.empty()


def display_results(instrument: str, timeframe: str, data: pd.DataFrame, patterns: List[PatternResult]):
    """Display comprehensive analysis results with professional UI"""
    
    # Summary metrics
    st.markdown("### 📊 Analysis Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Data Points", f"{len(data):,}", help="Total candlesticks analyzed")
    with col2:
        st.metric("🔍 Patterns Found", len(patterns), help="Total patterns detected")
    with col3:
        if patterns:
            avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
            st.metric("📈 Avg Confidence", f"{avg_confidence:.1%}", help="Average pattern confidence")
        else:
            st.metric("📈 Avg Confidence", "N/A")
    with col4:
        st.metric("⏰ Timeframe", timeframe, help="Chart timeframe")
    
    # Main Chart Section
    st.markdown("---")
    col1, col2 = st.columns([4, 1])
    
    with col1:
        st.markdown("### 📈 Main Chart")
    with col2:
        if st.session_state.fast_mode:
            st.success("⚡ Fast Mode")
        else:
            st.info("🎨 Full Mode")
    
    if not data.empty:
        try:
            if st.session_state.fast_mode:
                main_chart = create_fast_candlestick_chart(data, patterns, instrument, timeframe)
            else:
                main_chart = create_themed_candlestick_chart(
                    data, patterns, instrument, timeframe, 
                    chart_type="main", selected_pattern=st.session_state.selected_pattern
                )
            st.plotly_chart(main_chart, use_container_width=True, key="main_chart")
        except Exception as e:
            st.error(f"❌ Error creating main chart: {e}")
    
    # Pattern Analysis Section
    if patterns:
        st.markdown("---")
        st.markdown("### 🔍 Pattern Analysis")
        
        # Pattern controls
        pattern_types = list(set(p.pattern_type.replace('_', ' ').title() for p in patterns))
        
        selected_pattern_type = st.selectbox(
            "Select Pattern Type to View:",
            ["All Patterns"] + pattern_types,
            help="Choose a specific pattern type to focus on"
        )
        
        # Reset pagination when pattern type changes
        if 'last_selected_pattern_type' not in st.session_state:
            st.session_state.last_selected_pattern_type = selected_pattern_type
        elif st.session_state.last_selected_pattern_type != selected_pattern_type:
            st.session_state.current_page = 1  # Reset to first page
            st.session_state.last_selected_pattern_type = selected_pattern_type
        
        # Filter patterns
        if selected_pattern_type == "All Patterns":
            filtered_patterns = patterns
        else:
            filtered_patterns = [p for p in patterns if p.pattern_type.replace('_', ' ').title() == selected_pattern_type]
        
        # Pattern Navigation Table with Pagination
        display_pattern_navigation_table(filtered_patterns, data, instrument, timeframe)
        
        # Focused Pattern View
        if st.session_state.selected_pattern and st.session_state.show_pattern_details:
            display_focused_pattern_view(data, instrument, timeframe)
        
        # Export and Summary
        st.markdown("---")
        display_export_and_summary(patterns, instrument, timeframe)
    
    else:
        st.info("🔍 No patterns detected in the selected timeframe and criteria.")


def display_individual_pattern_charts(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str):
    """Display simplified individual charts for better performance"""
    st.markdown("#### 📊 Pattern Charts by Type")
    
    # Group patterns by type
    pattern_groups = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.replace('_', ' ').title()
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append(pattern)
    
    # Show each pattern type in a clean expander
    for pattern_type, type_patterns in pattern_groups.items():
        with st.expander(f"📈 {pattern_type} ({len(type_patterns)} patterns)", expanded=False):
            # Show top 3 patterns for this type
            top_patterns = sorted(type_patterns, key=lambda p: p.confidence, reverse=True)[:3]
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Best Confidence:** {top_patterns[0].confidence:.1%}")
                st.write(f"**Total Found:** {len(type_patterns)}")
            with col2:
                if len(top_patterns) > 0:
                    avg_conf = sum(p.confidence for p in type_patterns) / len(type_patterns)
                    st.write(f"**Average:** {avg_conf:.1%}")
                    st.write(f"**Date Range:** {top_patterns[-1].datetime.date()} to {top_patterns[0].datetime.date()}")
            
            # Simple chart for this pattern type
            if st.session_state.fast_mode:
                # Fast mode - just show a simple line chart of confidence over time
                dates = [p.datetime for p in type_patterns]
                confidences = [p.confidence for p in type_patterns]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=confidences,
                    mode='markers+lines',
                    name=pattern_type,
                    marker=dict(size=8, color='blue'),
                    line=dict(width=2)
                ))
                
                fig.update_layout(
                    height=250,
                    title=f"{pattern_type} Confidence Over Time",
                    showlegend=False,
                    margin=dict(l=40, r=40, t=60, b=40)
                )
                
                st.plotly_chart(fig, use_container_width=True, key=f"fast_chart_{pattern_type}")
            else:
                # Full chart with candlesticks - but limit to recent data for performance
                recent_data = data.tail(200) if len(data) > 200 else data
                chart = create_themed_candlestick_chart(
                    recent_data, top_patterns, instrument, timeframe, 
                    chart_type="pattern_specific", pattern_type=pattern_type
                )
                st.plotly_chart(chart, use_container_width=True, key=f"chart_{pattern_type}")
            
            # Show top patterns in a simple table
            st.write("**Top Patterns:**")
            for i, pattern in enumerate(top_patterns):
                st.write(f"{i+1}. {pattern.datetime.strftime('%m/%d %H:%M')} - {pattern.confidence:.1%}")
            
            if len(type_patterns) > 3:
                st.write(f"... and {len(type_patterns) - 3} more")


def display_enhanced_pattern_type_analysis(
    data: pd.DataFrame, 
    patterns: List[PatternResult], 
    instrument: str, 
    timeframe: str, 
    pattern_type: str,
    chart_style: str,
    show_statistics: bool
):
    """Display enhanced analysis for a specific pattern type"""
    
    if show_statistics:
        # Pattern type statistics
        st.markdown(f"**📈 {pattern_type} Analysis Summary**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patterns", len(patterns))
        with col2:
            avg_conf = sum(p.confidence for p in patterns) / len(patterns)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            max_conf = max(p.confidence for p in patterns)
            st.metric("Max Confidence", f"{max_conf:.1%}")
        with col4:
            min_conf = min(p.confidence for p in patterns)
            st.metric("Min Confidence", f"{min_conf:.1%}")
    
    # Chart creation based on style
    if chart_style == "Fast":
        # Fast, simplified chart
        chart = create_fast_candlestick_chart(data, patterns, instrument, timeframe)
        st.plotly_chart(chart, use_container_width=True, key=f"fast_chart_{pattern_type}")
    
    elif chart_style == "Detailed":
        # Full detailed chart for this pattern type
        chart = create_detailed_pattern_chart(data, patterns, instrument, timeframe, pattern_type)
        st.plotly_chart(chart, use_container_width=True, key=f"detailed_chart_{pattern_type}")
    
    elif chart_style == "Compact":
        # Multiple smaller charts showing individual patterns
        display_compact_pattern_charts(data, patterns, instrument, timeframe, pattern_type)
    
    elif chart_style == "Overlay":
        # Single chart with all patterns overlaid
        chart = create_overlay_pattern_chart(data, patterns, instrument, timeframe, pattern_type)
        st.plotly_chart(chart, use_container_width=True, key=f"overlay_chart_{pattern_type}")
    
    if show_statistics:
        # Detailed pattern list
        st.markdown("**📋 Pattern Details:**")
        
        # Create sortable table
        pattern_df = pd.DataFrame([
            {
                'DateTime': p.datetime.strftime('%Y-%m-%d %H:%M'),
                'Confidence': f"{p.confidence:.1%}",
                'Price': f"${data.iloc[p.candle_index]['close']:.2f}" if p.candle_index < len(data) else "N/A",
                'Volume': f"{data.iloc[p.candle_index]['volume']:,.0f}" if p.candle_index < len(data) and 'volume' in data.columns else "N/A"
            }
            for p in patterns
        ])
        
        st.dataframe(
            pattern_df,
            use_container_width=True,
            hide_index=True
        )


def display_fast_pattern_analysis(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, pattern_type: str, show_statistics: bool):
    """Display fast, simplified pattern analysis for better performance"""
    
    if show_statistics:
        # Simple statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patterns", len(patterns))
        with col2:
            avg_conf = sum(p.confidence for p in patterns) / len(patterns)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            max_conf = max(p.confidence for p in patterns)
            st.metric("Best", f"{max_conf:.1%}")
    
    # Fast chart
    chart = create_fast_candlestick_chart(data, patterns, instrument, timeframe)
    st.plotly_chart(chart, use_container_width=True, key=f"fast_{pattern_type}")
    
    if show_statistics:
        # Simple pattern table
        pattern_data = []
        for p in patterns:
            if p.candle_index < len(data):
                candle = data.iloc[p.candle_index]
                pattern_data.append({
                    'Time': p.datetime.strftime('%m-%d %H:%M'),
                    'Confidence': f"{p.confidence:.1%}",
                    'Price': f"${candle['close']:.2f}"
                })
        
        if pattern_data:
            df = pd.DataFrame(pattern_data)
            st.dataframe(df, use_container_width=True, hide_index=True)


def create_detailed_pattern_chart(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, pattern_type: str):
    """Create detailed chart for specific pattern type"""
    theme = get_chart_theme()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f"{instrument} - {pattern_type} Patterns ({timeframe})", 'Volume'),
        row_heights=[0.75, 0.25]
    )
    
    # Add candlestick
    candlestick = go.Candlestick(
        x=data['datetime'] if 'datetime' in data.columns else data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="OHLC",
        increasing_line_color=theme['bullish_color'],
        decreasing_line_color=theme['bearish_color'],
        increasing_fillcolor=theme['bullish_fill'],
        decreasing_fillcolor=theme['bearish_fill'],
        showlegend=False
    )
    fig.add_trace(candlestick, row=1, col=1)
    
    # Add volume
    if 'volume' in data.columns:
        volume_colors = [
            theme['bullish_volume'] if close >= open else theme['bearish_volume']
            for close, open in zip(data['close'], data['open'])
        ]
        
        volume_bars = go.Bar(
            x=data['datetime'] if 'datetime' in data.columns else data.index,
            y=data['volume'],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        )
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Add pattern markers with confidence-based sizing
    for i, pattern in enumerate(patterns):
        if pattern.candle_index < len(data):
            candle = data.iloc[pattern.candle_index]
            pattern_x = candle['datetime'] if 'datetime' in data.columns else pattern.candle_index
            
            # Size marker based on confidence
            marker_size = 8 + (pattern.confidence * 12)  # 8-20 size range
            
            fig.add_trace(go.Scatter(
                x=[pattern_x],
                y=[candle['high'] * 1.02],
                mode='markers+text',
                name=f"{i+1}. {pattern.confidence:.1%}",
                marker=dict(
                    symbol='star',
                    size=marker_size,
                    color=theme['highlight_color'],
                    line=dict(width=2, color='white')
                ),
                text=[f"{i+1}"],
                textposition="middle center",
                textfont=dict(size=8, color='white'),
                hovertemplate=f"<b>Pattern #{i+1}</b><br>" +
                             f"Confidence: {pattern.confidence:.1%}<br>" +
                             f"Time: {pattern.datetime}<br>" +
                             f"Price: ${candle['close']:.2f}<extra></extra>",
                showlegend=True
            ), row=1, col=1)
    
    # Enhanced styling
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=150, t=80, b=50),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_color']
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=theme['grid_color'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=theme['grid_color'])
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def display_compact_pattern_charts(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, pattern_type: str):
    """Display multiple compact charts for individual patterns"""
    st.markdown(f"**Individual {pattern_type} Pattern Views:**")
    
    # Show up to 6 patterns in a grid
    patterns_to_show = patterns[:6]
    
    # Create grid layout
    cols_per_row = 2
    rows_needed = (len(patterns_to_show) + cols_per_row - 1) // cols_per_row
    
    for row in range(rows_needed):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            pattern_idx = row * cols_per_row + col_idx
            
            if pattern_idx < len(patterns_to_show):
                pattern = patterns_to_show[pattern_idx]
                
                with cols[col_idx]:
                    # Create mini focused chart
                    mini_chart = create_mini_pattern_chart(data, pattern, f"{pattern_idx + 1}")
                    st.plotly_chart(mini_chart, use_container_width=True, key=f"mini_{pattern_type}_{pattern_idx}")
                    
                    # Pattern info
                    st.write(f"**#{pattern_idx + 1}:** {pattern.confidence:.1%} confidence")
                    st.write(f"📅 {pattern.datetime.strftime('%m/%d %H:%M')}")


def create_mini_pattern_chart(data: pd.DataFrame, pattern: PatternResult, label: str):
    """Create a mini chart focused on a single pattern"""
    theme = get_chart_theme()
    
    # Focus window
    window_size = 20
    pattern_index = pattern.candle_index
    start_idx = max(0, pattern_index - window_size // 2)
    end_idx = min(len(data), pattern_index + window_size // 2)
    
    focused_data = data.iloc[start_idx:end_idx].copy()
    
    fig = go.Figure()
    
    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=list(range(len(focused_data))),
        open=focused_data['open'],
        high=focused_data['high'],
        low=focused_data['low'],
        close=focused_data['close'],
        increasing_line_color=theme['bullish_color'],
        decreasing_line_color=theme['bearish_color'],
        showlegend=False
    ))
    
    # Highlight pattern
    adjusted_idx = pattern_index - start_idx
    if 0 <= adjusted_idx < len(focused_data):
        pattern_candle = focused_data.iloc[adjusted_idx]
        
        fig.add_trace(go.Scatter(
            x=[adjusted_idx],
            y=[pattern_candle['high'] * 1.03],
            mode='markers+text',
            marker=dict(symbol='star', size=15, color='gold'),
            text=[label],
            textposition="top center",
            showlegend=False
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=5, r=5, t=30, b=5),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        title=dict(text=f"Pattern {label}", x=0.5, font=dict(size=12))
    )
    
    return fig


def create_overlay_pattern_chart(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, pattern_type: str):
    """Create chart with all patterns overlaid with different colors"""
    theme = get_chart_theme()
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(f"{instrument} - All {pattern_type} Patterns ({timeframe})", 'Volume'),
        row_heights=[0.8, 0.2]
    )
    
    # Add candlestick
    candlestick = go.Candlestick(
        x=data['datetime'] if 'datetime' in data.columns else data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="OHLC",
        increasing_line_color=theme['bullish_color'],
        decreasing_line_color=theme['bearish_color'],
        showlegend=False
    )
    fig.add_trace(candlestick, row=1, col=1)
    
    # Add volume
    if 'volume' in data.columns:
        volume_colors = [
            theme['bullish_volume'] if close >= open else theme['bearish_volume']
            for close, open in zip(data['close'], data['open'])
        ]
        
        volume_bars = go.Bar(
            x=data['datetime'] if 'datetime' in data.columns else data.index,
            y=data['volume'],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        )
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Color palette for different confidence levels
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Group patterns by confidence ranges for color coding
    confidence_ranges = [
        (0.9, 1.0, "🔥 Very High (90%+)"),
        (0.8, 0.9, "📈 High (80-90%)"),
        (0.7, 0.8, "📊 Good (70-80%)"),
        (0.6, 0.7, "⚡ Medium (60-70%)"),
        (0.5, 0.6, "📉 Low (50-60%)"),
        (0.0, 0.5, "🔍 Very Low (<50%)")
    ]
    
    for i, (min_conf, max_conf, label) in enumerate(confidence_ranges):
        range_patterns = [p for p in patterns if min_conf <= p.confidence < max_conf]
        
        if range_patterns:
            pattern_x = []
            pattern_y = []
            hover_text = []
            
            for pattern in range_patterns:
                if pattern.candle_index < len(data):
                    candle = data.iloc[pattern.candle_index]
                    pattern_x.append(candle['datetime'] if 'datetime' in data.columns else pattern.candle_index)
                    pattern_y.append(candle['high'] * 1.02)
                    hover_text.append(
                        f"Confidence: {pattern.confidence:.1%}<br>" +
                        f"Time: {pattern.datetime}<br>" +
                        f"Price: ${candle['close']:.2f}"
                    )
            
            if pattern_x:
                fig.add_trace(go.Scatter(
                    x=pattern_x,
                    y=pattern_y,
                    mode='markers',
                    name=label,
                    marker=dict(
                        symbol='star',
                        size=10,
                        color=colors[i % len(colors)],
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate="%{hovertext}<extra></extra>",
                    hovertext=hover_text,
                    showlegend=True
                ), row=1, col=1)
    
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_color']
    )
    
    return fig


def display_pattern_navigation_table(patterns: List[PatternResult], data: pd.DataFrame, instrument: str, timeframe: str):
    """Display paginated pattern navigation"""
    st.markdown("#### 🎯 Pattern Details")
    
    if not patterns:
        st.info("No patterns found")
        return
    
    # Sort patterns by confidence (highest first)
    sorted_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)
    
    # Pagination setup
    patterns_per_page = 10
    total_patterns = len(sorted_patterns)
    total_pages = (total_patterns + patterns_per_page - 1) // patterns_per_page
    
    # Initialize page state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 1
    
    # Pagination controls at the top
    if total_pages > 1:
        col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous", disabled=st.session_state.current_page <= 1):
                st.session_state.current_page = max(1, st.session_state.current_page - 1)
                st.rerun()
        
        with col2:
            if st.button("➡️ Next", disabled=st.session_state.current_page >= total_pages):
                st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
                st.rerun()
        
        with col3:
            st.write(f"**Page {st.session_state.current_page} of {total_pages}** • Showing {patterns_per_page} of {total_patterns} patterns")
        
        with col4:
            # Quick page jump
            selected_page = st.selectbox(
                "Jump to page:",
                range(1, total_pages + 1),
                index=st.session_state.current_page - 1,
                key="page_selector"
            )
            if selected_page != st.session_state.current_page:
                st.session_state.current_page = selected_page
                st.rerun()
        
        with col5:
            if st.button("🔄 Reset", help="Go to first page"):
                st.session_state.current_page = 1
                st.rerun()
    
    # Calculate patterns for current page
    start_idx = (st.session_state.current_page - 1) * patterns_per_page
    end_idx = min(start_idx + patterns_per_page, total_patterns)
    current_page_patterns = sorted_patterns[start_idx:end_idx]
    
    # Display patterns for current page
    for i, pattern in enumerate(current_page_patterns):
        pattern_number = start_idx + i + 1  # Global pattern number
        
        with st.expander(
            f"#{pattern_number} {pattern.pattern_type.replace('_', ' ').title()} - {pattern.confidence:.1%} confidence",
            expanded=False
        ):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Time:** {pattern.datetime.strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Confidence:** {pattern.confidence:.2%}")
                st.write(f"**Timeframe:** {pattern.timeframe}")
                
                if pattern.candle_index < len(data):
                    candle = data.iloc[pattern.candle_index]
                    st.write(f"**Price:** ${candle['close']:.2f}")
            
            with col2:
                # Simple zoom button that works
                if st.button(f"🔍 View Pattern #{pattern_number}", key=f"view_{pattern_number}_{pattern.candle_index}"):
                    st.session_state.show_zoom_view = True
                    st.session_state.zoom_pattern_index = start_idx + i  # Adjust for pagination
                    st.rerun()
                
                # Pattern details inline
                if pattern.candle_index < len(data):
                    candle = data.iloc[pattern.candle_index]
                    price_change = candle['close'] - candle['open']
                    price_change_pct = (price_change / candle['open']) * 100
                    st.write(f"**Change:** {price_change_pct:+.2f}%")
                    st.write(f"**Range:** ${candle['high'] - candle['low']:.2f}")
    
    # Pagination controls at the bottom (if more than 1 page)
    if total_pages > 1:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("⬅️ Prev", disabled=st.session_state.current_page <= 1, key="bottom_prev"):
                st.session_state.current_page = max(1, st.session_state.current_page - 1)
                st.rerun()
        
        with col2:
            st.write(f"**Page {st.session_state.current_page} of {total_pages}**")
        
        with col3:
            if st.button("➡️ Next", disabled=st.session_state.current_page >= total_pages, key="bottom_next"):
                st.session_state.current_page = min(total_pages, st.session_state.current_page + 1)
                st.rerun()
    
    # Show zoom view if requested
    if st.session_state.show_zoom_view and st.session_state.zoom_pattern_index is not None:
        if st.session_state.zoom_pattern_index < len(sorted_patterns):
            display_zoom_view(data, sorted_patterns[st.session_state.zoom_pattern_index], instrument, timeframe)


def display_zoom_view(data: pd.DataFrame, pattern: PatternResult, instrument: str, timeframe: str):
    """Display zoomed view of a pattern"""
    st.markdown("---")
    st.markdown(f"#### 🔍 Zoomed View: {pattern.pattern_type.replace('_', ' ').title()}")
    
    # Close button
    col1, col2 = st.columns([4, 1])
    with col2:
        if st.button("❌ Close Zoom", key="close_zoom"):
            st.session_state.show_zoom_view = False
            st.session_state.zoom_pattern_index = None
            st.rerun()
    
    # Create zoomed chart
    window_size = 30  # Show 30 candles around the pattern
    start_idx = max(0, pattern.candle_index - window_size // 2)
    end_idx = min(len(data), pattern.candle_index + window_size // 2)
    
    zoomed_data = data.iloc[start_idx:end_idx].copy()
    
    if len(zoomed_data) > 0:
        # Create simple chart
        fig = go.Figure()
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=zoomed_data.index,
            open=zoomed_data['open'],
            high=zoomed_data['high'],
            low=zoomed_data['low'],
            close=zoomed_data['close'],
            name=f"{instrument} - {timeframe}",
            increasing_line_color='#00C851',
            decreasing_line_color='#FF4444'
        ))
        
        # Highlight pattern
        pattern_idx_in_zoom = pattern.candle_index - start_idx
        if 0 <= pattern_idx_in_zoom < len(zoomed_data):
            candle = zoomed_data.iloc[pattern_idx_in_zoom]
            fig.add_trace(go.Scatter(
                x=[candle.name],
                y=[candle['high'] * 1.02],
                mode='markers+text',
                marker=dict(symbol='star', size=20, color='gold'),
                text=['📍'],
                textposition="top center",
                name="Pattern",
                showlegend=False
            ))
        
        # Simple layout
        fig.update_layout(
            height=400,
            title=f"Pattern: {pattern.confidence:.1%} confidence at {pattern.datetime}",
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40),
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True, key="zoom_chart")
        
        # Pattern info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{pattern.confidence:.1%}")
        with col2:
            if pattern.candle_index < len(data):
                st.metric("Price", f"${data.iloc[pattern.candle_index]['close']:.2f}")
        with col3:
            st.metric("Time", pattern.datetime.strftime('%H:%M'))


def display_focused_pattern_view(data: pd.DataFrame, instrument: str, timeframe: str):
    """Display focused view of selected pattern"""
    st.markdown("---")
    st.markdown("#### 🎯 Focused Pattern View")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        focused_chart = create_focused_pattern_chart(
            data, st.session_state.selected_pattern, instrument, timeframe
        )
        st.plotly_chart(focused_chart, use_container_width=True, key="focused_chart")
    
    with col2:
        display_pattern_info_panel(st.session_state.selected_pattern, data)
        
        if st.button("🔙 Back to Overview", type="secondary"):
            st.session_state.selected_pattern = None
            st.session_state.show_pattern_details = False
            st.rerun()


def display_pattern_type_chart(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, pattern_type: str):
    """Legacy function - now handled by display_enhanced_pattern_type_analysis"""
    # This function is kept for backward compatibility but redirects to the enhanced version
    display_enhanced_pattern_type_analysis(
        data, patterns, instrument, timeframe, pattern_type, 
        chart_style="Detailed", show_statistics=True
    )


def display_pattern_details(pattern: PatternResult, data: pd.DataFrame):
    """Display detailed pattern information with enhanced analysis"""
    try:
        if pattern.candle_index < len(data):
            candle = data.iloc[pattern.candle_index]
            
            # Create tabs for different detail views
            detail_tabs = st.tabs(["📊 Basic Info", "📈 Technical Analysis", "🕒 Context"])
            
            with detail_tabs[0]:
                # Basic pattern information
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Pattern Information:**")
                    st.write(f"• **Type:** {pattern.pattern_type.replace('_', ' ').title()}")
                    st.write(f"• **Confidence:** {pattern.confidence:.2%}")
                    st.write(f"• **DateTime:** {pattern.datetime}")
                    st.write(f"• **Timeframe:** {pattern.timeframe}")
                    st.write(f"• **Candle Index:** {pattern.candle_index}")
                
                with col2:
                    st.markdown("**Candle Data:**")
                    st.write(f"• **Open:** ${candle['open']:.2f}")
                    st.write(f"• **High:** ${candle['high']:.2f}")
                    st.write(f"• **Low:** ${candle['low']:.2f}")
                    st.write(f"• **Close:** ${candle['close']:.2f}")
                    if 'volume' in candle:
                        st.write(f"• **Volume:** {candle['volume']:,.0f}")
            
            with detail_tabs[1]:
                # Technical analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Price Action:**")
                    price_change = candle['close'] - candle['open']
                    price_change_pct = (price_change / candle['open']) * 100
                    
                    st.write(f"• **Price Change:** ${price_change:.2f} ({price_change_pct:+.2f}%)")
                    st.write(f"• **Range:** ${candle['high'] - candle['low']:.2f}")
                    st.write(f"• **Body Size:** ${abs(candle['close'] - candle['open']):.2f}")
                    
                    # Candle type
                    if candle['close'] > candle['open']:
                        candle_type = "🟢 Bullish"
                    elif candle['close'] < candle['open']:
                        candle_type = "🔴 Bearish"
                    else:
                        candle_type = "⚪ Doji"
                    st.write(f"• **Candle Type:** {candle_type}")
                
                with col2:
                    st.markdown("**Market Context:**")
                    
                    # Calculate position within range
                    if candle['high'] != candle['low']:
                        close_position = (candle['close'] - candle['low']) / (candle['high'] - candle['low'])
                        st.write(f"• **Close Position:** {close_position:.1%} of range")
                    
                    # Upper and lower shadows
                    upper_shadow = candle['high'] - max(candle['open'], candle['close'])
                    lower_shadow = min(candle['open'], candle['close']) - candle['low']
                    st.write(f"• **Upper Shadow:** ${upper_shadow:.2f}")
                    st.write(f"• **Lower Shadow:** ${lower_shadow:.2f}")
            
            with detail_tabs[2]:
                # Context analysis
                st.markdown("**Pattern Context:**")
                
                # Look at surrounding candles for context
                context_start = max(0, pattern.candle_index - 5)
                context_end = min(len(data), pattern.candle_index + 6)
                context_data = data.iloc[context_start:context_end]
                
                if len(context_data) > 1:
                    # Price trend before pattern
                    if pattern.candle_index > 0:
                        prev_close = data.iloc[pattern.candle_index - 1]['close']
                        trend = "📈 Uptrend" if candle['close'] > prev_close else "📉 Downtrend"
                        st.write(f"• **Previous Trend:** {trend}")
                    
                    # Volume context
                    if 'volume' in candle and len(context_data) > 3:
                        avg_volume = context_data['volume'].mean()
                        volume_ratio = candle['volume'] / avg_volume
                        volume_context = "🔊 High" if volume_ratio > 1.5 else "🔉 Normal" if volume_ratio > 0.5 else "🔇 Low"
                        st.write(f"• **Volume Context:** {volume_context} ({volume_ratio:.1f}x avg)")
                    
                    # Create mini context chart
                    st.markdown("**Context Chart:**")
                    mini_chart = create_mini_context_chart(context_data, pattern.candle_index - context_start)
                    st.plotly_chart(mini_chart, use_container_width=True, key=f"mini_chart_{pattern.candle_index}")
                    
    except Exception as e:
        st.error(f"❌ Error displaying pattern details: {e}")


def create_mini_context_chart(context_data: pd.DataFrame, pattern_position: int):
    """Create a mini chart showing pattern context"""
    fig = go.Figure()
    
    # Add candlestick trace
    fig.add_trace(go.Candlestick(
        x=list(range(len(context_data))),
        open=context_data['open'],
        high=context_data['high'],
        low=context_data['low'],
        close=context_data['close'],
        name="Context",
        increasing_line_color='#00C851',
        decreasing_line_color='#FF4444',
        showlegend=False
    ))
    
    # Highlight the pattern candle
    if 0 <= pattern_position < len(context_data):
        fig.add_trace(go.Scatter(
            x=[pattern_position],
            y=[context_data.iloc[pattern_position]['high'] * 1.02],
            mode='markers',
            marker=dict(symbol='star', size=15, color='gold'),
            name="Pattern",
            showlegend=False
        ))
    
    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray')
    )
    
    return fig


def display_pattern_info_panel(pattern: PatternResult, data: pd.DataFrame):
    """Display pattern information panel"""
    st.markdown("**Pattern Information**")
    
    st.write(f"**Type:** {pattern.pattern_type.replace('_', ' ').title()}")
    st.write(f"**Confidence:** {pattern.confidence:.2%}")
    st.write(f"**Time:** {pattern.datetime.strftime('%Y-%m-%d %H:%M')}")
    st.write(f"**Timeframe:** {pattern.timeframe}")
    
    if pattern.candle_index < len(data):
        candle = data.iloc[pattern.candle_index]
        st.markdown("**Candle Data**")
        st.write(f"**OHLC:** {candle['open']:.2f} / {candle['high']:.2f} / {candle['low']:.2f} / {candle['close']:.2f}")
        if 'volume' in candle:
            st.write(f"**Volume:** {candle['volume']:,.0f}")


def display_export_and_summary(patterns: List[PatternResult], instrument: str, timeframe: str):
    """Display enhanced export functionality and pattern summary"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📥 Export Results")
        
        # Export options
        export_format = st.selectbox(
            "Export Format:",
            ["CSV"],
            help="Choose the export format for pattern results"
        )
        
        # Export button with progress indicator
        if st.button("📄 Export Patterns", type="primary", help="Export pattern detection results"):
            try:
                from export.csv_exporter import CSVExporter
                
                # Show progress
                with st.spinner("🔄 Preparing export..."):
                    exporter = CSVExporter()
                    
                    if patterns:
                        export_path = exporter.export_patterns(patterns, instrument, timeframe)
                        
                        if export_path and os.path.exists(export_path):
                            st.success(f"✅ Successfully exported {len(patterns)} patterns!")
                            
                            # Get export summary
                            summary = exporter.get_export_summary(export_path)
                            
                            # Display export info
                            st.info(f"📁 **File:** {os.path.basename(export_path)}")
                            st.info(f"📊 **Size:** {summary.get('file_size_bytes', 0)} bytes")
                            st.info(f"📋 **Rows:** {summary.get('row_count', 0)}")
                            
                            # Download button
                            try:
                                with open(export_path, 'rb') as f:
                                    file_data = f.read()
                                    
                                st.download_button(
                                    label="⬇️ Download CSV File",
                                    data=file_data,
                                    file_name=os.path.basename(export_path),
                                    mime='text/csv',
                                    type="secondary",
                                    help="Click to download the exported CSV file"
                                )
                                
                            except Exception as download_error:
                                st.error(f"❌ Download preparation failed: {download_error}")
                        else:
                            st.error("❌ Export failed - file not created")
                    else:
                        # Export empty results
                        export_path = exporter.export_empty_results(
                            instrument, timeframe, "No patterns detected in current analysis"
                        )
                        
                        if export_path:
                            st.warning("⚠️ No patterns found - exported empty results file")
                            
                            with open(export_path, 'rb') as f:
                                st.download_button(
                                    label="⬇️ Download Empty CSV",
                                    data=f.read(),
                                    file_name=os.path.basename(export_path),
                                    mime='text/csv',
                                    type="secondary"
                                )
                        else:
                            st.error("❌ Failed to create export file")
                            
            except Exception as e:
                st.error(f"❌ Export error: {str(e)}")
                logger.error(f"Export error in UI: {e}")
        
        # Export information
        with st.expander("ℹ️ Export Information", expanded=False):
            st.write("**Export includes:**")
            st.write("• Dataset name and timeframe")
            st.write("• Pattern detection timestamps")
            st.write("• Pattern types and confidence scores")
            st.write("• Candle index positions")
            st.write("")
            st.write("**File format:** CSV with headers")
            st.write("**Filename:** Auto-generated with timestamp")
    
    with col2:
        st.markdown("#### 📊 Pattern Summary")
        
        if patterns:
            # Pattern type breakdown
            pattern_counts = {}
            confidence_sum = {}
            
            for pattern in patterns:
                pattern_type = pattern.pattern_type.replace('_', ' ').title()
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
                confidence_sum[pattern_type] = confidence_sum.get(pattern_type, 0) + pattern.confidence
            
            # Display pattern statistics
            for pattern_type, count in pattern_counts.items():
                avg_confidence = confidence_sum[pattern_type] / count
                st.write(f"**{pattern_type}:** {count} patterns (avg: {avg_confidence:.1%})")
            
            st.write("---")
            st.write(f"**Total Patterns:** {len(patterns)}")
            
            # Overall statistics
            if patterns:
                overall_avg = sum(p.confidence for p in patterns) / len(patterns)
                highest_conf = max(p.confidence for p in patterns)
                lowest_conf = min(p.confidence for p in patterns)
                
                st.write(f"**Average Confidence:** {overall_avg:.1%}")
                st.write(f"**Highest Confidence:** {highest_conf:.1%}")
                st.write(f"**Lowest Confidence:** {lowest_conf:.1%}")
        else:
            st.write("**No patterns detected**")
            st.write("Try adjusting:")
            st.write("• Different timeframe")
            st.write("• Additional pattern types")
            st.write("• Different instrument")
        
        st.write("---")
        st.write(f"**Instrument:** {instrument}")
        st.write(f"**Timeframe:** {timeframe}")
        st.write(f"**Analysis Time:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")


def create_themed_candlestick_chart(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str, chart_type: str = "main", **kwargs):
    """Create professional themed candlestick chart"""
    theme = get_chart_theme()
    
    # Chart configuration
    if chart_type == "main":
        height = 800
        show_volume = True
        title = f"{instrument} - {timeframe} Overview"
    elif chart_type == "pattern_specific":
        height = 600
        show_volume = False
        pattern_type = kwargs.get('pattern_type', 'Pattern')
        title = f"{instrument} - {pattern_type} Patterns"
    else:
        height = 700
        show_volume = True
        title = f"{instrument} - {timeframe}"
    
    # Create subplots
    if show_volume and 'volume' in data.columns:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=(title, 'Volume'),
            row_heights=[0.75, 0.25]
        )
        volume_row = 2
    else:
        fig = make_subplots(rows=1, cols=1)
        fig.update_layout(title=title)
        volume_row = None
    
    # Add candlestick trace
    candlestick = go.Candlestick(
        x=data['datetime'] if 'datetime' in data.columns else data.index,
        open=data['open'],
        high=data['high'],
        low=data['low'],
        close=data['close'],
        name="OHLC",
        increasing_line_color=theme['bullish_color'],
        decreasing_line_color=theme['bearish_color'],
        increasing_fillcolor=theme['bullish_fill'],
        decreasing_fillcolor=theme['bearish_fill'],
        line=dict(width=1.2),
        showlegend=False
    )
    
    if volume_row:
        fig.add_trace(candlestick, row=1, col=1)
    else:
        fig.add_trace(candlestick)
    
    # Add volume bars
    if volume_row and 'volume' in data.columns:
        volume_colors = [
            theme['bullish_volume'] if close >= open else theme['bearish_volume']
            for close, open in zip(data['close'], data['open'])
        ]
        
        volume_bars = go.Bar(
            x=data['datetime'] if 'datetime' in data.columns else data.index,
            y=data['volume'],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        )
        
        fig.add_trace(volume_bars, row=volume_row, col=1)
    
    # Add pattern markers
    if patterns:
        add_pattern_markers(fig, patterns, data, theme, chart_type, kwargs.get('selected_pattern'))
    
    # Apply theme
    apply_chart_theme(fig, theme, height, show_volume)
    
    return fig


def create_focused_pattern_chart(data: pd.DataFrame, pattern: PatternResult, instrument: str, timeframe: str):
    """Create focused chart for specific pattern with enhanced zoom functionality"""
    theme = get_chart_theme()
    
    # Calculate dynamic focus window based on timeframe
    timeframe_windows = {
        '1min': 100,
        '5min': 50,
        '15min': 30,
        '1hour': 24,
        '2hour': 12,
        '5hour': 10,
        '1day': 7
    }
    
    window_size = timeframe_windows.get(timeframe, 50)
    pattern_index = pattern.candle_index
    
    # Ensure we don't go out of bounds
    start_idx = max(0, pattern_index - window_size // 2)
    end_idx = min(len(data), pattern_index + window_size // 2)
    
    # Adjust if we're near the beginning or end
    if start_idx == 0:
        end_idx = min(len(data), window_size)
    elif end_idx == len(data):
        start_idx = max(0, len(data) - window_size)
    
    focused_data = data.iloc[start_idx:end_idx].copy()
    
    # Create focused chart with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=(
            f"🎯 {pattern.pattern_type.replace('_', ' ').title()} Pattern - {pattern.datetime.strftime('%Y-%m-%d %H:%M')}",
            'Volume'
        ),
        row_heights=[0.8, 0.2]
    )
    
    # Add candlestick trace with enhanced styling
    candlestick = go.Candlestick(
        x=focused_data['datetime'] if 'datetime' in focused_data.columns else focused_data.index,
        open=focused_data['open'],
        high=focused_data['high'],
        low=focused_data['low'],
        close=focused_data['close'],
        name="OHLC",
        increasing_line_color=theme['bullish_color'],
        decreasing_line_color=theme['bearish_color'],
        increasing_fillcolor=theme['bullish_fill'],
        decreasing_fillcolor=theme['bearish_fill'],
        line=dict(width=1.5),
        showlegend=False
    )
    
    fig.add_trace(candlestick, row=1, col=1)
    
    # Add volume with matching colors
    if 'volume' in focused_data.columns:
        volume_colors = [
            theme['bullish_volume'] if close >= open else theme['bearish_volume']
            for close, open in zip(focused_data['close'], focused_data['open'])
        ]
        
        volume_bars = go.Bar(
            x=focused_data['datetime'] if 'datetime' in focused_data.columns else focused_data.index,
            y=focused_data['volume'],
            name="Volume",
            marker_color=volume_colors,
            opacity=0.7,
            showlegend=False
        )
        
        fig.add_trace(volume_bars, row=2, col=1)
    
    # Highlight the pattern candle with enhanced markers
    adjusted_pattern_index = pattern_index - start_idx
    if 0 <= adjusted_pattern_index < len(focused_data):
        pattern_candle = focused_data.iloc[adjusted_pattern_index]
        pattern_x = pattern_candle['datetime'] if 'datetime' in focused_data.columns else adjusted_pattern_index
        
        # Add large star marker
        fig.add_trace(go.Scatter(
            x=[pattern_x],
            y=[pattern_candle['high'] * 1.03],
            mode='markers',
            name=f"{pattern.pattern_type.replace('_', ' ').title()}",
            marker=dict(
                symbol='star',
                size=25,
                color=theme['highlight_color'],
                line=dict(width=3, color='white')
            ),
            hovertemplate=f"<b>Pattern: {pattern.pattern_type.replace('_', ' ').title()}</b><br>" +
                         f"Confidence: {pattern.confidence:.1%}<br>" +
                         f"Time: {pattern.datetime}<br>" +
                         f"Price: ${pattern_candle['close']:.2f}<br>" +
                         f"OHLC: {pattern_candle['open']:.2f}/{pattern_candle['high']:.2f}/" +
                         f"{pattern_candle['low']:.2f}/{pattern_candle['close']:.2f}<extra></extra>",
            showlegend=True
        ), row=1, col=1)
        
        # Add vertical line to mark the pattern
        fig.add_vline(
            x=pattern_x,
            line_dash="dash",
            line_color=theme['highlight_color'],
            line_width=2,
            opacity=0.8,
            annotation_text=f"Pattern: {pattern.confidence:.1%}",
            annotation_position="top"
        )
        
        # Add horizontal line at pattern price level
        fig.add_hline(
            y=pattern_candle['close'],
            line_dash="dot",
            line_color=theme['highlight_color'],
            line_width=1,
            opacity=0.6,
            annotation_text=f"${pattern_candle['close']:.2f}",
            annotation_position="right"
        )
    
    # Enhanced styling with better layout
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h", 
            yanchor="bottom", 
            y=1.02, 
            xanchor="center", 
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.2)",
            borderwidth=1
        ),
        margin=dict(l=50, r=50, t=120, b=50),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_color'],
        font=dict(color=theme['text_color'], size=12),
        hovermode='x unified'
    )
    
    # Enhanced grid and axes
    fig.update_xaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=theme['grid_color'],
        showline=True,
        linewidth=1,
        linecolor=theme['grid_color'],
        title_text="Time"
    )
    fig.update_yaxes(
        showgrid=True, 
        gridwidth=1, 
        gridcolor=theme['grid_color'],
        showline=True,
        linewidth=1,
        linecolor=theme['grid_color']
    )
    
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def get_chart_theme():
    """Professional chart theme configuration"""
    return {
        'bullish_color': '#00C851',      # Green for bullish candles
        'bearish_color': '#FF4444',      # Red for bearish candles
        'bullish_fill': '#00C851',       # Green fill
        'bearish_fill': '#FF4444',       # Red fill
        'bullish_volume': '#00C851',     # Green volume
        'bearish_volume': '#FF4444',     # Red volume
        'highlight_color': '#FFD700',    # Gold for highlights
        'bg_color': '#FAFAFA',          # Light background
        'paper_color': '#FFFFFF',        # White paper
        'grid_color': '#E0E0E0',        # Light grid
        'text_color': '#333333'         # Dark text
    }


def add_pattern_markers(fig, patterns: List[PatternResult], data: pd.DataFrame, theme: dict, chart_type: str, selected_pattern=None):
    """Add pattern markers to chart"""
    try:
        for pattern in patterns:
            if pattern.candle_index < len(data):
                candle = data.iloc[pattern.candle_index]
                pattern_x = candle['datetime'] if 'datetime' in data.columns else pattern.candle_index
                
                # Determine marker properties
                if selected_pattern and pattern == selected_pattern:
                    marker_size = 15
                    marker_color = theme['highlight_color']
                    marker_symbol = 'star'
                else:
                    marker_size = 10
                    marker_color = theme['highlight_color']
                    marker_symbol = 'circle'
                
                # Add pattern marker
                fig.add_trace(go.Scatter(
                    x=[pattern_x],
                    y=[candle['high'] * 1.01],
                    mode='markers',
                    name=pattern.pattern_type.replace('_', ' ').title(),
                    marker=dict(
                        symbol=marker_symbol,
                        size=marker_size,
                        color=marker_color,
                        line=dict(width=1, color='white')
                    ),
                    hovertemplate=f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>" +
                                 f"Confidence: {pattern.confidence:.1%}<br>" +
                                 f"Time: {pattern.datetime}<br>" +
                                 f"Price: ${candle['close']:.2f}<extra></extra>",
                    showlegend=False
                ), row=1, col=1)
                
    except Exception as e:
        logger.error(f"Error adding pattern markers: {e}")


def apply_chart_theme(fig, theme: dict, height: int, show_volume: bool):
    """Apply professional theme to chart"""
    try:
        fig.update_layout(
            height=height,
            showlegend=False,
            margin=dict(l=50, r=50, t=80, b=50),
            plot_bgcolor=theme['bg_color'],
            paper_bgcolor=theme['paper_color'],
            font=dict(color=theme['text_color'], size=12),
            hovermode='x unified'
        )
        
        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=theme['grid_color'],
            showline=True,
            linewidth=1,
            linecolor=theme['grid_color']
        )
        
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=theme['grid_color'],
            showline=True,
            linewidth=1,
            linecolor=theme['grid_color']
        )
        
        # Volume chart specific styling
        if show_volume:
            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            
    except Exception as e:
        logger.error(f"Error applying chart theme: {e}")


def show_zoomed_pattern_view(data: pd.DataFrame, pattern: PatternResult, instrument_name: str, timeframe_name: str):
    """Show zoomed view of a pattern without reloading the page"""
    st.markdown("---")
    st.markdown(f"#### 🔍 Zoomed View: {pattern.pattern_type.replace('_', ' ').title()}")
    
    # Create focused view with smaller window
    window_size = 20  # Show 20 candles around the pattern
    start_idx = max(0, pattern.candle_index - window_size // 2)
    end_idx = min(len(data), pattern.candle_index + window_size // 2)
    
    zoomed_data = data.iloc[start_idx:end_idx].copy()
    
    if len(zoomed_data) > 0:
        # Create simplified chart for performance
        if st.session_state.fast_mode:
            chart = create_fast_candlestick_chart(zoomed_data, [pattern], instrument_name, timeframe_name)
        else:
            chart = create_themed_candlestick_chart(zoomed_data, [pattern], instrument_name, timeframe_name, chart_type="focused", height=400)
        
        st.plotly_chart(chart, use_container_width=True)
        
        # Pattern info
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Confidence", f"{pattern.confidence:.1%}")
            st.metric("Date/Time", pattern.datetime.strftime('%Y-%m-%d %H:%M'))
        with col2:
            if pattern.candle_index < len(data):
                candle = data.iloc[pattern.candle_index]
                st.metric("Close Price", f"${candle['close']:.2f}")
                price_change = candle['close'] - candle['open']
                st.metric("Price Change", f"${price_change:.2f}")


def display_pattern_details_inline(pattern: PatternResult, data: pd.DataFrame):
    """Display pattern details inline without triggering reloads"""
    try:
        if pattern.candle_index < len(data):
            candle = data.iloc[pattern.candle_index]
            
            # Simple two-column layout for performance
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Pattern Info:**")
                st.write(f"• Type: {pattern.pattern_type.replace('_', ' ').title()}")
                st.write(f"• Confidence: {pattern.confidence:.2%}")
                st.write(f"• Time: {pattern.datetime}")
                st.write(f"• Timeframe: {pattern.timeframe}")
                
            with col2:
                st.markdown("**Price Data:**")
                st.write(f"• Open: ${candle['open']:.2f}")
                st.write(f"• High: ${candle['high']:.2f}")
                st.write(f"• Low: ${candle['low']:.2f}")
                st.write(f"• Close: ${candle['close']:.2f}")
                
                # Price change
                price_change = candle['close'] - candle['open']
                price_change_pct = (price_change / candle['open']) * 100
                st.write(f"• Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
                
    except Exception as e:
        st.error(f"Error displaying pattern details: {e}")


def create_fast_candlestick_chart(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str):
    """Create a fast, simplified candlestick chart for better performance"""
    # Limit data for performance (show last 500 candles max)
    if len(data) > 500:
        display_data = data.tail(500).copy()
        # Adjust pattern indices
        offset = len(data) - 500
        adjusted_patterns = [p for p in patterns if p.candle_index >= offset]
        for pattern in adjusted_patterns:
            pattern.candle_index -= offset
    else:
        display_data = data
        adjusted_patterns = patterns
    
    fig = go.Figure()
    
    # Simple candlestick without volume
    fig.add_trace(go.Candlestick(
        x=display_data.index,
        open=display_data['open'],
        high=display_data['high'],
        low=display_data['low'],
        close=display_data['close'],
        name=f"{instrument} - {timeframe}",
        increasing_line_color='#26C281',
        decreasing_line_color='#E74C3C',
        increasing_fillcolor='#26C281',
        decreasing_fillcolor='#E74C3C',
        line=dict(width=1)
    ))
    
    # Add pattern markers (simplified and colored by confidence)
    for i, pattern in enumerate(adjusted_patterns[:20]):  # Limit to 20 patterns for performance
        if pattern.candle_index < len(display_data):
            candle = display_data.iloc[pattern.candle_index]
            
            # Color by confidence
            if pattern.confidence >= 0.8:
                color = '#FFD700'  # Gold for high confidence
                size = 14
            elif pattern.confidence >= 0.6:
                color = '#FFA500'  # Orange for medium confidence
                size = 12
            else:
                color = '#87CEEB'  # Light blue for low confidence
                size = 10
            
            fig.add_trace(go.Scatter(
                x=[pattern.candle_index],
                y=[candle['high'] * 1.02],
                mode='markers+text',
                text=[f"{pattern.confidence:.0%}"],
                textposition="top center",
                textfont=dict(size=8, color='white'),
                marker=dict(
                    symbol='star',
                    size=size,
                    color=color,
                    line=dict(width=1, color='black')
                ),
                hovertemplate=f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>" +
                             f"Confidence: {pattern.confidence:.1%}<br>" +
                             f"Time: {pattern.datetime}<extra></extra>",
                showlegend=False
            ))
    
    # Clean, simple layout
    fig.update_layout(
        height=500,
        showlegend=False,
        margin=dict(l=50, r=50, t=60, b=50),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=12),
        title=dict(
            text=f"{instrument} - {timeframe} (Fast Mode) - {len(adjusted_patterns)} patterns",
            x=0.5,
            font=dict(size=16)
        ),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            title="Price"
        )
    )
    
    return fig


if __name__ == "__main__":
    main()