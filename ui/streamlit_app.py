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
    """Main application entry point"""
    st.title("📈 Stock Pattern Detector")
    st.markdown("Professional trading interface for candlestick pattern analysis")
    
    # Initialize session state
    initialize_session_state()
    
    # Create sidebar controls
    config = create_sidebar()
    if not config:
        return
    
    # Main content area
    display_main_content(config)


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
    selected_timeframe = st.sidebar.selectbox(
        "Select timeframe:",
        timeframes,
        index=1,  # Default to 5min
        help="Choose the time period for candlestick aggregation"
    )
    
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
    if st.sidebar.button("🔍 Start Analysis", type="primary", help="Begin pattern detection analysis"):
        return {
            'instrument': selected_instrument,
            'timeframe': selected_timeframe,
            'patterns': selected_patterns,
            'trigger_analysis': True
        }
    
    return {
        'instrument': selected_instrument,
        'timeframe': selected_timeframe,
        'patterns': selected_patterns,
        'trigger_analysis': False
    }


def display_main_content(config: Dict[str, Any]):
    """Display main content area with analysis results"""
    if config['trigger_analysis']:
        run_analysis(config['instrument'], config['timeframe'], config['patterns'])
    else:
        st.info("👈 Configure your analysis settings in the sidebar and click **Start Analysis** to begin.")


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
    st.markdown("### 📈 Main Chart")
    
    if not data.empty:
        try:
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
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_pattern_type = st.selectbox(
                "Select Pattern Type to View:",
                ["All Patterns"] + pattern_types,
                help="Choose a specific pattern type to focus on"
            )
        
        with col2:
            show_individual_charts = st.checkbox(
                "Show Individual Pattern Charts", 
                value=False,
                help="Display separate charts for each pattern type"
            )
        
        # Filter patterns
        if selected_pattern_type == "All Patterns":
            filtered_patterns = patterns
        else:
            filtered_patterns = [p for p in patterns if p.pattern_type.replace('_', ' ').title() == selected_pattern_type]
        
        # Individual Pattern Charts
        if show_individual_charts and filtered_patterns:
            display_individual_pattern_charts(data, filtered_patterns, instrument, timeframe)
        
        # Pattern Navigation Table
        display_pattern_navigation_table(filtered_patterns, data)
        
        # Focused Pattern View
        if st.session_state.selected_pattern and st.session_state.show_pattern_details:
            display_focused_pattern_view(data, instrument, timeframe)
        
        # Export and Summary
        st.markdown("---")
        display_export_and_summary(patterns, instrument, timeframe)
    
    else:
        st.info("🔍 No patterns detected in the selected timeframe and criteria.")


def display_individual_pattern_charts(data: pd.DataFrame, patterns: List[PatternResult], instrument: str, timeframe: str):
    """Display individual charts for each pattern type"""
    st.markdown("#### 📊 Individual Pattern Charts")
    
    # Group patterns by type
    pattern_groups = {}
    for pattern in patterns:
        pattern_type = pattern.pattern_type.replace('_', ' ').title()
        if pattern_type not in pattern_groups:
            pattern_groups[pattern_type] = []
        pattern_groups[pattern_type].append(pattern)
    
    # Create tabs for multiple pattern types
    if len(pattern_groups) > 1:
        tabs = st.tabs(list(pattern_groups.keys()))
        for i, (pattern_type, type_patterns) in enumerate(pattern_groups.items()):
            with tabs[i]:
                display_pattern_type_chart(data, type_patterns, instrument, timeframe, pattern_type)
    else:
        # Single pattern type
        pattern_type = list(pattern_groups.keys())[0]
        type_patterns = pattern_groups[pattern_type]
        display_pattern_type_chart(data, type_patterns, instrument, timeframe, pattern_type)


def display_pattern_navigation_table(patterns: List[PatternResult], data: pd.DataFrame):
    """Display enhanced pattern navigation table"""
    st.markdown("#### 🎯 Pattern Details & Navigation")
    
    # Sort patterns by datetime (most recent first)
    sorted_patterns = sorted(patterns, key=lambda p: p.datetime, reverse=True)
    
    # Create navigation table
    for i, pattern in enumerate(sorted_patterns):
        with st.container():
            col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 1, 1, 1, 1])
            
            with col1:
                st.write(f"**{pattern.datetime.strftime('%Y-%m-%d %H:%M')}**")
            with col2:
                st.write(f"**{pattern.pattern_type.replace('_', ' ').title()}**")
            with col3:
                st.write(f"{pattern.confidence:.1%}")
            with col4:
                st.write(pattern.timeframe)
            with col5:
                if st.button("🔍 Zoom", key=f"zoom_{i}", help="Zoom into this pattern"):
                    st.session_state.selected_pattern = pattern
                    st.session_state.show_pattern_details = True
                    st.rerun()
            with col6:
                if st.button("📋 Details", key=f"details_{i}", help="Show pattern details"):
                    with st.expander(f"Pattern Details - {pattern.pattern_type.replace('_', ' ').title()}", expanded=True):
                        display_pattern_details(pattern, data)


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
    """Display chart for specific pattern type"""
    try:
        chart = create_themed_candlestick_chart(
            data, patterns, instrument, timeframe, 
            chart_type="pattern_specific", pattern_type=pattern_type
        )
        st.plotly_chart(chart, use_container_width=True, key=f"chart_{pattern_type}")
        
        # Pattern statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Count", len(patterns))
        with col2:
            avg_conf = sum(p.confidence for p in patterns) / len(patterns)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col3:
            st.metric("Timeframe", timeframe)
            
    except Exception as e:
        st.error(f"❌ Error creating {pattern_type} chart: {e}")


def display_pattern_details(pattern: PatternResult, data: pd.DataFrame):
    """Display detailed pattern information"""
    try:
        if pattern.candle_index < len(data):
            candle = data.iloc[pattern.candle_index]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Pattern Information:**")
                st.write(f"• Type: {pattern.pattern_type.replace('_', ' ').title()}")
                st.write(f"• Confidence: {pattern.confidence:.2%}")
                st.write(f"• DateTime: {pattern.datetime}")
                st.write(f"• Timeframe: {pattern.timeframe}")
                st.write(f"• Candle Index: {pattern.candle_index}")
            
            with col2:
                st.write("**Candle Data:**")
                st.write(f"• Open: ${candle['open']:.2f}")
                st.write(f"• High: ${candle['high']:.2f}")
                st.write(f"• Low: ${candle['low']:.2f}")
                st.write(f"• Close: ${candle['close']:.2f}")
                if 'volume' in candle:
                    st.write(f"• Volume: {candle['volume']:,.0f}")
                    
    except Exception as e:
        st.error(f"❌ Error displaying pattern details: {e}")


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
    """Create focused chart for specific pattern"""
    theme = get_chart_theme()
    
    # Calculate focus window
    pattern_index = pattern.candle_index
    window_size = 50
    
    start_idx = max(0, pattern_index - window_size // 2)
    end_idx = min(len(data), pattern_index + window_size // 2)
    
    focused_data = data.iloc[start_idx:end_idx].copy()
    
    # Create focused chart
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
    
    # Add candlestick trace
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
    
    # Add volume
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
    
    # Highlight pattern candle
    if pattern_index >= start_idx and pattern_index < end_idx:
        pattern_candle = data.iloc[pattern_index]
        pattern_x = pattern_candle['datetime'] if 'datetime' in data.columns else pattern_index
        
        # Add star marker
        fig.add_trace(go.Scatter(
            x=[pattern_x],
            y=[pattern_candle['high'] * 1.02],
            mode='markers',
            name=f"{pattern.pattern_type.replace('_', ' ').title()}",
            marker=dict(
                symbol='star',
                size=20,
                color=theme['highlight_color'],
                line=dict(width=2, color='white')
            ),
            hovertemplate=f"<b>Pattern: {pattern.pattern_type.replace('_', ' ').title()}</b><br>" +
                         f"Confidence: {pattern.confidence:.1%}<br>" +
                         f"Time: {pattern.datetime}<br>" +
                         f"Price: ${pattern_candle['close']:.2f}<extra></extra>",
            showlegend=True
        ), row=1, col=1)
        
        # Add vertical line
        fig.add_vline(
            x=pattern_x,
            line_dash="dash",
            line_color=theme['highlight_color'],
            opacity=0.7,
            annotation_text=f"Pattern: {pattern.confidence:.1%}",
            annotation_position="top"
        )
    
    # Apply styling
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=100, b=50),
        plot_bgcolor=theme['bg_color'],
        paper_bgcolor=theme['paper_color']
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=theme['grid_color'])
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=theme['grid_color'])
    
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


if __name__ == "__main__":
    main()