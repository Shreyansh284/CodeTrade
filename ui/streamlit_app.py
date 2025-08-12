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
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional
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
from patterns.double_top import DoubleTopDetector
from patterns.double_bottom import DoubleBottomDetector
from patterns.head_and_shoulders import HeadAndShouldersDetector
from patterns.inverse_head_and_shoulders import InverseHeadAndShouldersDetector
from patterns.base import PatternResult
from visualization.charts import ChartRenderer
from utils.logging_config import get_logger

# NEW: structural scan
from patterns.structural_scan import StructuralScanner

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
    'Three White Soldiers': ThreeWhiteSoldiersDetector(),
    'Double Top': DoubleTopDetector(),
    'Double Bottom': DoubleBottomDetector(),
    'Head and Shoulders': HeadAndShouldersDetector(),
    'Inverse Head and Shoulders': InverseHeadAndShouldersDetector()
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
    if 'struct_scanner' not in st.session_state:
        st.session_state.struct_scanner = StructuralScanner()


def create_sidebar():
    """Create clean sidebar configuration"""
    st.sidebar.header("📊 Configuration")
    
    # Get available instruments with optional filter
    try:
        instruments_all = st.session_state.data_loader.get_available_instruments()
        if not instruments_all:
            st.sidebar.error("No instruments found")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading instruments: {e}")
        return None

    search_term = st.sidebar.text_input("🔎 Filter Stocks", "", help="Type part of symbol to filter list")
    if search_term:
        instruments = [s for s in instruments_all if search_term.lower() in s.lower()]
        if not instruments:
            st.sidebar.warning("No matches for filter")
            instruments = instruments_all
    else:
        instruments = instruments_all

    loader = st.session_state.data_loader
    multi_mode = getattr(loader, 'flat_mode', False)
    if multi_mode:
        selected_instruments = st.sidebar.multiselect(
            "📈 Select Stock(s)",
            instruments,
            default=instruments[:1],
            help="Choose one or more stocks to analyze"
        )
    else:
        selected_instruments = [st.sidebar.selectbox(
            "📈 Select Stock",
            instruments,
            help="Choose the stock to analyze"
        )]
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    
    # Get available dates for primary selected instrument
    available_dates = []
    primary_instrument = selected_instruments[0] if selected_instruments else None
    if not primary_instrument:
        st.sidebar.warning("Select at least one instrument")
        return None
    try:
        available_dates = st.session_state.data_loader.get_available_dates(primary_instrument)
        if not available_dates:
            st.sidebar.error("No dates available for selected instrument")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading dates: {e}")
        logger.error(f"Error getting available dates for {primary_instrument}: {e}")
        return None
    
    # Date range selection option
    date_selection_mode = st.sidebar.radio(
        "Select Date Range Mode:",
        options=["All Available Data", "Custom Date Range"],
        help="Choose whether to use all data or select specific date range"
    )
    
    start_date = None
    end_date = None
    
    if date_selection_mode == "Custom Date Range":
        min_date = min(available_dates)
        max_date = max(available_dates)
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=min_date,
                min_value=min_date,
                max_value=max_date,
                help="Select starting date for analysis"
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date,
                min_value=min_date,
                max_value=max_date,
                help="Select ending date for analysis"
            )
        
        # Validate date range
        if start_date > end_date:
            st.sidebar.error("⚠️ Start date must be before end date!")
            return None
        
        # Filter available dates to only include those in the selected range
        dates_in_range = [d for d in available_dates if start_date <= d <= end_date]
        if not dates_in_range:
            st.sidebar.error("⚠️ No data available in the selected date range!")
            return None
        
        # Show selected date range info
        st.sidebar.info(f"📊 Selected: {len(dates_in_range)} day(s) of data")
    else:
        # Show info about all available data
        total_days = len(available_dates)
        date_range = f"{min(available_dates).strftime('%d-%m-%Y')} to {max(available_dates).strftime('%d-%m-%Y')}"
        st.sidebar.info(f"📊 Using all data: {total_days} day(s)")
        st.sidebar.caption(f"Range: {date_range}")
    
    # Timeframe removed for flat daily mode; retain selection only if legacy intraday available
    loader = st.session_state.data_loader
    if getattr(loader, 'flat_mode', False):
        selected_timeframe = '1day'
        st.sidebar.info("Full dataset daily analysis (no timeframe selection needed).")
    else:
        all_timeframes = st.session_state.data_aggregator.get_supported_timeframes()
        default_index = 1 if len(all_timeframes) > 1 else 0
        selected_timeframe = st.sidebar.selectbox(
            "⏱️ Timeframe",
            all_timeframes,
            index=default_index,
            help="Select analysis timeframe"
        )
    
    # Pattern selection
    st.sidebar.subheader("🔍 Patterns")
    selected_patterns = {}
    all_checked = st.sidebar.checkbox("Select All", value=True)
    for name in PATTERN_DETECTORS.keys():
        selected_patterns[name] = st.sidebar.checkbox(
            name.replace('_', ' '),
            value=all_checked,
            help=f"Detect {name} patterns"
        )

    # Structural scan controls
    st.sidebar.subheader("🏗️ Structural Scan (Multi-month)")
    enable_struct = st.sidebar.checkbox("Enable 6-month structural scan", value=True)
    struct_months = st.sidebar.select_slider("Window", options=[3, 6, 9, 12], value=6, help="Sliding window size in months")
    struct_patterns = st.sidebar.multiselect(
        "Structural Patterns",
        options=["Head & Shoulders", "Double Top", "Double Bottom"],
        default=["Head & Shoulders", "Double Top", "Double Bottom"],
        help="Patterns to search across the multi-month window"
    )

    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Min Confidence", 0.0, 1.0, 0.0, 0.05,
        help="Only show patterns with confidence >= threshold"
    )

    
    # Quick settings
    st.sidebar.subheader("⚙️ Settings")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        max_candles = st.number_input(
            "Max Candles",
            min_value=50,
            max_value=2000,
            value=500,
            step=50,
            help="Maximum candles to display"
        )
    with col2:
        chart_height = st.number_input(
            "Chart Height",
            min_value=300,
            max_value=1000,
            value=550,
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
        'instruments': selected_instruments,
        'instrument': selected_instruments[0],  # backward compat for single use
        'timeframe': selected_timeframe,
        'patterns': selected_patterns,
        'max_candles': max_candles,
        'chart_height': chart_height,
        'run_analysis': run_analysis,
        'date_mode': date_selection_mode,
        'start_date': start_date,
        'end_date': end_date,
        'confidence_threshold': confidence_threshold,
        'multi_mode': multi_mode,
        # structural
        'enable_struct': enable_struct,
        'struct_months': struct_months,
        'struct_patterns': struct_patterns
    }


def show_welcome_screen():
    """Show welcome screen when no analysis is running"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 👋 Welcome to Stock Pattern Detector
        
        **Get Started:**
        1. Select a stock instrument from the sidebar
        2. Select patterns to detect
        3. Click "Analyze Patterns"
        
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
    instruments = config.get('instruments', [config['instrument']])
    instrument = instruments[0]
    timeframe = config['timeframe']
    patterns = config['patterns']
    date_mode = config['date_mode']
    start_date = config['start_date']
    end_date = config['end_date']
    
    # Check if any patterns are selected
    selected_pattern_names = [name for name, selected in patterns.items() if selected]
    if not selected_pattern_names:
        st.warning("⚠️ Please select at least one pattern to detect")
        return
    
    # Progress tracking
    progress_container = st.container()
    with progress_container:
        date_info = ""
        if date_mode == "Custom Date Range" and start_date and end_date:
            date_info = f" ({start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')})"
        
    title_tf = '' if (getattr(st.session_state.data_loader, 'flat_mode', False)) else f" - {timeframe}"
    multi_suffix = '' if len(instruments) == 1 else f" (+{len(instruments)-1} more)"
    st.markdown(f"### 📊 Analyzing {instrument}{multi_suffix}{title_tf}{date_info}")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Load data with optional date filtering
    status_text.info("📥 Loading data...")
    progress_bar.progress(20)

    # Convert date objects for the loader if needed
    start_datetime = None
    end_datetime = None

    if date_mode == "Custom Date Range":
        if start_date:
            start_datetime = start_date
        if end_date:
            end_datetime = end_date
        
    # Load all instruments (sequential for simplicity)
    data_map = {}
    for inst in instruments:
        data_map[inst] = st.session_state.data_loader.load_instrument_data(inst, start_date=start_datetime, end_date=end_datetime)
    data = data_map[instrument]
        
    if data is None or data.empty:
        if date_mode == "Custom Date Range":
            st.error(f"❌ No data available for {instrument} in the selected date range")
        else:
            st.error(f"❌ No data available for {instrument}")
        return
        
    # Aggregate data
    if getattr(st.session_state.data_loader, 'flat_mode', False):
        status_text.info("🔄 Preparing data (daily)...")
        progress_bar.progress(40)
        agg_map = {inst: df.copy() for inst, df in data_map.items() if df is not None}
    else:
        status_text.info("🔄 Processing timeframe...")
        progress_bar.progress(40)
        agg_map = {}
        for inst, df in data_map.items():
            if df is not None:
                agg_map[inst] = st.session_state.data_aggregator.aggregate_data(df, timeframe)
    agg_data = agg_map.get(instrument)
    if agg_data is None or agg_data.empty:
        st.error(f"❌ Failed to process {timeframe} data")
        return
        
    # Detect standard patterns
    status_text.info("🔍 Detecting candlestick patterns...")
    progress_bar.progress(60)
    
    pattern_results_map = {}
    cf = config['confidence_threshold']
    for inst, df in agg_map.items():
        inst_patterns = []
        if df is None or df.empty:
            pattern_results_map[inst] = []
            continue
        for pattern_name in selected_pattern_names:
            detector = PATTERN_DETECTORS[pattern_name]
            try:
                found_patterns = detector.detect(df, timeframe)
                found_patterns = [p for p in found_patterns if p.confidence >= cf]
                inst_patterns.extend(found_patterns)
            except Exception as e:
                st.warning(f"⚠️ Error detecting {pattern_name} in {inst}: {e}")
        pattern_results_map[inst] = inst_patterns
    all_patterns = pattern_results_map[instrument]

    # Structural scan over multi-month window (sliding)
    struct_segments = []
    if config.get('enable_struct', True):
        status_text.info("🏗️ Running structural scan...")
        progress_bar.progress(75)
        months = int(config.get('struct_months', 6))
        days_window = int(months * 21)  # ~21 trading days per month
        # Use daily candles for structural scan to represent months clearly
        if getattr(st.session_state.data_loader, 'flat_mode', False):
            df_struct = agg_data.copy()
        else:
            try:
                df_struct = st.session_state.data_aggregator.aggregate_data(data, '1day')
            except Exception:
                df_struct = agg_data.copy()
        df = df_struct
        if df is not None and len(df) >= max(60, days_window):
            # Sliding window with step ~10-20% of window
            step = max(10, days_window // 6)
            patterns_map = {
                "Head & Shoulders": "head_and_shoulders",
                "Double Top": "double_top",
                "Double Bottom": "double_bottom",
            }
            wanted = [patterns_map[p] for p in config.get('struct_patterns', []) if p in patterns_map]
            if not wanted:
                wanted = ["head_and_shoulders", "double_top", "double_bottom"]
            segments_collected: List[Dict[str, Any]] = []
            for start in range(0, max(1, len(df) - days_window + 1), step):
                end = start + days_window
                window = df.iloc[start:end]
                try:
                    segs = st.session_state.struct_scanner.scan(window, patterns=wanted)
                    # Offset indices to global df
                    for s in segs:
                        s['start_idx'] = s['start_idx'] + start
                        s['end_idx'] = s['end_idx'] + start
                        # adjust datetimes
                        if 'datetime' in df.columns:
                            s['start_dt'] = df['datetime'].iloc[s['start_idx']]
                            s['end_dt'] = df['datetime'].iloc[s['end_idx']]
                    segments_collected.extend(segs)
                except Exception:
                    pass
            # sort and keep top by confidence
            struct_segments = sorted(segments_collected, key=lambda x: x.get('confidence', 0), reverse=True)[:10]
        else:
            st.info("Not enough data for structural scan window")

    # Create visualization
    status_text.info("📈 Creating chart...")
    progress_bar.progress(90)
    
    # Clear progress
    progress_bar.progress(100)
    time.sleep(0.3)
    progress_container.empty()
    
    # Display results
    display_results(instrument, timeframe, agg_data, all_patterns, config, pattern_results_map if len(instruments)>1 else None, struct_segments)


def display_results(instrument: str, timeframe: str, data: pd.DataFrame, 
                   patterns: List[PatternResult], config: Dict[str, Any], multi_results: Dict[str, List[PatternResult]] = None,
                   struct_segments: List[Dict[str, Any]] = None):
    """Display analysis results with clean layout"""
    
    # Show date range info if custom range was used
    if config.get('date_mode') == "Custom Date Range":
        start_date = config.get('start_date')
        end_date = config.get('end_date')
        if start_date and end_date:
            st.info(f"📅 Analysis period: {start_date.strftime('%d-%m-%Y')} to {end_date.strftime('%d-%m-%Y')} ({(end_date - start_date).days + 1} days)")
    
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
        # Overlay structural segments if any
        if struct_segments:
            st.session_state.chart_renderer.add_structural_segments(chart, data, struct_segments)
        st.plotly_chart(chart, use_container_width=True)
        # Download detected patterns (if any) as CSV
        if patterns or struct_segments:
            try:
                export_rows = []
                for p in patterns or []:
                    export_rows.append({
                        'instrument': instrument,
                        'datetime': p.datetime.strftime('%Y-%m-%d %H:%M:%S'),
                        'pattern_type': p.pattern_type,
                        'confidence': p.confidence,
                        'type': 'candle'
                    })
                for s in struct_segments or []:
                    export_rows.append({
                        'instrument': instrument,
                        'start_dt': s.get('start_dt'),
                        'end_dt': s.get('end_dt'),
                        'pattern_type': s.get('pattern_type'),
                        'confidence': s.get('confidence'),
                        'status': s.get('status'),
                        'type': 'structural'
                    })
                if export_rows:
                    export_df = pd.DataFrame(export_rows)
                    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="💾 Download Findings CSV",
                        data=csv_bytes,
                        file_name=f"findings_{instrument}_{timeframe}.csv",
                        mime='text/csv',
                        use_container_width=True
                    )
            except Exception as e:
                st.warning(f"Export unavailable: {e}")
        
    except Exception as e:
        st.error(f"❌ Error creating chart: {e}")
    
    # Multi-instrument summary (if applicable)
    if multi_results:
        try:
            st.markdown("### 🧮 Multi-Instrument Pattern Summary")
            summary_rows = []
            for inst, plist in multi_results.items():
                counts = {}
                for p in plist:
                    counts[p.pattern_type] = counts.get(p.pattern_type, 0) + 1
                summary_rows.append({
                    'Instrument': inst,
                    'Total': len(plist),
                    **{k.replace('_',' ').title(): v for k, v in counts.items()}
                })
            if summary_rows:
                summary_df = pd.DataFrame(summary_rows).sort_values('Total', ascending=False)
                st.dataframe(summary_df, use_container_width=True)
        except Exception as e:
            st.warning(f"Summary error: {e}")

    # Pattern details
    if patterns:
        display_pattern_details(patterns, data, config)
    else:
        st.info("ℹ️ No single-candle patterns detected in the selected timeframe")

    # Structural segments details
    if struct_segments:
        st.markdown("### 🧩 Structural Pattern Segments")
        try:
            df = pd.DataFrame(struct_segments)
            if not df.empty:
                # prettify
                df['pattern'] = df['pattern_type'].str.replace('_', ' ').str.title()
                df['confidence'] = (df['confidence'].astype(float) * 100).round(1).astype(str) + '%'
                df = df[['pattern', 'status', 'confidence', 'start_dt', 'end_dt', 'start_idx', 'end_idx']]
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display structural segments: {e}")

    # Instrument summary (always show at bottom)
    try:
        info = st.session_state.data_loader.get_data_info(instrument)
        if info:
            with st.expander("📄 Instrument Data Summary"):
                st.write({
                    'Records': info.get('total_records'),
                    'Date Range': info.get('date_range'),
                    'Price Range': info.get('price_range'),
                    'Total Volume': info.get('total_volume'),
                    'Price Change %': round(info.get('price_change_pct', 0.0), 2)
                })
    except Exception:
        pass


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
