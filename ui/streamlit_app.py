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
    
    # Get available instruments
    try:
        instruments_all = st.session_state.data_loader.get_available_instruments()
        if not instruments_all:
            st.sidebar.error("No instruments found")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading instruments: {e}")
        return None

    # Simple instrument picker (no clutter)
    instrument = st.sidebar.selectbox("📈 Select Stock", instruments_all)
    selected_instruments = [instrument]

    # Get available dates to drive timeframe slider
    try:
        available_dates = st.session_state.data_loader.get_available_dates(instrument)
        if not available_dates:
            st.sidebar.error("No dates available for selected instrument")
            return None
    except Exception as e:
        st.sidebar.error(f"Error loading dates: {e}")
        return None

    min_date = min(available_dates)
    max_date = max(available_dates)
    total_years = max(1, int((max_date - min_date).days // 365))

    # Timeframe selection: handle small datasets gracefully
    st.sidebar.subheader("⏱️ Timeframe")
    if total_years <= 1:
        years_window = 1
        st.sidebar.caption(
            f"Limited history available (~{(max_date - min_date).days} days). Using all data.")
    else:
        years_window = st.sidebar.slider(
            "Years to analyze",
            min_value=1,
            max_value=total_years,
            value=total_years,
            key=f"years_slider_{instrument}",  # reset when stock changes
            help="Select how many recent years to include (ending at the most recent date)"
        )
    st.sidebar.caption(f"Data range: {min_date.strftime('%d-%m-%Y')} → {max_date.strftime('%d-%m-%Y')}")

    # Structural pattern selection (only the three requested)
    st.sidebar.subheader("🔍 Patterns")
    pattern_options = ["Head & Shoulders", "Double Top", "Double Bottom"]
    selected_struct_patterns = st.sidebar.multiselect(
        "Select Patterns",
        options=pattern_options,
        default=pattern_options,
        help="Choose which structural patterns to find"
    )

    # Confidence threshold
    min_confidence = st.sidebar.slider(
        "Min Confidence",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Only show patterns with confidence ≥ this value"
    )

    # Run button
    run_analysis = st.sidebar.button("🚀 Analyze", type="primary", use_container_width=True)

    # Minimal config
    return {
        'instruments': selected_instruments,
        'instrument': instrument,
        'timeframe': '1day',
        'run_analysis': run_analysis,
        # timeframe selection
        'years_window': years_window,
        'total_years': total_years,
        # selected structural patterns
        'struct_patterns': selected_struct_patterns,
        'min_confidence': min_confidence,
        # Simple chart defaults
        'chart_height': 600,
        'max_candles': 100000,
    }


def show_welcome_screen():
    """Show welcome screen when no analysis is running"""
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        ### 👋 Welcome
        
        Detect large multi-month patterns:
        - Head & Shoulders
        - Double Top
        - Double Bottom
        
        Default view scans the full available timeframe. Use the Years slider in the sidebar to reduce the window.
        """)
        
        st.info("👈 Select a stock, adjust Years if needed, and click Analyze")
    
    # Quick preview remains minimal
    show_quick_preview()


def show_quick_preview():
    """Show a quick preview chart of sample data"""
    try:
        st.markdown("### 📊 Quick Preview")
        
        instruments = st.session_state.data_loader.get_available_instruments()
        if instruments:
            preview_instrument = instruments[0]
            data = st.session_state.data_loader.load_instrument_data(preview_instrument)
            if data is not None and not data.empty:
                preview_data = data.tail(80)
                preview_chart = st.session_state.chart_renderer.create_compact_chart(
                    data=preview_data,
                    patterns=None,
                    title=f"{preview_instrument} - Sample",
                    height=300
                )
                st.plotly_chart(preview_chart, use_container_width=True, key="preview_chart")
    except Exception:
        pass


def run_analysis(config: Dict[str, Any]):
    """Run structural pattern analysis with two charts"""
    instrument = config['instrument']

    progress_container = st.container()

    # Build the selected window using the years slider (ending at latest date)
    try:
        available_dates = st.session_state.data_loader.get_available_dates(instrument)
        max_date = max(available_dates)
        min_date = min(available_dates)
        total_years = max(1, int((max_date - min_date).days // 365))
    except Exception:
        st.error("Failed to determine available dates for instrument")
        return

    years = int(config.get('years_window', total_years))
    # Clamp years to available range (handles stock change where old slider value was larger)
    years = max(1, min(years, total_years))
    start_date = max_date - timedelta(days=365 * years)
    date_info = f" (Last {years} year(s))"

    with progress_container:
        st.markdown(f"### 📊 Analyzing {instrument}{date_info}")

    progress_bar = st.progress(0)
    status_text = st.empty()

    # Load data (selected window)
    status_text.info("📥 Loading data...")
    progress_bar.progress(15)

    df_sel = st.session_state.data_loader.load_instrument_data(
        instrument,
        start_date=start_date,
        end_date=None
    )

    if df_sel is None or df_sel.empty:
        st.error("❌ No data available for selected timeframe")
        return

    # Full data (for bigger timeframe context)
    df_full = st.session_state.data_loader.load_instrument_data(instrument)
    if df_full is None or df_full.empty:
        df_full = df_sel.copy()

    # Ensure daily candles
    try:
        df_sel_daily = st.session_state.data_aggregator.aggregate_data(df_sel, '1day') if not getattr(st.session_state.data_loader, 'flat_mode', False) else df_sel.copy()
    except Exception:
        df_sel_daily = df_sel.copy()
    try:
        df_full_daily = st.session_state.data_aggregator.aggregate_data(df_full, '1day') if not getattr(st.session_state.data_loader, 'flat_mode', False) else df_full.copy()
    except Exception:
        df_full_daily = df_full.copy()

    # Structural scan with adaptive timeframe fallback
    status_text.info("🏗️ Scanning for structural patterns...")
    progress_bar.progress(55)

    # Map selected human names to internal codes
    mapping = {
        "Head & Shoulders": 'head_and_shoulders',
        "Double Top": 'double_top',
        "Double Bottom": 'double_bottom',
    }
    wanted = [mapping[p] for p in config.get('struct_patterns', []) if p in mapping]
    if not wanted:
        wanted = list(mapping.values())

    # Build candidate months list (decreasing timeframe)
    sel_months = years * 12
    candidates = []
    for m in [sel_months, 96, 72, 60, 48, 36, 24, 12, 9, 6, 3]:
        if m > 0 and m <= sel_months:
            candidates.append(m)
    # Ensure unique order
    seen = set()
    months_list = []
    for m in candidates:
        if m not in seen:
            months_list.append(m)
            seen.add(m)

    struct_segments = []
    used_months = 0
    attempted_months: List[int] = []

    for months in months_list:
        days_window = int(months * 21)
        if len(df_full_daily) < max(60, days_window):
            continue
        attempted_months.append(months)
        window = df_full_daily.iloc[-days_window:]
        try:
            segs = st.session_state.struct_scanner.scan(window, patterns=wanted)
            # Adjust to global indices (relative to df_full_daily)
            offset = len(df_full_daily) - len(window)
            for s in segs:
                s['start_idx'] = s['start_idx'] + offset
                s['end_idx'] = s['end_idx'] + offset
                if 'datetime' in df_full_daily.columns:
                    s['start_dt'] = df_full_daily['datetime'].iloc[s['start_idx']]
                    s['end_dt'] = df_full_daily['datetime'].iloc[s['end_idx']]
            # Filter by confidence threshold
            threshold = float(config.get('min_confidence', 0.5))
            segs = [s for s in segs if s.get('confidence', 0.0) >= threshold]
            if segs:
                struct_segments = sorted(segs, key=lambda x: x.get('confidence', 0.0), reverse=True)
                used_months = months
                break
        except Exception:
            continue

    # Count those happening "now" (ending within last 3 bars of full data)
    now_cut = len(df_full_daily) - 3
    now_segments = [s for s in struct_segments if s.get('end_idx', 0) >= now_cut]

    # Charts
    status_text.info("📈 Creating charts...")
    progress_bar.progress(80)

    # Finish progress and render in display_results
    progress_bar.progress(100)
    time.sleep(0.2)
    progress_container.empty()

    # Display results
    config['used_months'] = used_months
    config['attempted_months'] = attempted_months
    config['years_window'] = years
    display_results(
        instrument=instrument,
        data_main=df_sel_daily,
        data_big=df_full_daily,
        struct_segments=struct_segments,
        now_count=len(now_segments),
        config=config
    )


def display_results(instrument: str, data_main: pd.DataFrame, data_big: pd.DataFrame,
                    struct_segments: List[Dict[str, Any]], now_count: int, config: Dict[str, Any]):
    """Display two charts and a concise summary"""

    used_months = int(config.get('used_months', 0))
    used_years = used_months / 12 if used_months else None

    if not struct_segments:
        # No detections – be explicit and helpful
        attempted = config.get('attempted_months', [])
        attempted_str = ", ".join(str(m) for m in attempted) if attempted else "-"
        st.warning(
            f"No structural patterns found ≥ {config.get('min_confidence',0.5):.2f} confidence.\n"
            f"Tried windows (months): {attempted_str}.\n"
            f"Tips: try a smaller Years window or lower the Min Confidence."
        )
    else:
        window_note = f" using ~{used_years:.1f}y window" if used_years else ""
        st.success(f"Found {len(struct_segments)} structural pattern segment(s){window_note}. {now_count} near the current date.")

    # Helper: map global segments to main data indices (if overlapping)
    def map_segments_to_df(df: pd.DataFrame, segments: List[Dict[str, Any]]):
        if df is None or df.empty or not segments or 'datetime' not in df.columns:
            return []
        dts = df['datetime']
        idx_map = {dt: i for i, dt in enumerate(dts)}
        mapped = []
        for s in segments:
            sd = s.get('start_dt')
            ed = s.get('end_dt')
            if sd in idx_map and ed in idx_map:
                ns = dict(s)
                ns['start_idx'] = idx_map[sd]
                ns['end_idx'] = idx_map[ed]
                mapped.append(ns)
        return mapped

    # Main chart (selected timeframe)
    st.markdown("### 📈 Main Chart (Selected Timeframe)")
    try:
        chart = st.session_state.chart_renderer.create_simple_chart(
            data=data_main,
            patterns=None,
            title=f"{instrument}",
            height=config['chart_height'],
            max_candles=config['max_candles']
        )
        segs_on_main = map_segments_to_df(data_main, struct_segments)
        if segs_on_main:
            st.session_state.chart_renderer.add_structural_segments(chart, data_main, segs_on_main)
        st.plotly_chart(chart, use_container_width=True, key=f"main_chart_{instrument}_{config.get('years_window','all')}_{len(data_main)}")
    except Exception as e:
        st.error(f"❌ Error creating main chart: {e}")

    # Bigger timeframe context chart
    st.markdown("### 🧭 Bigger Timeframe Context")
    try:
        big_chart = st.session_state.chart_renderer.create_simple_chart(
            data=data_big,
            patterns=None,
            title=f"{instrument}",
            height=config['chart_height'],
            max_candles=config['max_candles']
        )
        if struct_segments:
            st.session_state.chart_renderer.add_structural_segments(big_chart, data_big, struct_segments)
        st.plotly_chart(big_chart, use_container_width=True, key=f"context_chart_{instrument}_{used_months}_{len(data_big)}")
    except Exception as e:
        st.error(f"❌ Error creating context chart: {e}")

    # List all segments grouped by pattern
    if struct_segments:
        st.markdown("### 🧩 Detected Segments")
        try:
            df = pd.DataFrame(struct_segments)
            if not df.empty:
                df['pattern'] = df['pattern_type'].str.replace('_', ' ').str.title()
                df = df.sort_values(['pattern', 'confidence'], ascending=[True, False])
                # Pretty confidence
                df['confidence'] = (df['confidence'].astype(float) * 100).round(1)
                # Select and rename columns for display
                out = df[['pattern', 'status', 'confidence', 'start_dt', 'end_dt', 'start_idx', 'end_idx']]
                out = out.rename(columns={'confidence': 'confidence %'})
                st.dataframe(out, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not display segments: {e}")

    # Per-pattern mini charts for each detection
    if struct_segments:
        st.markdown("### 📌 Pattern Windows")
        show_all = st.checkbox("Show all detections (may be slow)", value=False, key=f"show_all_{instrument}")
        # group by pattern
        by_pat: Dict[str, List[Dict[str, Any]]] = {}
        for s in struct_segments:
            by_pat.setdefault(s['pattern_type'], []).append(s)
        for pat, segs in by_pat.items():
            st.subheader(pat.replace('_',' ').title())
            segs = sorted(segs, key=lambda x: x.get('confidence', 0.0), reverse=True)
            limit = len(segs) if show_all else min(5, len(segs))
            cols = st.columns(1 if limit <= 1 else 2)
            for i, seg in enumerate(segs[:limit]):
                try:
                    s_idx = max(0, seg['start_idx'] - 10)
                    e_idx = min(len(data_big) - 1, seg['end_idx'] + 10)
                    window = data_big.iloc[s_idx:e_idx+1]
                    title = f"{pat.replace('_',' ').title()} | {seg.get('status','')} | {seg.get('confidence',0.0):.0%}"
                    chart = st.session_state.chart_renderer.create_simple_chart(
                        data=window,
                        patterns=None,
                        title=title,
                        height=350,
                        max_candles=100000
                    )
                    # Highlight only this segment in the window for clarity
                    st.session_state.chart_renderer.add_structural_segments(
                        chart,
                        window,
                        [{**seg, 'start_idx': seg['start_idx']-s_idx, 'end_idx': seg['end_idx']-s_idx}]
                    )
                    cols[i % len(cols)].plotly_chart(chart, use_container_width=True, key=f"mini_chart_{pat}_{seg.get('start_idx')}_{seg.get('end_idx')}")
                except Exception as e:
                    st.warning(f"Mini chart error: {e}")

    # Focused zoom on top segment
    if struct_segments:
        top = struct_segments[0]
        st.markdown(f"### 🔍 Top Pattern Region: {top.get('pattern_type','').replace('_',' ').title()} ({top.get('confidence',0.0):.0%})")
        try:
            s_idx = max(0, top['start_idx'] - 10)
            e_idx = min(len(data_big) - 1, top['end_idx'] + 10)
            window = data_big.iloc[s_idx:e_idx+1]
            zoom_chart = st.session_state.chart_renderer.create_simple_chart(
                data=window,
                patterns=None,
                title=f"{instrument} - Focused Region",
                height=config['chart_height'],
                max_candles=config['max_candles']
            )
            st.plotly_chart(zoom_chart, use_container_width=True, key=f"zoom_chart_{instrument}_{top.get('start_idx')}_{top.get('end_idx')}")
        except Exception as e:
            st.warning(f"Zoom error: {e}")

    # Basic data summary
    try:
        info = st.session_state.data_loader.get_data_info(instrument)
        if info:
            with st.expander("📄 Data Summary"):
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
    """(Deprecated in simplified UI)"""
    pass


if __name__ == "__main__":
    main()
