#!/usr/bin/env python3
"""
Optimized Stock Pattern Detector - Streamlit Web Interface

High-performance web application for professional stock pattern analysis.
Optimized for speed, responsiveness, and user experience.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, date, timedelta
from typing import List, Dict, Any, Optional, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import concurrent.futures
from functools import lru_cache

# Configure Streamlit for optimal performance
st.set_page_config(
    page_title="Stock Pattern Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "# Stock Pattern Detector\nOptimized for professional pattern analysis"
    }
)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import application modules with error handling
try:
    from data.loader import CSVLoader
    from data.aggregator import DataAggregator
    from patterns.double_top import DoubleTopDetector
    from patterns.double_bottom import DoubleBottomDetector
    from patterns.head_and_shoulders import ImprovedHeadAndShouldersDetector
    from patterns.inverse_head_and_shoulders import InverseHeadAndShouldersDetector
    from patterns.base import PatternResult
    from visualization.charts import ChartRenderer
    from utils.logging_config import get_logger
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.stop()

# Initialize logger
logger = get_logger(__name__)

# Custom CSS for modern, responsive UI
st.markdown("""
<style>
    /* Global styles */
    .main > div {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .app-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .app-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.9;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Cards and containers */
    .metric-card {
        background: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin: 1rem 0;
    }
    
    .pattern-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Form elements */
    .stSelectbox > label, .stMultiSelect > label, .stSlider > label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Loading indicator */
    .loading-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 2rem;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .app-header h1 {
            font-size: 2rem;
        }
        .app-header p {
            font-size: 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Pattern detector configuration
@st.cache_resource
def get_pattern_detectors():
    """Initialize and cache pattern detectors for optimal performance."""
    return {
        'Head & Shoulders': ImprovedHeadAndShouldersDetector(),
        'Inverse Head & Shoulders': InverseHeadAndShouldersDetector(),
        'Double Top': DoubleTopDetector(),
        'Double Bottom': DoubleBottomDetector()
    }

PATTERN_DETECTORS = get_pattern_detectors()

@st.cache_resource
def initialize_components():
    """Initialize and cache core application components."""
    return {
        'data_loader': CSVLoader(),
        'data_aggregator': DataAggregator(),
        'chart_renderer': ChartRenderer()
    }

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_instrument_data(instrument: str, start_date: date, end_date: date) -> Optional[pd.DataFrame]:
    """Load and cache instrument data with automatic cache invalidation."""
    try:
        components = initialize_components()
        return components['data_loader'].load_data(instrument, start_date, end_date)
    except Exception as e:
        logger.error(f"Error loading data for {instrument}: {e}")
        return None

@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_available_instruments() -> List[str]:
    """Get and cache available instruments list."""
    try:
        components = initialize_components()
        return components['data_loader'].get_available_instruments()
    except Exception as e:
        logger.error(f"Error getting instruments: {e}")
        return []

def detect_patterns_parallel(data: pd.DataFrame, selected_patterns: List[str], 
                           confidence_threshold: float) -> List[PatternResult]:
    """Detect patterns using parallel processing for better performance."""
    results = []
    
    # Use ThreadPoolExecutor for I/O bound pattern detection
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        future_to_pattern = {}
        
        for pattern_name in selected_patterns:
            if pattern_name in PATTERN_DETECTORS:
                detector = PATTERN_DETECTORS[pattern_name]
                future = executor.submit(detector.detect, data)
                future_to_pattern[future] = pattern_name
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_pattern):
            pattern_name = future_to_pattern[future]
            try:
                pattern_results = future.result(timeout=30)  # 30 second timeout
                if pattern_results:
                    # Filter by confidence threshold
                    filtered_results = [
                        result for result in pattern_results 
                        if result.confidence >= confidence_threshold
                    ]
                    results.extend(filtered_results)
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")
                st.warning(f"⚠️ Pattern detection failed for {pattern_name}: {str(e)}")
    
    return sorted(results, key=lambda x: x.datetime, reverse=True)

def render_app_header():
    """Render the application header with modern styling."""
    st.markdown("""
    <div class="app-header">
        <h1>📈 Stock Pattern Detector</h1>
        <p>Professional pattern analysis with real-time insights and advanced visualization</p>
    </div>
    """, unsafe_allow_html=True)

def create_sidebar_config() -> Optional[Dict[str, Any]]:
    """Create optimized sidebar configuration with enhanced UX."""
    st.sidebar.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <h3 style="color: white; margin: 0; text-align: center;">⚙️ Analysis Configuration</h3>
    </div>
    """, unsafe_allow_html=True)

    # Get available instruments with caching
    instruments_all = get_available_instruments()
    if not instruments_all:
        st.sidebar.error("❌ No stock data found. Please check your data directory.")
        return None

    # Stock selection
    st.sidebar.markdown("### 📊 Stock Selection")
    
    # Quick selection presets
    preset_selection = st.sidebar.selectbox(
        "Quick Select:",
        options=["Custom Selection", "Top 5 Stocks", "Random 3 Stocks"],
        help="Choose a preset or make a custom selection"
    )
    
    if preset_selection == "Top 5 Stocks":
        default_instruments = instruments_all[:5]
    elif preset_selection == "Random 3 Stocks":
        import random
        default_instruments = random.sample(instruments_all, min(3, len(instruments_all)))
    else:
        default_instruments = instruments_all[:3] if len(instruments_all) >= 3 else instruments_all

    selected_instruments = st.sidebar.multiselect(
        "Select stocks for analysis:",
        options=instruments_all,
        default=default_instruments,
        help="💡 Select multiple stocks for comprehensive analysis"
    )
    
    if not selected_instruments:
        st.sidebar.warning("⚠️ Please select at least one stock")
        return None

    # Primary instrument
    primary_instrument = st.sidebar.selectbox(
        "🎯 Primary stock (detailed view):",
        options=selected_instruments,
        help="This stock will be featured in the main analysis view"
    )

    # Date range configuration
    st.sidebar.markdown("### 📅 Time Period")
    
    # Get date ranges for validation
    sample_data = load_instrument_data(primary_instrument, date.today() - timedelta(days=365*5), date.today())
    if sample_data is not None and not sample_data.empty:
        available_start = sample_data.index.min().date()
        available_end = sample_data.index.max().date()
    else:
        available_start = date.today() - timedelta(days=365*2)
        available_end = date.today()

    # Time period selection
    period_option = st.sidebar.radio(
        "Select time period:",
        options=["📊 Last 1 Year", "📈 Last 2 Years", "🔍 Custom Range"],
        help="Choose the analysis time window"
    )
    
    if period_option == "📊 Last 1 Year":
        end_date = available_end
        start_date = max(available_start, end_date - timedelta(days=365))
    elif period_option == "📈 Last 2 Years":
        end_date = available_end
        start_date = max(available_start, end_date - timedelta(days=365*2))
    else:
        date_range = st.sidebar.date_input(
            "Custom date range:",
            value=(available_start, available_end),
            min_value=available_start,
            max_value=available_end,
            help="Select your custom analysis period"
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date, end_date = available_start, available_end

    # Pattern selection
    st.sidebar.markdown("### 🔍 Pattern Detection")
    selected_patterns = st.sidebar.multiselect(
        "Patterns to detect:",
        options=list(PATTERN_DETECTORS.keys()),
        default=list(PATTERN_DETECTORS.keys()),
        help="Select which patterns to analyze"
    )

    # Advanced settings
    with st.sidebar.expander("⚙️ Advanced Settings", expanded=False):
        confidence_threshold = st.slider(
            "Minimum confidence level:",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            help="Filter patterns below this confidence level"
        )
        
        show_volume = st.checkbox(
            "Include volume analysis",
            value=True,
            help="Show volume indicators in charts"
        )
        
        chart_height = st.slider(
            "Chart height (pixels):",
            min_value=400,
            max_value=1000,
            value=650,
            step=50
        )

    # Analysis button
    st.sidebar.markdown("---")
    run_analysis = st.sidebar.button(
        "🚀 Start Analysis",
        type="primary",
        use_container_width=True,
        help="Begin pattern detection analysis"
    )

    if not run_analysis:
        return None

    return {
        'instruments': selected_instruments,
        'primary_instrument': primary_instrument,
        'start_date': start_date,
        'end_date': end_date,
        'selected_patterns': selected_patterns,
        'confidence_threshold': confidence_threshold,
        'show_volume': show_volume,
        'chart_height': chart_height
    }

def display_analysis_results(config: Dict[str, Any]):
    """Display comprehensive analysis results with optimized performance."""
    
    # Create progress container
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    try:
        # Load data for primary instrument
        status_text.text("📊 Loading market data...")
        progress_bar.progress(0.2)
        
        primary_data = load_instrument_data(
            config['primary_instrument'], 
            config['start_date'], 
            config['end_date']
        )
        
        if primary_data is None or primary_data.empty:
            st.error(f"❌ No data available for {config['primary_instrument']} in the selected period")
            return
        
        progress_bar.progress(0.4)
        status_text.text("🔍 Detecting patterns...")
        
        # Detect patterns
        pattern_results = detect_patterns_parallel(
            primary_data,
            config['selected_patterns'],
            config['confidence_threshold']
        )
        
        progress_bar.progress(0.7)
        status_text.text("📈 Rendering visualizations...")
        
        # Clear progress
        progress_container.empty()
        
        # Display results
        display_pattern_summary(pattern_results, config['primary_instrument'])
        display_interactive_chart(primary_data, pattern_results, config)
        display_pattern_details(pattern_results)
        
        # Multi-instrument comparison if multiple selected
        if len(config['instruments']) > 1:
            display_multi_instrument_analysis(config)
            
    except Exception as e:
        progress_container.empty()
        st.error(f"❌ Analysis failed: {str(e)}")
        logger.error(f"Analysis error: {e}", exc_info=True)

def display_pattern_summary(results: List[PatternResult], instrument: str):
    """Display pattern detection summary with metrics."""
    st.markdown("## 📊 Pattern Detection Summary")
    
    if not results:
        st.info("🔍 No patterns detected with the current confidence threshold. Try lowering the threshold or adjusting the time period.")
        return
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Patterns",
            value=len(results),
            help="Total number of patterns detected"
        )
    
    with col2:
        avg_confidence = np.mean([r.confidence for r in results]) if results else 0
        st.metric(
            label="Avg Confidence",
            value=f"{avg_confidence:.2%}",
            help="Average confidence level of detected patterns"
        )
    
    with col3:
        pattern_types = len(set(r.pattern_type for r in results))
        st.metric(
            label="Pattern Types",
            value=pattern_types,
            help="Number of different pattern types found"
        )
    
    with col4:
        recent_patterns = len([r for r in results if (datetime.now().date() - r.datetime.date()).days <= 30])
        st.metric(
            label="Recent (30d)",
            value=recent_patterns,
            help="Patterns detected in the last 30 days"
        )
    
    # Pattern type breakdown
    if results:
        pattern_counts = {}
        for result in results:
            pattern_counts[result.pattern_type] = pattern_counts.get(result.pattern_type, 0) + 1
        
        st.markdown("### Pattern Distribution")
        
        # Create a horizontal bar chart for pattern distribution
        fig = go.Figure(data=[
            go.Bar(
                y=list(pattern_counts.keys()),
                x=list(pattern_counts.values()),
                orientation='h',
                marker_color='rgba(102, 126, 234, 0.8)',
                text=list(pattern_counts.values()),
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Pattern Type Distribution",
            xaxis_title="Count",
            yaxis_title="Pattern Type",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_interactive_chart(data: pd.DataFrame, patterns: List[PatternResult], config: Dict[str, Any]):
    """Display interactive candlestick chart with pattern annotations."""
    st.markdown("## 📈 Interactive Price Chart")
    
    try:
        components = initialize_components()
        chart_renderer = components['chart_renderer']
        
        # Create chart with patterns
        fig = chart_renderer.create_pattern_chart(
            data=data,
            patterns=patterns,
            title=f"{config['primary_instrument']} - Pattern Analysis",
            height=config['chart_height'],
            show_volume=config['show_volume']
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("❌ Failed to create chart")
            
    except Exception as e:
        st.error(f"❌ Chart rendering failed: {str(e)}")
        logger.error(f"Chart error: {e}", exc_info=True)

def display_pattern_details(results: List[PatternResult]):
    """Display detailed pattern information in an organized table."""
    if not results:
        return
        
    st.markdown("## 🔍 Pattern Details")
    
    # Convert results to DataFrame for better display
    pattern_data = []
    for result in results:
        pattern_data.append({
            'Date': result.datetime.strftime('%Y-%m-%d'),
            'Time': result.datetime.strftime('%H:%M'),
            'Pattern': result.pattern_type,
            'Confidence': f"{result.confidence:.1%}",
            'Timeframe': result.timeframe
        })
    
    df = pd.DataFrame(pattern_data)
    
    # Add filtering options
    col1, col2 = st.columns(2)
    with col1:
        pattern_filter = st.selectbox(
            "Filter by pattern type:",
            options=['All'] + df['Pattern'].unique().tolist(),
            key="pattern_filter"
        )
    
    with col2:
        min_confidence = st.slider(
            "Minimum confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.05,
            key="detail_confidence_filter"
        )
    
    # Apply filters
    filtered_df = df.copy()
    if pattern_filter != 'All':
        filtered_df = filtered_df[filtered_df['Pattern'] == pattern_filter]
    
    # Convert confidence back to float for filtering
    filtered_df['Confidence_Float'] = filtered_df['Confidence'].str.rstrip('%').astype(float) / 100
    filtered_df = filtered_df[filtered_df['Confidence_Float'] >= min_confidence]
    filtered_df = filtered_df.drop('Confidence_Float', axis=1)
    
    # Display filtered results
    if not filtered_df.empty:
        st.dataframe(
            filtered_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Date": st.column_config.DateColumn("Date"),
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    help="Pattern confidence level",
                    min_value=0,
                    max_value=1,
                ),
            }
        )
    else:
        st.info("🔍 No patterns match the current filters")

def display_multi_instrument_analysis(config: Dict[str, Any]):
    """Display comparative analysis for multiple instruments."""
    st.markdown("## 🔄 Multi-Instrument Comparison")
    
    with st.spinner("Loading comparison data..."):
        comparison_data = {}
        comparison_patterns = {}
        
        for instrument in config['instruments']:
            try:
                # Load data
                data = load_instrument_data(instrument, config['start_date'], config['end_date'])
                if data is not None and not data.empty:
                    comparison_data[instrument] = data
                    
                    # Detect patterns
                    patterns = detect_patterns_parallel(
                        data,
                        config['selected_patterns'],
                        config['confidence_threshold']
                    )
                    comparison_patterns[instrument] = patterns
                    
            except Exception as e:
                st.warning(f"⚠️ Could not load data for {instrument}: {str(e)}")
    
    if comparison_data:
        # Create comparison metrics
        cols = st.columns(len(comparison_data))
        
        for idx, (instrument, patterns) in enumerate(comparison_patterns.items()):
            with cols[idx]:
                st.markdown(f"### {instrument}")
                st.metric("Patterns Found", len(patterns))
                if patterns:
                    avg_conf = np.mean([p.confidence for p in patterns])
                    st.metric("Avg Confidence", f"{avg_conf:.1%}")
                    
                    # Most recent pattern
                    recent_pattern = max(patterns, key=lambda x: x.datetime)
                    st.info(f"Latest: {recent_pattern.pattern_type}")

def show_welcome_screen():
    """Display welcome screen when no configuration is provided."""
    st.markdown("## 👋 Welcome to Stock Pattern Detector")
    
    st.markdown("""
    <div class="info-card">
        <h4>🚀 Getting Started</h4>
        <p>Configure your analysis settings in the sidebar to begin:</p>
        <ul>
            <li>📊 Select one or more stocks</li>
            <li>📅 Choose your analysis time period</li>
            <li>🔍 Pick patterns to detect</li>
            <li>⚙️ Adjust advanced settings (optional)</li>
        </ul>
        <p>Then click <strong>"Start Analysis"</strong> to begin pattern detection!</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show available stocks preview
    instruments = get_available_instruments()
    if instruments:
        st.markdown("### 📈 Available Stocks")
        
        # Display in a grid format
        cols = st.columns(4)
        for idx, instrument in enumerate(instruments[:20]):  # Show first 20
            with cols[idx % 4]:
                st.code(instrument)
        
        if len(instruments) > 20:
            st.info(f"... and {len(instruments) - 20} more stocks available")
    
    # Feature highlights
    st.markdown("### ✨ Key Features")
    
    feature_cols = st.columns(3)
    
    with feature_cols[0]:
        st.markdown("""
        **🔍 Pattern Detection**
        - Head & Shoulders
        - Inverse Head & Shoulders  
        - Double Top
        - Double Bottom
        """)
    
    with feature_cols[1]:
        st.markdown("""
        **📊 Advanced Analytics**
        - Real-time visualization
        - Confidence scoring
        - Multi-timeframe analysis
        - Parallel processing
        """)
    
    with feature_cols[2]:
        st.markdown("""
        **💡 Smart Features**
        - Automatic caching
        - Responsive design
        - Export capabilities
        - Performance optimized
        """)

def main():
    """Main application entry point with optimized flow."""
    try:
        # Render header
        render_app_header()
        
        # Create sidebar configuration
        config = create_sidebar_config()
        
        # Display content based on configuration
        if config:
            display_analysis_results(config)
        else:
            show_welcome_screen()
            
    except Exception as e:
        st.error(f"❌ Application error: {str(e)}")
        logger.error(f"Main application error: {e}", exc_info=True)
        
        # Display error details for debugging
        with st.expander("🔧 Error Details (for debugging)"):
            st.code(str(e))

if __name__ == "__main__":
    main()
