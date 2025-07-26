"""
Chart rendering module for stock pattern detector.

This module provides functionality to create interactive candlestick charts
using Plotly with responsive design and pattern highlighting capabilities.
"""

import logging
from typing import List, Optional, Dict, Any, Tuple, Callable
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from patterns.base import PatternResult

# Configure logging
logger = logging.getLogger(__name__)


class ChartRenderer:
    """
    Handles creation and management of interactive candlestick charts.
    
    Provides responsive chart configuration, interactive features,
    and pattern highlighting capabilities.
    """
    
    def __init__(self):
        """Initialize the chart renderer with default configuration."""
        self.default_config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 'hoverClosestCartesian',
                'hoverCompareCartesian', 'toggleSpikelines'
            ],
            'responsive': True
        }
        
        # Pattern color mapping
        self.pattern_colors = {
            'dragonfly_doji': '#FF6B6B',      # Red
            'hammer': '#4ECDC4',              # Teal
            'rising_window': '#45B7D1',       # Blue
            'evening_star': '#96CEB4',        # Green
            'three_white_soldiers': '#FFEAA7' # Yellow
        }
        
        # Default pattern color for unknown patterns
        self.default_pattern_color = '#DDA0DD'  # Plum
        
        # Initialize pattern highlighter for advanced features
        from .pattern_highlighter import PatternHighlighter
        self.pattern_highlighter = PatternHighlighter()
    
    def create_candlestick_chart(
        self, 
        data: pd.DataFrame, 
        title: str = "Stock Price Chart",
        height: int = 600,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create an interactive candlestick chart with optional volume subplot.
        
        Args:
            data: OHLCV DataFrame with datetime index
            title: Chart title
            height: Chart height in pixels
            show_volume: Whether to show volume subplot
            
        Returns:
            Plotly Figure object with candlestick chart
        """
        try:
            if data is None or data.empty:
                logger.warning("Empty data provided to chart renderer")
                return self._create_empty_chart(title, height)
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            if not all(col in data.columns for col in required_cols):
                logger.error(f"Missing required OHLC columns in data")
                return self._create_empty_chart(title, height)
            
            # Create subplots if volume is requested
            if show_volume and 'volume' in data.columns:
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.03,
                    subplot_titles=(title, 'Volume'),
                    row_width=[0.7, 0.3]
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
                increasing_line_color='#00C851',  # Green for bullish
                decreasing_line_color='#FF4444',  # Red for bearish
                increasing_fillcolor='#00C851',
                decreasing_fillcolor='#FF4444',
                line=dict(width=1),
                hovertext=[
                    f'Open: ${row["open"]:.2f}<br>High: ${row["high"]:.2f}<br>Low: ${row["low"]:.2f}<br>Close: ${row["close"]:.2f}'
                    for _, row in data.iterrows()
                ]
            )
            
            if volume_row:
                fig.add_trace(candlestick, row=1, col=1)
            else:
                fig.add_trace(candlestick)
            
            # Add volume bars if requested
            if volume_row and 'volume' in data.columns:
                # Color volume bars based on price movement
                colors = []
                for i in range(len(data)):
                    if data.iloc[i]['close'] >= data.iloc[i]['open']:
                        colors.append('#00C851')  # Green for up days
                    else:
                        colors.append('#FF4444')  # Red for down days
                
                volume_trace = go.Bar(
                    x=data['datetime'] if 'datetime' in data.columns else data.index,
                    y=data['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7,
                    hovertext=[
                        f'Volume: {vol:,.0f}'
                        for vol in data['volume']
                    ]
                )
                
                fig.add_trace(volume_trace, row=volume_row, col=1)
            
            # Configure layout for responsiveness and interactivity
            self._configure_chart_layout(fig, height, show_volume)
            
            logger.info(f"Created candlestick chart with {len(data)} data points")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating candlestick chart: {e}")
            return self._create_empty_chart(title, height)
    
    def highlight_patterns(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        data: pd.DataFrame,
        advanced_highlighting: bool = True
    ) -> go.Figure:
        """
        Add pattern markers and annotations to the chart.
        
        Args:
            fig: Existing Plotly figure
            patterns: List of detected patterns
            data: Original OHLCV data for reference
            advanced_highlighting: Whether to use advanced highlighting features
            
        Returns:
            Updated figure with pattern highlights
        """
        try:
            if not patterns:
                logger.info("No patterns to highlight")
                return fig
            
            logger.info(f"Highlighting {len(patterns)} patterns on chart")
            
            if advanced_highlighting:
                # Use advanced pattern highlighter
                fig = self.pattern_highlighter.add_pattern_overlays(
                    fig, patterns, data, enable_click_navigation=True
                )
                fig = self.pattern_highlighter.add_confidence_indicators(
                    fig, patterns, data
                )
            else:
                # Use basic highlighting
                fig = self._add_basic_pattern_markers(fig, patterns, data)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error highlighting patterns: {e}")
            return fig
    
    def _add_basic_pattern_markers(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        data: pd.DataFrame
    ) -> go.Figure:
        """
        Add basic pattern markers (fallback method).
        
        Args:
            fig: Existing Plotly figure
            patterns: List of detected patterns
            data: Original OHLCV data for reference
            
        Returns:
            Updated figure with basic pattern highlights
        """
        try:
            # Group patterns by type for legend organization
            pattern_groups = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type
                if pattern_type not in pattern_groups:
                    pattern_groups[pattern_type] = []
                pattern_groups[pattern_type].append(pattern)
            
            # Add markers for each pattern type
            for pattern_type, pattern_list in pattern_groups.items():
                color = self.pattern_colors.get(pattern_type, self.default_pattern_color)
                
                # Extract coordinates for markers
                x_coords = []
                y_coords = []
                hover_texts = []
                
                for pattern in pattern_list:
                    try:
                        # Find the corresponding data point
                        if 'datetime' in data.columns:
                            mask = data['datetime'] == pattern.datetime
                        else:
                            # Use index if datetime column not available
                            mask = data.index == pattern.candle_index
                        
                        if mask.any():
                            row = data[mask].iloc[0]
                            x_coords.append(pattern.datetime)
                            # Position marker above the high of the candle
                            y_coords.append(row['high'] * 1.02)
                            
                            hover_text = (
                                f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>"
                                f"Confidence: {pattern.confidence:.1%}<br>"
                                f"Time: {pattern.datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                                f"Timeframe: {pattern.timeframe}"
                            )
                            hover_texts.append(hover_text)
                            
                    except Exception as e:
                        logger.warning(f"Error processing pattern {pattern.pattern_type}: {e}")
                        continue
                
                if x_coords:
                    # Add scatter trace for pattern markers
                    pattern_trace = go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        name=pattern_type.replace('_', ' ').title(),
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='%{text}<extra></extra>',
                        text=hover_texts,
                        showlegend=True
                    )
                    
                    fig.add_trace(pattern_trace)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error adding basic pattern markers: {e}")
            return fig
    
    def add_interactivity(self, fig: go.Figure) -> go.Figure:
        """
        Add interactive features like zoom, pan, and hover tooltips.
        
        Args:
            fig: Plotly figure to enhance
            
        Returns:
            Enhanced figure with interactive features
        """
        try:
            # Update layout for better interactivity
            fig.update_layout(
                # Enable crossfilter interactions
                hovermode='x unified',
                
                # Configure zoom and pan
                dragmode='zoom',
                
                # Add range selector buttons
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1, label="1D", step="day", stepmode="backward"),
                            dict(count=7, label="7D", step="day", stepmode="backward"),
                            dict(count=30, label="30D", step="day", stepmode="backward"),
                            dict(count=90, label="3M", step="day", stepmode="backward"),
                            dict(step="all", label="All")
                        ])
                    ),
                    rangeslider=dict(visible=False),  # Disable range slider for cleaner look
                    type="date"
                )
            )
            
            # Add custom config for better user experience
            fig.update_layout(
                # Improve legend
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1
                )
            )
            
            logger.info("Added interactive features to chart")
            return fig
            
        except Exception as e:
            logger.error(f"Error adding interactivity: {e}")
            return fig
    
    def create_pattern_summary_chart(self, patterns: List[PatternResult]) -> go.Figure:
        """
        Create a summary chart showing pattern distribution.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Bar chart showing pattern counts
        """
        try:
            if not patterns:
                return self._create_empty_chart("No Patterns Detected", 400)
            
            # Count patterns by type
            pattern_counts = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type.replace('_', ' ').title()
                pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=list(pattern_counts.keys()),
                    y=list(pattern_counts.values()),
                    marker_color=[
                        self.pattern_colors.get(k.lower().replace(' ', '_'), self.default_pattern_color)
                        for k in pattern_counts.keys()
                    ],
                    hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title="Detected Patterns Summary",
                xaxis_title="Pattern Type",
                yaxis_title="Count",
                height=400,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern summary chart: {e}")
            return self._create_empty_chart("Error Creating Summary", 400)
    
    def _configure_chart_layout(self, fig: go.Figure, height: int, show_volume: bool) -> None:
        """
        Configure chart layout for responsiveness and aesthetics.
        
        Args:
            fig: Figure to configure
            height: Chart height
            show_volume: Whether volume subplot is shown
        """
        try:
            # Base layout configuration
            layout_config = {
                'height': height,
                'margin': dict(l=50, r=50, t=50, b=50),
                'paper_bgcolor': 'white',
                'plot_bgcolor': 'white',
                'font': dict(size=12),
                
                # Grid configuration
                'xaxis': dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(128,128,128,0.5)'
                ),
                'yaxis': dict(
                    title='Price ($)',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)',
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(128,128,128,0.5)'
                )
            }
            
            # Configure volume subplot if present
            if show_volume:
                layout_config['yaxis2'] = dict(
                    title='Volume',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128,128,128,0.2)'
                )
            
            fig.update_layout(**layout_config)
            
        except Exception as e:
            logger.error(f"Error configuring chart layout: {e}")
    
    def _create_empty_chart(self, title: str, height: int) -> go.Figure:
        """
        Create an empty chart with error message.
        
        Args:
            title: Chart title
            height: Chart height
            
        Returns:
            Empty figure with message
        """
        fig = go.Figure()
        fig.update_layout(
            title=title,
            height=height,
            annotations=[
                dict(
                    text="No data available or error occurred",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=16, color="gray")
                )
            ]
        )
        return fig
    
    def get_pattern_color(self, pattern_type: str) -> str:
        """
        Get the color associated with a pattern type.
        
        Args:
            pattern_type: Type of pattern
            
        Returns:
            Hex color code
        """
        return self.pattern_colors.get(pattern_type, self.default_pattern_color)
    
    def update_chart_timeframe_focus(
        self, 
        fig: go.Figure, 
        center_datetime: datetime, 
        window_minutes: int = 60
    ) -> go.Figure:
        """
        Focus the chart on a specific datetime with a time window.
        
        Args:
            fig: Figure to update
            center_datetime: Datetime to center on
            window_minutes: Minutes to show before and after center
            
        Returns:
            Updated figure with focused time range
        """
        try:
            from datetime import timedelta
            
            start_time = center_datetime - timedelta(minutes=window_minutes // 2)
            end_time = center_datetime + timedelta(minutes=window_minutes // 2)
            
            fig.update_layout(
                xaxis=dict(
                    range=[start_time, end_time]
                )
            )
            
            logger.info(f"Focused chart on {center_datetime} with {window_minutes}min window")
            return fig
            
        except Exception as e:
            logger.error(f"Error focusing chart timeframe: {e}")
            return fig
    
    def create_pattern_detail_view(
        self, 
        data: pd.DataFrame, 
        pattern: PatternResult,
        context_candles: int = 10,
        title: str = None
    ) -> go.Figure:
        """
        Create a detailed view focused on a specific pattern.
        
        Args:
            data: OHLCV DataFrame
            pattern: Pattern to focus on
            context_candles: Number of candles to show around pattern
            title: Custom title for the chart
            
        Returns:
            Detailed chart focused on the pattern
        """
        try:
            if title is None:
                title = f"{pattern.pattern_type.replace('_', ' ').title()} Pattern Detail"
            
            # Create base chart
            fig = self.create_candlestick_chart(data, title=title, height=500, show_volume=False)
            
            # Add pattern detail overlay
            fig = self.pattern_highlighter.create_pattern_detail_overlay(
                fig, pattern, data, context_candles
            )
            
            # Focus on pattern timeframe
            fig = self.update_chart_timeframe_focus(
                fig, pattern.datetime, window_minutes=context_candles * 5
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern detail view: {e}")
            return self._create_empty_chart(title or "Pattern Detail", 500)
    
    def create_pattern_navigation_interface(
        self, 
        patterns: List[PatternResult]
    ) -> Dict[str, Any]:
        """
        Create navigation interface data for pattern browsing.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Dictionary with navigation interface data
        """
        try:
            if not patterns:
                return {'patterns': [], 'timeline': None, 'summary': None}
            
            # Sort patterns by datetime
            sorted_patterns = sorted(patterns, key=lambda p: p.datetime)
            
            # Create pattern summary data
            pattern_summary = {}
            for pattern in patterns:
                pattern_type = pattern.pattern_type.replace('_', ' ').title()
                if pattern_type not in pattern_summary:
                    pattern_summary[pattern_type] = {
                        'count': 0,
                        'avg_confidence': 0.0,
                        'color': self.get_pattern_color(pattern.pattern_type)
                    }
                pattern_summary[pattern_type]['count'] += 1
                pattern_summary[pattern_type]['avg_confidence'] += pattern.confidence
            
            # Calculate average confidence
            for pattern_type in pattern_summary:
                count = pattern_summary[pattern_type]['count']
                pattern_summary[pattern_type]['avg_confidence'] /= count
            
            # Create timeline chart
            timeline_fig = self.pattern_highlighter.create_pattern_timeline(patterns)
            
            # Create summary chart
            summary_fig = self.create_pattern_summary_chart(patterns)
            
            return {
                'patterns': [
                    {
                        'datetime': p.datetime,
                        'pattern_type': p.pattern_type.replace('_', ' ').title(),
                        'confidence': p.confidence,
                        'timeframe': p.timeframe,
                        'color': self.get_pattern_color(p.pattern_type),
                        'index': i
                    }
                    for i, p in enumerate(sorted_patterns)
                ],
                'timeline': timeline_fig,
                'summary': summary_fig,
                'pattern_summary': pattern_summary
            }
            
        except Exception as e:
            logger.error(f"Error creating navigation interface: {e}")
            return {'patterns': [], 'timeline': None, 'summary': None}
    
    def enable_pattern_click_navigation(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        callback_function: Optional[Callable] = None
    ) -> go.Figure:
        """
        Enable click-to-navigate functionality for patterns.
        
        Note: This prepares the chart for click navigation.
        Actual navigation requires frontend integration.
        
        Args:
            fig: Figure to enhance
            patterns: List of patterns
            callback_function: Optional callback for click events
            
        Returns:
            Enhanced figure with click navigation setup
        """
        try:
            # Add custom data attributes for click handling
            for trace in fig.data:
                if hasattr(trace, 'name') and any(
                    pattern_type.replace('_', ' ').title() in str(trace.name) 
                    for pattern_type in [p.pattern_type for p in patterns]
                ):
                    # Add custom data for click handling
                    if hasattr(trace, 'customdata'):
                        trace.customdata = [
                            {'pattern_index': i, 'navigable': True}
                            for i in range(len(trace.x))
                        ]
            
            # Update layout for click events
            fig.update_layout(
                clickmode='event+select',
                annotations=[
                    dict(
                        text="Click on pattern markers to navigate",
                        xref="paper", yref="paper",
                        x=0.02, y=0.98,
                        xanchor='left', yanchor='top',
                        showarrow=False,
                        font=dict(size=10, color="gray"),
                        bgcolor="rgba(255,255,255,0.8)"
                    )
                ]
            )
            
            logger.info("Enabled click navigation interface")
            return fig
            
        except Exception as e:
            logger.error(f"Error enabling click navigation: {e}")
            return fig