"""
Simple and effective chart rendering for stock pattern detection.

Focused on clean, readable candlestick charts with clear pattern indicators.
"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from patterns.base import PatternResult

logger = logging.getLogger(__name__)


class ChartRenderer:
    """
    Simple chart renderer focused on clarity and performance.
    
    Creates clean candlestick charts with distinctive pattern markers.
    """
    
    def __init__(self):
        """Initialize with clean, professional settings."""
        self.config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'responsive': True
        }
        
        # Clean color scheme for patterns
        self.pattern_styles = {
            'dragonfly_doji': {'color': '#FF4444', 'symbol': 'triangle-down', 'size': 12},
            'hammer': {'color': '#00CC88', 'symbol': 'diamond', 'size': 11},
            'rising_window': {'color': '#4488FF', 'symbol': 'arrow-up', 'size': 13},
            'evening_star': {'color': '#FF8800', 'symbol': 'star', 'size': 12},
            'three_white_soldiers': {'color': '#8844FF', 'symbol': 'triangle-up', 'size': 11}
        }
        
        self.default_pattern_style = {'color': '#888888', 'symbol': 'circle', 'size': 8}
    
    def create_simple_chart(
        self, 
        data: pd.DataFrame, 
        patterns: List[PatternResult] = None,
        title: str = "Stock Chart",
        height: int = 500,
        max_candles: int = 300
    ) -> go.Figure:
        """
        Create a clean, simple candlestick chart with pattern markers.
        
        Args:
            data: OHLCV DataFrame with datetime index
            patterns: List of detected patterns to highlight
            title: Chart title
            height: Chart height in pixels
            max_candles: Maximum number of candles to display
            
        Returns:
            Clean Plotly candlestick chart
        """
        try:
            if data is None or data.empty:
                return self._create_empty_chart(title, height)
            
            # Ensure max_candles is reasonable to prevent performance issues
            max_candles = min(max_candles, 1000)  # Cap at 1000 candles
            
            # Limit data for better performance and readability
            if len(data) > max_candles:
                data = data.tail(max_candles).copy()
                # Adjust pattern indices if needed
                if patterns:
                    offset = len(data) - max_candles
                    adjusted_patterns = []
                    for p in patterns:
                        if hasattr(p, 'candle_index') and p.candle_index >= offset:
                            # Create new pattern with adjusted index
                            adjusted_pattern = PatternResult(
                                pattern_type=p.pattern_type,
                                confidence=p.confidence,
                                datetime=p.datetime,
                                timeframe=p.timeframe,
                                candle_index=p.candle_index - offset
                            )
                            adjusted_patterns.append(adjusted_pattern)
                    patterns = adjusted_patterns
            
            # Create figure
            fig = go.Figure()
            
            # Add candlestick trace with better visibility
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#2E7D32',    # Darker green for better visibility
                decreasing_line_color='#C62828',    # Darker red for better visibility
                increasing_fillcolor='rgba(76, 175, 80, 0.7)',   # Material green
                decreasing_fillcolor='rgba(244, 67, 54, 0.7)',   # Material red
                line=dict(width=1.5)  # Slightly thicker lines
            ))
            
            # Add pattern markers
            if patterns:
                self._add_simple_pattern_markers(fig, patterns, data)
            
            # Clean layout with better grid visibility
            fig.update_layout(
                title=dict(
                    text=title,
                    x=0.5,
                    font=dict(size=16, color='#2E2E2E')
                ),
                height=height,
                margin=dict(l=60, r=60, t=80, b=60),
                plot_bgcolor='#FAFAFA',  # Light gray background for better contrast
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color='#2E2E2E'),
                showlegend=False,
                xaxis=dict(
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',  # More visible grid
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(128, 128, 128, 0.5)',
                    rangeslider=dict(visible=False),
                    tickfont=dict(size=10)
                ),
                yaxis=dict(
                    title="Price",
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='rgba(128, 128, 128, 0.3)',  # More visible grid
                    showline=True,
                    linewidth=1,
                    linecolor='rgba(128, 128, 128, 0.5)',
                    tickfont=dict(size=10)
                ),
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple chart: {e}")
            return self._create_empty_chart(title, height)
    
    def _add_simple_pattern_markers(self, fig: go.Figure, patterns: List[PatternResult], data: pd.DataFrame):
        """Add clean pattern markers to the chart."""
        try:
            # Limit patterns for performance
            limited_patterns = patterns[:10] if patterns else []  # Max 10 patterns per chart
            
            for pattern in limited_patterns:
                if not hasattr(pattern, 'candle_index') or pattern.candle_index >= len(data):
                    continue
                
                candle_time = data.index[pattern.candle_index]
                candle_data = data.iloc[pattern.candle_index]
                
                # Get pattern style
                style = self.pattern_styles.get(
                    pattern.pattern_type, 
                    self.default_pattern_style
                )
                
                # Position marker above high with better spacing
                marker_y = candle_data['high'] * 1.025
                
                # Confidence-based size and opacity
                base_size = style['size']
                confidence_size = base_size + int(pattern.confidence * 4)  # 0-4 extra pixels
                opacity = 0.8 + (pattern.confidence * 0.2)  # 0.8 to 1.0
                
                fig.add_trace(go.Scatter(
                    x=[candle_time],
                    y=[marker_y],
                    mode='markers',
                    marker=dict(
                        symbol=style['symbol'],
                        size=confidence_size,
                        color=style['color'],
                        opacity=opacity,
                        line=dict(width=2, color='white')  # Better outline
                    ),
                    hovertemplate=(
                        f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>"
                        f"Confidence: {pattern.confidence:.1%}<br>"
                        f"Time: {pattern.datetime.strftime('%Y-%m-%d %H:%M')}<br>"
                        "<extra></extra>"
                    ),
                    showlegend=False
                ))
                
        except Exception as e:
            logger.error(f"Error adding pattern markers: {e}")
    
    def create_compact_chart(
        self,
        data: pd.DataFrame,
        patterns: List[PatternResult] = None,
        title: str = "Quick View",
        height: int = 350
    ) -> go.Figure:
        """
        Create a compact chart for overview display.
        
        Args:
            data: OHLCV DataFrame
            patterns: List of patterns
            title: Chart title
            height: Chart height
            
        Returns:
            Compact chart figure
        """
        # Use last 100 candles for compact view
        return self.create_simple_chart(
            data=data,
            patterns=patterns,
            title=title,
            height=height,
            max_candles=100
        )
    
    def create_pattern_detail_chart(
        self,
        data: pd.DataFrame,
        pattern: PatternResult,
        context_candles: int = 20
    ) -> go.Figure:
        """
        Create a focused chart showing a specific pattern with context.
        
        Args:
            data: OHLCV DataFrame
            pattern: The pattern to focus on
            context_candles: Number of candles to show around the pattern
            
        Returns:
            Focused pattern chart
        """
        try:
            # Validate inputs
            if data is None or data.empty:
                logger.warning("Empty data provided to create_pattern_detail_chart")
                return self._create_empty_chart("Pattern Detail", 400)
                
            if not hasattr(pattern, 'candle_index'):
                logger.warning("Pattern missing candle_index attribute")
                return self._create_empty_chart("Pattern Detail", 400)
                
            if pattern.candle_index < 0 or pattern.candle_index >= len(data):
                logger.warning(f"Pattern candle_index {pattern.candle_index} out of range for data length {len(data)}")
                return self._create_empty_chart("Pattern Detail", 400)
            
            # Limit context candles for performance
            context_candles = min(context_candles, 50)  # Max 50 candles
            
            # Extract context around pattern
            start_idx = max(0, pattern.candle_index - context_candles)
            end_idx = min(len(data), pattern.candle_index + context_candles + 1)
            context_data = data.iloc[start_idx:end_idx].copy()
            
            if context_data.empty:
                logger.warning("Context data is empty")
                return self._create_empty_chart("Pattern Detail", 400)
            
            # Adjust pattern index for context data
            adjusted_pattern = PatternResult(
                pattern_type=pattern.pattern_type,
                confidence=pattern.confidence,
                datetime=pattern.datetime,
                timeframe=pattern.timeframe,
                candle_index=pattern.candle_index - start_idx
            )
            
            title = f"{pattern.pattern_type.replace('_', ' ').title()} - {pattern.confidence:.1%} Confidence"
            
            return self.create_simple_chart(
                data=context_data,
                patterns=[adjusted_pattern],
                title=title,
                height=400,
                max_candles=len(context_data)
            )
            
        except Exception as e:
            logger.error(f"Error creating pattern detail chart: {e}", exc_info=True)
            return self._create_empty_chart("Pattern Detail", 400)
    
    def _create_empty_chart(self, title: str, height: int) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            height=height,
            showlegend=False,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )
        return fig
