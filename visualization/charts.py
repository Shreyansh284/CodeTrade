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
    
    def _get_x_values(self, data: pd.DataFrame):
        """Return appropriate x-axis values (datetime column if present)."""
        try:
            if isinstance(data.index, pd.DatetimeIndex):
                return data.index
            if 'datetime' in data.columns:
                return data['datetime']
        except Exception:
            pass
        return data.index
    
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
            data: OHLCV DataFrame with datetime index or 'datetime' column
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
            
            # Ensure datetime column exists for x-axis
            if 'datetime' in data.columns:
                try:
                    data = data.copy()
                    data['datetime'] = pd.to_datetime(data['datetime'])
                except Exception:
                    pass
            
            # Sort by datetime if present
            if 'datetime' in data.columns:
                try:
                    data = data.sort_values('datetime')
                except Exception:
                    pass
            
            # Limit data for better performance and readability
            original_len = len(data)
            if len(data) > max_candles:
                data = data.tail(max_candles).copy()
                # Adjust pattern indices if needed
                if patterns:
                    offset = original_len - len(data)
                    adjusted_patterns = []
                    for p in patterns:
                        if hasattr(p, 'candle_index') and p.candle_index is not None and p.candle_index >= offset:
                            adjusted_patterns.append(PatternResult(
                                pattern_type=p.pattern_type,
                                confidence=p.confidence,
                                datetime=p.datetime,
                                timeframe=p.timeframe,
                                candle_index=p.candle_index - offset
                            ))
                    patterns = adjusted_patterns
            
            # Create figure
            fig = go.Figure()
            
            x_vals = self._get_x_values(data)
            # Convert to datetime for safety
            try:
                if hasattr(x_vals, 'dtype') and not isinstance(x_vals, pd.DatetimeIndex):
                    x_vals = pd.to_datetime(x_vals)
            except Exception:
                pass
            
            # Add candlestick trace with better visibility
            fig.add_trace(go. Candlestick(
                x=x_vals,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price",
                increasing_line_color='#2E7D32',
                decreasing_line_color='#C62828',
                increasing_fillcolor='rgba(76, 175, 80, 0.7)',
                decreasing_fillcolor='rgba(244, 67, 54, 0.7)',
                line=dict(width=1.5)
            ))
            
            # Add pattern markers
            if patterns:
                self._add_simple_pattern_markers(fig, patterns, data)
            
            # Compute x-range to avoid squeezing
            try:
                x_min = x_vals.min() if hasattr(x_vals, 'min') else x_vals[0]
                x_max = x_vals.max() if hasattr(x_vals, 'max') else x_vals[-1]
            except Exception:
                x_min, x_max = None, None
            
            # Clean layout with better grid visibility and fixed date axis
            fig.update_layout(
                title=dict(text=title, x=0.5, font=dict(size=16, color='#2E2E2E')),
                height=height,
                margin=dict(l=60, r=60, t=80, b=60),
                plot_bgcolor='#FAFAFA',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12, color='#2E2E2E'),
                showlegend=False,
                hovermode='x unified'
            )
            fig.update_xaxes(
                type='date',
                showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)',
                showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.5)',
                rangeslider=dict(visible=False)
            )
            if x_min is not None and x_max is not None:
                fig.update_xaxes(range=[x_min, x_max])
            fig.update_yaxes(
                title="Price",
                showgrid=True, gridwidth=1, gridcolor='rgba(128, 128, 128, 0.3)',
                showline=True, linewidth=1, linecolor='rgba(128, 128, 128, 0.5)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating simple chart: {e}")
            return self._create_empty_chart(title, height)
    
    def _add_simple_pattern_markers(self, fig: go.Figure, patterns: List[PatternResult], data: pd.DataFrame):
        """Add clean pattern markers to the chart."""
        try:
            x_vals = self._get_x_values(data)
            for pattern in patterns[:15]:  # Limit to 15 patterns for clarity
                if not hasattr(pattern, 'candle_index') or pattern.candle_index >= len(data):
                    continue
                
                candle_time = x_vals.iloc[pattern.candle_index] if hasattr(x_vals, 'iloc') else x_vals[pattern.candle_index]
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
    
    def add_structural_segments(self, fig: go.Figure, data: pd.DataFrame, segments: List[Dict[str, Any]]):
        """Overlay structural pattern segments as shaded regions with labels.
        Each segment dict should include: start_idx, end_idx, pattern_type, confidence, status,
        and optionally start_dt, end_dt for datetime boundaries.
        """
        try:
            if not segments:
                return
            x_vals = self._get_x_values(data)
            y_min = data['low'].min()
            y_max = data['high'].max()
            y_range = y_max - y_min if y_max > y_min else max(1.0, y_max)
            for seg in segments:
                try:
                    s_idx = max(0, int(seg.get('start_idx', 0)))
                    e_idx = min(len(data) - 1, int(seg.get('end_idx', len(data) - 1)))
                    if s_idx >= e_idx:
                        continue
                    x0 = seg.get('start_dt') if seg.get('start_dt') is not None else (x_vals.iloc[s_idx] if hasattr(x_vals, 'iloc') else x_vals[s_idx])
                    x1 = seg.get('end_dt') if seg.get('end_dt') is not None else (x_vals.iloc[e_idx] if hasattr(x_vals, 'iloc') else x_vals[e_idx])
                    label = f"{seg.get('pattern_type','pattern').replace('_',' ').title()}\n{seg.get('confidence',0):.0%} {seg.get('status','')}"
                    color = {
                        'head_and_shoulders': 'rgba(244, 67, 54, 0.15)',
                        'double_top': 'rgba(255, 152, 0, 0.15)',
                        'double_bottom': 'rgba(76, 175, 80, 0.15)'
                    }.get(seg.get('pattern_type',''), 'rgba(128,128,128,0.12)')
                    border_color = {
                        'head_and_shoulders': 'rgba(244, 67, 54, 0.6)',
                        'double_top': 'rgba(255, 152, 0, 0.6)',
                        'double_bottom': 'rgba(76, 175, 80, 0.6)'
                    }.get(seg.get('pattern_type',''), 'rgba(128,128,128,0.6)')
                    
                    fig.add_vrect(x0=x0, x1=x1, fillcolor=color, line_color=border_color, opacity=0.35, layer="below")
                    # Annotation near top of region
                    fig.add_annotation(
                        x=x0, y=y_max - 0.05 * y_range,
                        xref='x', yref='y',
                        text=label,
                        showarrow=False,
                        font=dict(size=11, color=border_color.replace('0.6','1.0') if isinstance(border_color, str) else '#333')
                    )
                except Exception as ie:
                    logger.warning(f"Failed to draw segment: {ie}")
        except Exception as e:
            logger.error(f"Error overlaying structural segments: {e}")
    
    def create_structural_chart(
        self,
        data: pd.DataFrame,
        segments: List[Dict[str, Any]],
        title: str = "Structural Pattern Scan",
        height: int = 500,
        max_candles: int = 300
    ) -> go.Figure:
        """Create a chart and overlay structural pattern segments."""
        fig = self.create_simple_chart(data=data, patterns=None, title=title, height=height, max_candles=max_candles)
        self.add_structural_segments(fig, data, segments)
        return fig
    
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
            if not hasattr(pattern, 'candle_index'):
                return self._create_empty_chart("Pattern Detail", 400)
            
            # Extract context around pattern
            start_idx = max(0, pattern.candle_index - context_candles)
            end_idx = min(len(data), pattern.candle_index + context_candles + 1)
            context_data = data.iloc[start_idx:end_idx].copy()
            
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
            logger.error(f"Error creating pattern detail chart: {e}")
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
