"""
Enhanced chart rendering for stock pattern detection with detailed component visualization.

Focused on:
- Head & Shoulders pattern components (left shoulder, head, right shoulder, neckline)
- Double Top/Bottom pattern components (first top/bottom, second top/bottom, valley/peak)
- Clear visual indicators with labels and annotations
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
    Enhanced chart renderer for structural pattern visualization.
    
    Creates detailed candlestick charts with comprehensive pattern component display.
    """
    
    def __init__(self):
        """Initialize with enhanced settings for pattern visualization."""
        self.config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'select2d', 'lasso2d', 'autoScale2d', 
                'hoverClosestCartesian', 'hoverCompareCartesian'
            ],
            'responsive': True,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'pattern_chart',
                'height': 600,
                'width': 1200,
                'scale': 1
            }
        }
        
        # Enhanced color scheme for structural patterns
        mono = {
            'accent': '#111111',
            'muted': '#777777',
            'fill_up': 'rgba(0,0,0,0.85)',
            'fill_down': 'rgba(0,0,0,0.35)'
        }
        self.pattern_styles = {
            'Head & Shoulders': {
                'color': mono['accent'],
                'left_shoulder': {'color': '#444444', 'symbol': 'triangle-up', 'size': 12},
                'head': {'color': mono['accent'], 'symbol': 'star', 'size': 16},
                'right_shoulder': {'color': '#444444', 'symbol': 'triangle-up', 'size': 12},
                'neckline': {'color': mono['accent'], 'dash': 'dash', 'width': 2}
            },
            'Inverse Head & Shoulders': {
                'color': mono['accent'],
                'left_shoulder': {'color': '#444444', 'symbol': 'triangle-down', 'size': 12},
                'head': {'color': mono['accent'], 'symbol': 'star', 'size': 16},
                'right_shoulder': {'color': '#444444', 'symbol': 'triangle-down', 'size': 12},
                'neckline': {'color': mono['accent'], 'dash': 'dash', 'width': 2}
            },
            'Double Top': {
                'color': mono['accent'],
                'first_top': {'color': mono['accent'], 'symbol': 'triangle-down', 'size': 14},
                'second_top': {'color': '#333333', 'symbol': 'triangle-down', 'size': 14},
                'valley': {'color': '#666666', 'symbol': 'triangle-up', 'size': 10}
            },
            'Double Bottom': {
                'color': mono['accent'],
                'first_bottom': {'color': mono['accent'], 'symbol': 'triangle-up', 'size': 14},
                'second_bottom': {'color': '#333333', 'symbol': 'triangle-up', 'size': 14},
                'peak': {'color': '#666666', 'symbol': 'triangle-down', 'size': 10}
            }
        }
        
        self.default_pattern_style = {'color': '#95A5A6', 'symbol': 'circle', 'size': 8}
    
    def create_enhanced_pattern_chart(
        self, 
        data: pd.DataFrame, 
        patterns: List[PatternResult] = None,
        title: str = "Stock Pattern Analysis",
        height: int = 600,
        show_volume: bool = True
    ) -> go.Figure:
        """
        Create enhanced chart with detailed pattern component visualization.
        
        Args:
            data: OHLCV DataFrame with datetime index
            patterns: List of detected patterns to visualize
            title: Chart title
            height: Chart height in pixels
            show_volume: Whether to include volume subplot
            
        Returns:
            Enhanced Plotly chart with pattern components
        """
        try:
            if data is None or data.empty:
                return self._create_empty_chart(title, height)
            
            # Ensure datetime index for plotting if a 'datetime' column exists
            data = self._ensure_datetime_index(data)
            
            # Create subplots
            rows = 2 if show_volume and 'volume' in data.columns else 1
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.1,
                row_heights=[0.7, 0.3] if rows == 2 else [1.0],
                subplot_titles=("Price & Patterns", "Volume") if rows == 2 else ("Price & Patterns",)
            )
            
            # Add main candlestick chart
            self._add_enhanced_candlestick(fig, data, row=1)
            
            # Add pattern visualizations
            if patterns:
                self._add_enhanced_pattern_visualization(fig, patterns, data, row=1)
            
            # Add volume chart if requested
            if rows == 2:
                self._add_volume_chart(fig, data, patterns, row=2)
            
            # Enhanced layout
            self._apply_enhanced_layout(fig, title, height, rows)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating enhanced chart: {e}")
            return self._create_empty_chart(title, height)

    def create_pattern_chart(self, data: pd.DataFrame, patterns: List[PatternResult] = None,
                             title: str = "Stock Pattern Analysis", height: int = 600,
                             show_volume: bool = True) -> go.Figure:
        """Compatibility alias for older callers.

        Delegates to create_enhanced_pattern_chart with the same parameters.
        """
        return self.create_enhanced_pattern_chart(
            data=data,
            patterns=patterns,
            title=title,
            height=height,
            show_volume=show_volume
        )
    
    def _add_enhanced_candlestick(self, fig: go.Figure, data: pd.DataFrame, row: int = 1):
        """Add enhanced candlestick chart with better styling."""
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="Price",
            increasing_line_color='#111111',
            decreasing_line_color='#555555',
            increasing_fillcolor='rgba(0, 0, 0, 0.85)',
            decreasing_fillcolor='rgba(0, 0, 0, 0.35)',
            line=dict(width=1.2),
            showlegend=False
        ), row=row, col=1)
    
    def _add_enhanced_pattern_visualization(self, fig: go.Figure, patterns: List[PatternResult], 
                                          data: pd.DataFrame, row: int = 1):
        """Add enhanced pattern component visualization."""
        for pattern in patterns:
            pattern_name = pattern.pattern_name
            
            if pattern_name == 'Head & Shoulders':
                self._add_head_shoulders_visualization(fig, pattern, data, row)
            elif pattern_name == 'Inverse Head & Shoulders':
                self._add_inverse_head_shoulders_visualization(fig, pattern, data, row)
            elif pattern_name == 'Double Top':
                self._add_double_top_visualization(fig, pattern, data, row)
            elif pattern_name == 'Double Bottom':
                self._add_double_bottom_visualization(fig, pattern, data, row)
            
            # Add pattern period highlight
            self._add_pattern_period_highlight(fig, pattern, data, row)
    
    def _add_head_shoulders_visualization(self, fig: go.Figure, pattern: PatternResult, 
                                        data: pd.DataFrame, row: int):
        """Add detailed Head & Shoulders pattern visualization."""
        style = self.pattern_styles['Head & Shoulders']
        
        if not hasattr(pattern, 'metadata') or not pattern.metadata:
            return
        
        metadata = pattern.metadata
        
        # Left Shoulder
        if 'left_shoulder_idx' in metadata and metadata['left_shoulder_idx'] < len(data):
            idx = metadata['left_shoulder_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['Left Shoulder'],
                textposition='top center',
                textfont=dict(size=10, color=style['left_shoulder']['color']),
                marker=dict(
                    size=style['left_shoulder']['size'],
                    color=style['left_shoulder']['color'],
                    symbol=style['left_shoulder']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Left Shoulder',
                showlegend=True,
                hovertemplate='<b>Left Shoulder</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Head
        if 'head_idx' in metadata and metadata['head_idx'] < len(data):
            idx = metadata['head_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['HEAD'],
                textposition='top center',
                textfont=dict(size=12, color=style['head']['color']),
                marker=dict(
                    size=style['head']['size'],
                    color=style['head']['color'],
                    symbol=style['head']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Head',
                showlegend=True,
                hovertemplate='<b>Head</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Right Shoulder
        if 'right_shoulder_idx' in metadata and metadata['right_shoulder_idx'] < len(data):
            idx = metadata['right_shoulder_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['Right Shoulder'],
                textposition='top center',
                textfont=dict(size=10, color=style['right_shoulder']['color']),
                marker=dict(
                    size=style['right_shoulder']['size'],
                    color=style['right_shoulder']['color'],
                    symbol=style['right_shoulder']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Right Shoulder',
                showlegend=True,
                hovertemplate='<b>Right Shoulder</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Neckline
        if ('left_valley_idx' in metadata and 'right_valley_idx' in metadata and
            metadata['left_valley_idx'] < len(data) and metadata['right_valley_idx'] < len(data)):
            
            left_idx = metadata['left_valley_idx']
            right_idx = metadata['right_valley_idx']
            
            fig.add_trace(go.Scatter(
                x=[data.index[left_idx], data.index[right_idx]],
                y=[data.iloc[left_idx]['low'], data.iloc[right_idx]['low']],
                mode='lines+markers',
                line=dict(
                    color=style['neckline']['color'],
                    width=style['neckline']['width'],
                    dash=style['neckline']['dash']
                ),
                marker=dict(size=8, color=style['neckline']['color']),
                name='Neckline',
                showlegend=True,
                hovertemplate='<b>Neckline</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
    
    def _add_inverse_head_shoulders_visualization(self, fig: go.Figure, pattern: PatternResult, 
                                                data: pd.DataFrame, row: int):
        """Add detailed Inverse Head & Shoulders pattern visualization."""
        style = self.pattern_styles['Inverse Head & Shoulders']
        
        if not hasattr(pattern, 'metadata') or not pattern.metadata:
            return
        
        metadata = pattern.metadata
        
        # Left Shoulder (for inverse, we use low points)
        if 'left_shoulder_idx' in metadata and metadata['left_shoulder_idx'] < len(data):
            idx = metadata['left_shoulder_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['Left Shoulder'],
                textposition='bottom center',
                textfont=dict(size=10, color=style['left_shoulder']['color']),
                marker=dict(
                    size=style['left_shoulder']['size'],
                    color=style['left_shoulder']['color'],
                    symbol=style['left_shoulder']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Left Shoulder',
                showlegend=True,
                hovertemplate='<b>Left Shoulder</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Head (lowest point for inverse)
        if 'head_idx' in metadata and metadata['head_idx'] < len(data):
            idx = metadata['head_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['HEAD'],
                textposition='bottom center',
                textfont=dict(size=12, color=style['head']['color']),
                marker=dict(
                    size=style['head']['size'],
                    color=style['head']['color'],
                    symbol=style['head']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Head',
                showlegend=True,
                hovertemplate='<b>Head</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Right Shoulder
        if 'right_shoulder_idx' in metadata and metadata['right_shoulder_idx'] < len(data):
            idx = metadata['right_shoulder_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['Right Shoulder'],
                textposition='bottom center',
                textfont=dict(size=10, color=style['right_shoulder']['color']),
                marker=dict(
                    size=style['right_shoulder']['size'],
                    color=style['right_shoulder']['color'],
                    symbol=style['right_shoulder']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Right Shoulder',
                showlegend=True,
                hovertemplate='<b>Right Shoulder</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Neckline (connects peaks for inverse pattern)
        if ('left_peak_idx' in metadata and 'right_peak_idx' in metadata and
            metadata['left_peak_idx'] < len(data) and metadata['right_peak_idx'] < len(data)):
            
            left_idx = metadata['left_peak_idx']
            right_idx = metadata['right_peak_idx']
            
            fig.add_trace(go.Scatter(
                x=[data.index[left_idx], data.index[right_idx]],
                y=[data.iloc[left_idx]['high'], data.iloc[right_idx]['high']],
                mode='lines+markers',
                line=dict(
                    color=style['neckline']['color'],
                    width=style['neckline']['width'],
                    dash=style['neckline']['dash']
                ),
                marker=dict(size=8, color=style['neckline']['color']),
                name='Neckline',
                showlegend=True,
                hovertemplate='<b>Neckline</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
    
    def _add_double_top_visualization(self, fig: go.Figure, pattern: PatternResult, 
                                    data: pd.DataFrame, row: int):
        """Add detailed Double Top pattern visualization."""
        style = self.pattern_styles['Double Top']
        
        if not hasattr(pattern, 'metadata') or not pattern.metadata:
            return
        
        metadata = pattern.metadata
        
        # First Top
        if 'first_peak_idx' in metadata and metadata['first_peak_idx'] < len(data):
            idx = metadata['first_peak_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['First Top'],
                textposition='top center',
                textfont=dict(size=10, color=style['first_top']['color']),
                marker=dict(
                    size=style['first_top']['size'],
                    color=style['first_top']['color'],
                    symbol=style['first_top']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='First Top',
                showlegend=True,
                hovertemplate='<b>First Top</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Second Top
        if 'second_peak_idx' in metadata and metadata['second_peak_idx'] < len(data):
            idx = metadata['second_peak_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['Second Top'],
                textposition='top center',
                textfont=dict(size=10, color=style['second_top']['color']),
                marker=dict(
                    size=style['second_top']['size'],
                    color=style['second_top']['color'],
                    symbol=style['second_top']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Second Top',
                showlegend=True,
                hovertemplate='<b>Second Top</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Valley between tops
        if 'valley_idx' in metadata and metadata['valley_idx'] < len(data):
            idx = metadata['valley_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['Valley'],
                textposition='bottom center',
                textfont=dict(size=9, color=style['valley']['color']),
                marker=dict(
                    size=style['valley']['size'],
                    color=style['valley']['color'],
                    symbol=style['valley']['symbol'],
                    line=dict(width=1, color='white')
                ),
                name='Valley',
                showlegend=True,
                hovertemplate='<b>Valley</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
    
    def _add_double_bottom_visualization(self, fig: go.Figure, pattern: PatternResult, 
                                       data: pd.DataFrame, row: int):
        """Add detailed Double Bottom pattern visualization."""
        style = self.pattern_styles['Double Bottom']
        
        if not hasattr(pattern, 'metadata') or not pattern.metadata:
            return
        
        metadata = pattern.metadata
        
        # First Bottom
        if 'first_trough_idx' in metadata and metadata['first_trough_idx'] < len(data):
            idx = metadata['first_trough_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['First Bottom'],
                textposition='bottom center',
                textfont=dict(size=10, color=style['first_bottom']['color']),
                marker=dict(
                    size=style['first_bottom']['size'],
                    color=style['first_bottom']['color'],
                    symbol=style['first_bottom']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='First Bottom',
                showlegend=True,
                hovertemplate='<b>First Bottom</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Second Bottom
        if 'second_trough_idx' in metadata and metadata['second_trough_idx'] < len(data):
            idx = metadata['second_trough_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['low']],
                mode='markers+text',
                text=['Second Bottom'],
                textposition='bottom center',
                textfont=dict(size=10, color=style['second_bottom']['color']),
                marker=dict(
                    size=style['second_bottom']['size'],
                    color=style['second_bottom']['color'],
                    symbol=style['second_bottom']['symbol'],
                    line=dict(width=2, color='white')
                ),
                name='Second Bottom',
                showlegend=True,
                hovertemplate='<b>Second Bottom</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
        
        # Peak between bottoms
        if 'peak_idx' in metadata and metadata['peak_idx'] < len(data):
            idx = metadata['peak_idx']
            fig.add_trace(go.Scatter(
                x=[data.index[idx]],
                y=[data.iloc[idx]['high']],
                mode='markers+text',
                text=['Peak'],
                textposition='top center',
                textfont=dict(size=9, color=style['peak']['color']),
                marker=dict(
                    size=style['peak']['size'],
                    color=style['peak']['color'],
                    symbol=style['peak']['symbol'],
                    line=dict(width=1, color='white')
                ),
                name='Peak',
                showlegend=True,
                hovertemplate='<b>Peak</b><br>Price: $%{y:.2f}<br>Date: %{x}<extra></extra>'
            ), row=row, col=1)
    
    def _add_pattern_period_highlight(self, fig: go.Figure, pattern: PatternResult, 
                                    data: pd.DataFrame, row: int):
        """Add subtle background highlight for pattern period."""
        if not hasattr(pattern, 'start_idx') or not hasattr(pattern, 'end_idx'):
            return
        
        if pattern.start_idx >= len(data) or pattern.end_idx >= len(data):
            return
        
        start_date = data.index[pattern.start_idx]
        end_date = data.index[pattern.end_idx]
        
        pattern_color = self.pattern_styles.get(pattern.pattern_name, {}).get('color', '#95A5A6')
        
        fig.add_vrect(
            x0=start_date,
            x1=end_date,
            fillcolor=pattern_color,
            opacity=0.1,
            layer="below",
            line_width=0,
            row=row,
            col=1
        )
    
    def _add_volume_chart(self, fig: go.Figure, data: pd.DataFrame, 
                         patterns: List[PatternResult], row: int):
        """Add volume chart with pattern highlights."""
        if 'volume' not in data.columns:
            return
        
        # Volume bars with color coding
        colors = ['#26a69a' if close >= open else '#ef5350' 
                  for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=colors,
            name='Volume',
            opacity=0.7,
            showlegend=False,
            hovertemplate='Volume: %{y:,.0f}<br>Date: %{x}<extra></extra>'
        ), row=row, col=1)
        
        # Highlight pattern periods in volume
        if patterns:
            for pattern in patterns:
                if (hasattr(pattern, 'start_idx') and hasattr(pattern, 'end_idx') and
                    pattern.start_idx < len(data) and pattern.end_idx < len(data)):
                    
                    start_date = data.index[pattern.start_idx]
                    end_date = data.index[pattern.end_idx]
                    
                    fig.add_vrect(
                        x0=start_date,
                        x1=end_date,
                        fillcolor="yellow",
                        opacity=0.2,
                        layer="below",
                        line_width=0,
                        row=row,
                        col=1
                    )
    
    def _apply_enhanced_layout(self, fig: go.Figure, title: str, height: int, rows: int):
        """Apply enhanced layout settings."""
        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                font=dict(size=18, color='#111111'),
                pad=dict(t=20)
            ),
            height=height,
            margin=dict(l=60, r=60, t=100, b=60),
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(family="Inter, Arial, sans-serif", size=12, color='#111111'),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode='x unified'
        )

        # Price chart axes
        fig.update_xaxes(
            type='date',
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.08)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.15)',
            rangeslider=dict(visible=False),
            rangebreaks=[dict(bounds=["sat", "mon"])],
            row=1,
            col=1
        )

        fig.update_yaxes(
            title="Price",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0, 0, 0, 0.08)',
            showline=True,
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.15)',
            row=1,
            col=1
        )

        # Volume chart axes (if present)
        if rows == 2:
            fig.update_xaxes(
                type='date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0, 0, 0, 0.08)',
                row=2,
                col=1
            )

            fig.update_yaxes(
                title="Volume",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0, 0, 0, 0.08)',
                row=2,
                col=1
            )
    
    def create_compact_chart(self, data: pd.DataFrame, patterns: List[PatternResult] = None,
                           title: str = "Preview", height: int = 300) -> go.Figure:
        """Create a compact chart for previews."""
        return self.create_enhanced_pattern_chart(
            data=data,
            patterns=patterns,
            title=title,
            height=height,
            show_volume=False
        )
    
    def _create_empty_chart(self, title: str, height: int) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            text="No data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            title=title,
            height=height,
            template="plotly_white"
        )
        return fig

    def _ensure_datetime_index(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with datetime index if 'datetime' column exists, else original.

        This avoids UI errors when upstream code passes data without setting the index.
        """
        try:
            if 'datetime' in data.columns:
                df = data.copy()
                df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                df = df.dropna(subset=['datetime'])
                df = df.set_index('datetime')
                df = df.sort_index()
                return df
        except Exception:
            pass
        return data
