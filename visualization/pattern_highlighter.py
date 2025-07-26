"""
Pattern highlighting module for enhanced pattern visualization.

This module provides advanced pattern highlighting functionality including
distinct visual indicators, interactive markers, and click-to-navigate features.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from patterns.base import PatternResult

# Configure logging
logger = logging.getLogger(__name__)


class PatternHighlighter:
    """
    Advanced pattern highlighting functionality for candlestick charts.
    
    Provides distinct visual indicators for different pattern types,
    interactive markers, and navigation capabilities.
    """
    
    def __init__(self):
        """Initialize the pattern highlighter with configuration."""
        # Pattern-specific visual configurations
        self.pattern_configs = {
            'dragonfly_doji': {
                'color': '#FF6B6B',
                'symbol': 'triangle-down',
                'size': 14,
                'description': 'Potential reversal signal - long lower shadow with small body'
            },
            'hammer': {
                'color': '#4ECDC4',
                'symbol': 'diamond',
                'size': 12,
                'description': 'Bullish reversal pattern - small body with long lower shadow'
            },
            'rising_window': {
                'color': '#45B7D1',
                'symbol': 'arrow-up',
                'size': 16,
                'description': 'Gap up pattern - bullish continuation signal'
            },
            'evening_star': {
                'color': '#96CEB4',
                'symbol': 'star',
                'size': 15,
                'description': 'Three-candle bearish reversal pattern'
            },
            'three_white_soldiers': {
                'color': '#FFEAA7',
                'symbol': 'triangle-up',
                'size': 13,
                'description': 'Strong bullish pattern - three consecutive rising candles'
            }
        }
        
        # Default configuration for unknown patterns
        self.default_config = {
            'color': '#DDA0DD',
            'symbol': 'circle',
            'size': 10,
            'description': 'Detected candlestick pattern'
        }
        
        # Confidence-based styling
        self.confidence_styles = {
            'high': {'opacity': 1.0, 'line_width': 3},      # > 0.8
            'medium': {'opacity': 0.8, 'line_width': 2},    # 0.6 - 0.8
            'low': {'opacity': 0.6, 'line_width': 1}        # < 0.6
        }
    
    def add_pattern_overlays(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        data: pd.DataFrame,
        enable_click_navigation: bool = True
    ) -> go.Figure:
        """
        Add comprehensive pattern overlays to the chart.
        
        Args:
            fig: Existing Plotly figure
            patterns: List of detected patterns
            data: Original OHLCV data
            enable_click_navigation: Whether to enable click navigation
            
        Returns:
            Enhanced figure with pattern overlays
        """
        try:
            if not patterns:
                logger.info("No patterns to overlay")
                return fig
            
            logger.info(f"Adding pattern overlays for {len(patterns)} patterns")
            
            # Group patterns by type and confidence
            pattern_groups = self._group_patterns(patterns)
            
            # Add markers for each pattern group
            for pattern_type, pattern_list in pattern_groups.items():
                self._add_pattern_markers(fig, pattern_type, pattern_list, data)
            
            # Add pattern annotations
            self._add_pattern_annotations(fig, patterns, data)
            
            # Add pattern legend
            self._enhance_pattern_legend(fig)
            
            if enable_click_navigation:
                self._enable_click_navigation(fig, patterns)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error adding pattern overlays: {e}")
            return fig
    
    def create_pattern_detail_overlay(
        self, 
        fig: go.Figure, 
        pattern: PatternResult,
        data: pd.DataFrame,
        context_candles: int = 5
    ) -> go.Figure:
        """
        Create detailed overlay for a specific pattern with context.
        
        Args:
            fig: Figure to enhance
            pattern: Specific pattern to highlight
            data: OHLCV data
            context_candles: Number of candles to show before/after pattern
            
        Returns:
            Enhanced figure with detailed pattern overlay
        """
        try:
            # Find pattern location in data
            pattern_idx = self._find_pattern_index(pattern, data)
            if pattern_idx is None:
                logger.warning(f"Could not find pattern {pattern.pattern_type} in data")
                return fig
            
            # Define context window
            start_idx = max(0, pattern_idx - context_candles)
            end_idx = min(len(data) - 1, pattern_idx + context_candles)
            
            # Add context highlighting
            self._add_context_highlighting(fig, data, start_idx, end_idx, pattern_idx)
            
            # Add detailed pattern annotation
            self._add_detailed_annotation(fig, pattern, data.iloc[pattern_idx])
            
            # Add pattern-specific visual elements
            self._add_pattern_specific_elements(fig, pattern, data, pattern_idx)
            
            logger.info(f"Added detailed overlay for {pattern.pattern_type}")
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern detail overlay: {e}")
            return fig
    
    def add_confidence_indicators(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        data: pd.DataFrame
    ) -> go.Figure:
        """
        Add visual confidence indicators to pattern markers.
        
        Args:
            fig: Figure to enhance
            patterns: List of patterns with confidence scores
            data: OHLCV data
            
        Returns:
            Enhanced figure with confidence indicators
        """
        try:
            for pattern in patterns:
                config = self.pattern_configs.get(pattern.pattern_type, self.default_config)
                confidence_style = self._get_confidence_style(pattern.confidence)
                
                # Find pattern coordinates
                pattern_idx = self._find_pattern_index(pattern, data)
                if pattern_idx is None:
                    continue
                
                row = data.iloc[pattern_idx]
                x_coord = pattern.datetime
                y_coord = row['high'] * 1.03
                
                # Add confidence ring around marker
                confidence_ring = go.Scatter(
                    x=[x_coord],
                    y=[y_coord],
                    mode='markers',
                    marker=dict(
                        symbol='circle-open',
                        size=config['size'] + 8,
                        color=config['color'],
                        opacity=confidence_style['opacity'],
                        line=dict(
                            width=confidence_style['line_width'],
                            color=config['color']
                        )
                    ),
                    showlegend=False,
                    hoverinfo='skip'
                )
                
                fig.add_trace(confidence_ring)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error adding confidence indicators: {e}")
            return fig
    
    def create_pattern_timeline(self, patterns: List[PatternResult]) -> go.Figure:
        """
        Create a timeline view of detected patterns.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            Timeline figure showing pattern occurrences
        """
        try:
            if not patterns:
                return self._create_empty_timeline()
            
            # Sort patterns by datetime
            sorted_patterns = sorted(patterns, key=lambda p: p.datetime)
            
            # Create timeline data
            x_coords = [p.datetime for p in sorted_patterns]
            y_coords = list(range(len(sorted_patterns)))
            colors = [self.pattern_configs.get(p.pattern_type, self.default_config)['color'] 
                     for p in sorted_patterns]
            
            # Create hover text
            hover_texts = []
            for pattern in sorted_patterns:
                hover_text = (
                    f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>"
                    f"Time: {pattern.datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Confidence: {pattern.confidence:.1%}<br>"
                    f"Timeframe: {pattern.timeframe}"
                )
                hover_texts.append(hover_text)
            
            # Create timeline figure
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='markers+lines',
                marker=dict(
                    size=12,
                    color=colors,
                    line=dict(width=2, color='white')
                ),
                line=dict(width=2, color='rgba(128,128,128,0.3)'),
                hovertext=hover_texts,
                text=hover_texts,
                name='Pattern Timeline'
            ))
            
            # Configure layout
            fig.update_layout(
                title="Pattern Detection Timeline",
                xaxis_title="Time",
                yaxis_title="Pattern Sequence",
                height=400,
                showlegend=False,
                yaxis=dict(showticklabels=False)
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating pattern timeline: {e}")
            return self._create_empty_timeline()
    
    def _group_patterns(self, patterns: List[PatternResult]) -> Dict[str, List[PatternResult]]:
        """Group patterns by type for organized rendering."""
        groups = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            if pattern_type not in groups:
                groups[pattern_type] = []
            groups[pattern_type].append(pattern)
        return groups
    
    def _add_pattern_markers(
        self, 
        fig: go.Figure, 
        pattern_type: str, 
        patterns: List[PatternResult],
        data: pd.DataFrame
    ) -> None:
        """Add markers for a specific pattern type."""
        try:
            config = self.pattern_configs.get(pattern_type, self.default_config)
            
            # Extract coordinates and metadata
            x_coords = []
            y_coords = []
            hover_texts = []
            marker_sizes = []
            marker_opacities = []
            
            for pattern in patterns:
                pattern_idx = self._find_pattern_index(pattern, data)
                if pattern_idx is None:
                    continue
                
                row = data.iloc[pattern_idx]
                x_coords.append(pattern.datetime)
                y_coords.append(row['high'] * 1.02)
                
                # Create detailed hover text
                hover_text = (
                    f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>"
                    f"Confidence: {pattern.confidence:.1%}<br>"
                    f"Time: {pattern.datetime.strftime('%Y-%m-%d %H:%M:%S')}<br>"
                    f"Timeframe: {pattern.timeframe}<br>"
                    f"Description: {config['description']}"
                )
                hover_texts.append(hover_text)
                
                # Adjust size and opacity based on confidence
                confidence_style = self._get_confidence_style(pattern.confidence)
                marker_sizes.append(config['size'] * (0.8 + 0.4 * pattern.confidence))
                marker_opacities.append(confidence_style['opacity'])
            
            if x_coords:
                # Add pattern markers
                pattern_trace = go.Scatter(
                    x=x_coords,
                    y=y_coords,
                    mode='markers',
                    name=pattern_type.replace('_', ' ').title(),
                    marker=dict(
                        symbol=config['symbol'],
                        size=marker_sizes,
                        color=config['color'],
                        opacity=marker_opacities,
                        line=dict(width=2, color='white')
                    ),
                    hovertext=hover_texts,
                    text=hover_texts,
                    showlegend=True
                )
                
                fig.add_trace(pattern_trace)
                
        except Exception as e:
            logger.error(f"Error adding markers for {pattern_type}: {e}")
    
    def _add_pattern_annotations(
        self, 
        fig: go.Figure, 
        patterns: List[PatternResult],
        data: pd.DataFrame
    ) -> None:
        """Add text annotations for high-confidence patterns."""
        try:
            high_confidence_patterns = [p for p in patterns if p.confidence > 0.8]
            
            for pattern in high_confidence_patterns[:5]:  # Limit to avoid clutter
                pattern_idx = self._find_pattern_index(pattern, data)
                if pattern_idx is None:
                    continue
                
                row = data.iloc[pattern_idx]
                
                fig.add_annotation(
                    x=pattern.datetime,
                    y=row['high'] * 1.08,
                    text=f"{pattern.pattern_type.replace('_', ' ').title()}<br>{pattern.confidence:.0%}",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=self.pattern_configs.get(pattern.pattern_type, self.default_config)['color'],
                    font=dict(size=10, color="black"),
                    bgcolor="rgba(255,255,255,0.8)",
                    bordercolor=self.pattern_configs.get(pattern.pattern_type, self.default_config)['color'],
                    borderwidth=1
                )
                
        except Exception as e:
            logger.error(f"Error adding pattern annotations: {e}")
    
    def _enhance_pattern_legend(self, fig: go.Figure) -> None:
        """Enhance the legend with pattern information."""
        try:
            fig.update_layout(
                legend=dict(
                    orientation="v",
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=1.01,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor="rgba(0,0,0,0.2)",
                    borderwidth=1,
                    font=dict(size=10)
                )
            )
        except Exception as e:
            logger.error(f"Error enhancing legend: {e}")
    
    def _enable_click_navigation(self, fig: go.Figure, patterns: List[PatternResult]) -> None:
        """Enable click-to-navigate functionality (placeholder for future implementation)."""
        # Note: Full click navigation requires JavaScript callbacks
        # This is a placeholder for the interface design
        logger.info("Click navigation interface prepared (requires frontend integration)")
    
    def _find_pattern_index(self, pattern: PatternResult, data: pd.DataFrame) -> Optional[int]:
        """Find the index of a pattern in the data."""
        try:
            if 'datetime' in data.columns:
                mask = data['datetime'] == pattern.datetime
                if mask.any():
                    return data[mask].index[0]
            
            # Fallback to candle_index if available
            if 0 <= pattern.candle_index < len(data):
                return pattern.candle_index
                
            return None
            
        except Exception as e:
            logger.error(f"Error finding pattern index: {e}")
            return None
    
    def _get_confidence_style(self, confidence: float) -> Dict[str, Any]:
        """Get styling based on confidence level."""
        if confidence > 0.8:
            return self.confidence_styles['high']
        elif confidence > 0.6:
            return self.confidence_styles['medium']
        else:
            return self.confidence_styles['low']
    
    def _add_context_highlighting(
        self, 
        fig: go.Figure, 
        data: pd.DataFrame, 
        start_idx: int, 
        end_idx: int, 
        pattern_idx: int
    ) -> None:
        """Add context highlighting around a pattern."""
        try:
            # Add background highlighting for context window
            context_data = data.iloc[start_idx:end_idx+1]
            
            fig.add_shape(
                type="rect",
                x0=context_data.iloc[0]['datetime'] if 'datetime' in context_data.columns else context_data.index[0],
                x1=context_data.iloc[-1]['datetime'] if 'datetime' in context_data.columns else context_data.index[-1],
                y0=context_data['low'].min() * 0.995,
                y1=context_data['high'].max() * 1.005,
                fillcolor="rgba(255,255,0,0.1)",
                line=dict(color="rgba(255,255,0,0.3)", width=1),
                layer="below"
            )
            
        except Exception as e:
            logger.error(f"Error adding context highlighting: {e}")
    
    def _add_detailed_annotation(self, fig: go.Figure, pattern: PatternResult, candle_data: pd.Series) -> None:
        """Add detailed annotation for a specific pattern."""
        try:
            config = self.pattern_configs.get(pattern.pattern_type, self.default_config)
            
            annotation_text = (
                f"<b>{pattern.pattern_type.replace('_', ' ').title()}</b><br>"
                f"Confidence: {pattern.confidence:.1%}<br>"
                f"O: ${candle_data['open']:.2f} H: ${candle_data['high']:.2f}<br>"
                f"L: ${candle_data['low']:.2f} C: ${candle_data['close']:.2f}"
            )
            
            fig.add_annotation(
                x=pattern.datetime,
                y=candle_data['high'] * 1.1,
                text=annotation_text,
                showarrow=True,
                arrowhead=2,
                arrowsize=1.5,
                arrowwidth=2,
                arrowcolor=config['color'],
                font=dict(size=11, color="black"),
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=config['color'],
                borderwidth=2
            )
            
        except Exception as e:
            logger.error(f"Error adding detailed annotation: {e}")
    
    def _add_pattern_specific_elements(
        self, 
        fig: go.Figure, 
        pattern: PatternResult, 
        data: pd.DataFrame, 
        pattern_idx: int
    ) -> None:
        """Add pattern-specific visual elements."""
        try:
            candle = data.iloc[pattern_idx]
            
            # Add pattern-specific visual cues based on pattern type
            if pattern.pattern_type == 'hammer':
                self._add_hammer_indicators(fig, pattern, candle)
            elif pattern.pattern_type == 'dragonfly_doji':
                self._add_doji_indicators(fig, pattern, candle)
            elif pattern.pattern_type == 'rising_window':
                self._add_gap_indicators(fig, pattern, data, pattern_idx)
            
        except Exception as e:
            logger.error(f"Error adding pattern-specific elements: {e}")
    
    def _add_hammer_indicators(self, fig: go.Figure, pattern: PatternResult, candle: pd.Series) -> None:
        """Add visual indicators specific to hammer patterns."""
        # Add line to highlight the long lower shadow
        fig.add_shape(
            type="line",
            x0=pattern.datetime, x1=pattern.datetime,
            y0=candle['low'], y1=min(candle['open'], candle['close']),
            line=dict(color="#4ECDC4", width=4, dash="dot")
        )
    
    def _add_doji_indicators(self, fig: go.Figure, pattern: PatternResult, candle: pd.Series) -> None:
        """Add visual indicators specific to doji patterns."""
        # Add horizontal line to emphasize the small body
        body_center = (candle['open'] + candle['close']) / 2
        fig.add_shape(
            type="line",
            x0=pattern.datetime, x1=pattern.datetime,
            y0=body_center * 0.999, y1=body_center * 1.001,
            line=dict(color="#FF6B6B", width=6)
        )
    
    def _add_gap_indicators(self, fig: go.Figure, pattern: PatternResult, data: pd.DataFrame, pattern_idx: int) -> None:
        """Add visual indicators for gap patterns."""
        if pattern_idx > 0:
            prev_candle = data.iloc[pattern_idx - 1]
            curr_candle = data.iloc[pattern_idx]
            
            # Add arrow to show the gap
            fig.add_annotation(
                x=pattern.datetime,
                y=(prev_candle['high'] + curr_candle['low']) / 2,
                text="GAP",
                showarrow=True,
                arrowhead=2,
                arrowcolor="#45B7D1",
                font=dict(size=10, color="#45B7D1")
            )
    
    def _create_empty_timeline(self) -> go.Figure:
        """Create empty timeline figure."""
        fig = go.Figure()
        fig.update_layout(
            title="Pattern Timeline - No Patterns Detected",
            height=400,
            annotations=[
                dict(
                    text="No patterns detected in the selected timeframe",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    xanchor='center', yanchor='middle',
                    showarrow=False,
                    font=dict(size=14, color="gray")
                )
            ]
        )
        return fig