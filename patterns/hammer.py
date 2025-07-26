"""
Hammer pattern detector implementation with vectorized operations.
"""

from typing import Optional, List
import pandas as pd
import numpy as np
from .base import BasePatternDetector, PatternResult


class HammerDetector(BasePatternDetector):
    """
    Detector for Hammer candlestick pattern.
    
    Characteristics:
    - Lower shadow >2x body size
    - Upper shadow <50% of body size
    - Body in upper 1/3 of total range
    - Appears after downtrend
    """
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Hammer"
    
    def _get_min_periods(self) -> int:
        """Hammer needs at least 3 periods to confirm downtrend."""
        return 3
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Hammer pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data) or index < 2:
            return None
            
        candle = data.iloc[index]
        
        # Calculate candle components
        body_size = self.calculate_body_size(candle)
        upper_shadow = self.calculate_upper_shadow(candle)
        lower_shadow = self.calculate_lower_shadow(candle)
        total_range = self.calculate_total_range(candle)
        
        # Avoid division by zero
        if total_range == 0 or body_size == 0:
            return None
        
        # Criteria scores for confidence calculation
        criteria_scores = []
        
        # Criterion 1: Lower shadow >2x body size
        lower_shadow_ratio = lower_shadow / body_size
        if lower_shadow_ratio >= 3.0:
            criteria_scores.append(1.0)
        elif lower_shadow_ratio >= 2.5:
            criteria_scores.append(0.9)
        elif lower_shadow_ratio >= 2.0:
            criteria_scores.append(0.7)
        else:
            return None  # Lower shadow too short
        
        # Criterion 2: Upper shadow <50% of body size
        upper_shadow_ratio = upper_shadow / body_size
        if upper_shadow_ratio <= 0.2:
            criteria_scores.append(1.0)
        elif upper_shadow_ratio <= 0.3:
            criteria_scores.append(0.8)
        elif upper_shadow_ratio <= 0.5:
            criteria_scores.append(0.6)
        else:
            return None  # Upper shadow too long
        
        # Criterion 3: Body in upper 1/3 of total range
        body_position = (min(candle['open'], candle['close']) - candle['low']) / total_range
        if body_position >= 0.67:
            criteria_scores.append(1.0)
        elif body_position >= 0.5:
            criteria_scores.append(0.8)
        elif body_position >= 0.33:
            criteria_scores.append(0.5)
        else:
            return None  # Body not in upper portion
        
        # Criterion 4: Check for preceding downtrend
        downtrend_score = self._check_downtrend(data, index)
        if downtrend_score > 0:
            criteria_scores.append(downtrend_score)
        else:
            # Still valid hammer but lower confidence without downtrend
            criteria_scores.append(0.3)
        
        # Criterion 5: Body size should be reasonable (not too small)
        body_range_ratio = body_size / total_range
        if body_range_ratio >= 0.2:
            criteria_scores.append(1.0)
        elif body_range_ratio >= 0.1:
            criteria_scores.append(0.8)
        elif body_range_ratio >= 0.05:
            criteria_scores.append(0.5)
        else:
            criteria_scores.append(0.2)  # Very small body
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _check_downtrend(self, data: pd.DataFrame, index: int) -> float:
        """
        Check for preceding downtrend before the hammer.
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Score indicating strength of downtrend (0.0 to 1.0)
        """
        if index < 2:
            return 0.0
        
        # Look at previous 2-3 candles for downtrend
        lookback = min(3, index)
        prev_candles = data.iloc[index-lookback:index]
        
        # Count bearish candles
        bearish_count = 0
        declining_closes = 0
        
        for i in range(len(prev_candles)):
            candle = prev_candles.iloc[i]
            if self.is_bearish_candle(candle):
                bearish_count += 1
            
            # Check if closes are declining
            if i > 0:
                prev_close = prev_candles.iloc[i-1]['close']
                if candle['close'] < prev_close:
                    declining_closes += 1
        
        # Calculate downtrend strength
        bearish_ratio = bearish_count / len(prev_candles)
        declining_ratio = declining_closes / max(1, len(prev_candles) - 1)
        
        # Combine both factors
        downtrend_strength = (bearish_ratio + declining_ratio) / 2
        
        if downtrend_strength >= 0.8:
            return 1.0
        elif downtrend_strength >= 0.6:
            return 0.8
        elif downtrend_strength >= 0.4:
            return 0.6
        else:
            return 0.3
    
    def _detect_vectorized(self, data: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        """
        Vectorized detection of Hammer patterns.
        
        Args:
            data: OHLCV DataFrame
            timeframe: Time period
            
        Returns:
            List of detected patterns
        """
        try:
            # Need at least 3 periods for downtrend check
            if len(data) < 3:
                return self._detect_iterative(data, timeframe)
            
            # Calculate vectorized components
            components = self._calculate_vectorized_components(data)
            if not components:
                return self._detect_iterative(data, timeframe)
            
            body_size = components['body_size']
            upper_shadow = components['upper_shadow']
            lower_shadow = components['lower_shadow']
            total_range = components['total_range']
            body_size_safe = components['body_size_safe']
            
            # Vectorized criteria checks
            # Criterion 1: Lower shadow >2x body size
            lower_shadow_ratio = lower_shadow / body_size_safe
            long_lower_shadow_mask = lower_shadow_ratio >= 2.0
            
            # Criterion 2: Upper shadow <50% of body size
            upper_shadow_ratio = upper_shadow / body_size_safe
            short_upper_shadow_mask = upper_shadow_ratio <= 0.5
            
            # Criterion 3: Body in upper 1/3 of total range
            body_position = (np.minimum(data['open'], data['close']) - data['low']) / total_range
            upper_body_mask = body_position >= 0.33
            
            # Criterion 4: Body size should be reasonable (not too small)
            body_range_ratio = body_size / total_range
            reasonable_body_mask = body_range_ratio >= 0.05
            
            # Criterion 5: Vectorized downtrend check (simplified)
            # Check if previous 2 candles show declining trend
            downtrend_mask = np.zeros(len(data), dtype=bool)
            if len(data) >= 3:
                # Simple downtrend: previous close < close before that
                prev_declining = np.roll(data['close'], 1) < np.roll(data['close'], 2)
                # Current candle should be after some decline
                recent_decline = data['close'] < np.roll(data['close'], 1)
                downtrend_mask[2:] = (prev_declining & recent_decline)[2:]
            
            # Combine all criteria
            hammer_mask = (
                long_lower_shadow_mask &
                short_upper_shadow_mask &
                upper_body_mask &
                reasonable_body_mask &
                downtrend_mask
            )
            
            # Apply filters and create pattern results
            return self._apply_vectorized_filters(hammer_mask, data, timeframe)
            
        except Exception as e:
            self.logger.warning(f"Vectorized Hammer detection failed: {e}")
            return self._detect_iterative(data, timeframe)
    
    def _get_pattern_description(self) -> str:
        """Get pattern description."""
        return ("Hammer: A bullish reversal pattern with a small body, long lower shadow, "
                "and short upper shadow, typically appearing after a downtrend")