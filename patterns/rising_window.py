"""
Rising Window (Gap Up) pattern detector implementation.
"""

from typing import Optional
import pandas as pd
from .base import BasePatternDetector


class RisingWindowDetector(BasePatternDetector):
    """
    Detector for Rising Window (Gap Up) pattern.
    
    Characteristics:
    - Current candle low > previous candle high
    - Minimum gap size: 0.5% of previous close
    - Volume confirmation preferred
    """
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Rising Window"
    
    def _get_min_periods(self) -> int:
        """Rising window needs at least 2 periods."""
        return 2
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Rising Window pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data) or index < 1:
            return None
            
        current_candle = data.iloc[index]
        previous_candle = data.iloc[index - 1]
        
        # Basic gap check: current low > previous high
        if current_candle['low'] <= previous_candle['high']:
            return None
        
        # Calculate gap size
        gap_size = current_candle['low'] - previous_candle['high']
        gap_percentage = gap_size / previous_candle['close']
        
        # Minimum gap threshold (0.5%)
        if gap_percentage < 0.005:
            return None
        
        # Criteria scores for confidence calculation
        criteria_scores = []
        
        # Criterion 1: Gap size significance
        if gap_percentage >= 0.03:  # 3% or more
            criteria_scores.append(1.0)
        elif gap_percentage >= 0.02:  # 2-3%
            criteria_scores.append(0.9)
        elif gap_percentage >= 0.01:  # 1-2%
            criteria_scores.append(0.8)
        elif gap_percentage >= 0.005:  # 0.5-1%
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.3)
        
        # Criterion 2: Current candle should be bullish (preferred)
        if self.is_bullish_candle(current_candle):
            criteria_scores.append(1.0)
        elif current_candle['close'] >= current_candle['open']:
            criteria_scores.append(0.8)  # Doji or very small bearish
        else:
            criteria_scores.append(0.5)  # Bearish but still valid gap
        
        # Criterion 3: Volume confirmation
        volume_score = self._check_volume_confirmation(data, index)
        criteria_scores.append(volume_score)
        
        # Criterion 4: Previous candle should ideally be bullish too
        if self.is_bullish_candle(previous_candle):
            criteria_scores.append(0.9)
        else:
            criteria_scores.append(0.6)  # Still valid but less strong
        
        # Criterion 5: Gap should be clean (no overlap in bodies)
        current_body_low = min(current_candle['open'], current_candle['close'])
        previous_body_high = max(previous_candle['open'], previous_candle['close'])
        
        if current_body_low > previous_body_high:
            criteria_scores.append(1.0)  # Clean body gap
        elif current_candle['low'] > previous_body_high:
            criteria_scores.append(0.8)  # Shadow overlap but body gap
        else:
            criteria_scores.append(0.6)  # Shadow gap only
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _check_volume_confirmation(self, data: pd.DataFrame, index: int) -> float:
        """
        Check for volume confirmation of the gap.
        
        Args:
            data: OHLCV DataFrame
            index: Current candle index
            
        Returns:
            Volume confirmation score (0.0 to 1.0)
        """
        if index < 1:
            return 0.5  # Neutral if no previous data
        
        current_volume = data.iloc[index]['volume']
        previous_volume = data.iloc[index - 1]['volume']
        
        if previous_volume == 0:
            return 0.5  # Neutral if no volume data
        
        volume_ratio = current_volume / previous_volume
        
        # Higher volume on gap day is preferred
        if volume_ratio >= 2.0:
            return 1.0
        elif volume_ratio >= 1.5:
            return 0.9
        elif volume_ratio >= 1.2:
            return 0.8
        elif volume_ratio >= 1.0:
            return 0.7
        elif volume_ratio >= 0.8:
            return 0.5
        else:
            return 0.3  # Low volume gaps are less reliable
    
    def _get_pattern_description(self) -> str:
        """Get pattern description."""
        return ("Rising Window: A bullish continuation pattern where the current candle "
                "gaps up from the previous candle, indicating strong buying pressure")