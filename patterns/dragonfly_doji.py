"""
Dragonfly Doji pattern detector implementation.
"""

from typing import Optional
import pandas as pd
from .base import BasePatternDetector


class DragonflyDojiDetector(BasePatternDetector):
    """
    Detector for Dragonfly Doji candlestick pattern.
    
    Characteristics:
    - Long lower shadow (>2x body size)
    - Minimal upper shadow (<10% of lower shadow)
    - Small body (<5% of total range)
    """
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Dragonfly Doji"
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Dragonfly Doji pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data):
            return None
            
        candle = data.iloc[index]
        
        # Calculate candle components
        body_size = self.calculate_body_size(candle)
        upper_shadow = self.calculate_upper_shadow(candle)
        lower_shadow = self.calculate_lower_shadow(candle)
        total_range = self.calculate_total_range(candle)
        
        # Avoid division by zero
        if total_range == 0:
            return None
        
        # Criteria scores for confidence calculation
        criteria_scores = []
        
        # Criterion 1: Small body (<5% of total range)
        body_ratio = body_size / total_range
        if body_ratio <= 0.05:
            criteria_scores.append(1.0)
        elif body_ratio <= 0.10:
            criteria_scores.append(0.7)
        elif body_ratio <= 0.15:
            criteria_scores.append(0.4)
        else:
            return None  # Body too large
        
        # Criterion 2: Long lower shadow (>2x body size)
        if body_size == 0:
            # Perfect doji case
            if lower_shadow > total_range * 0.6:  # Lower shadow > 60% of total range
                criteria_scores.append(1.0)
            elif lower_shadow > total_range * 0.4:
                criteria_scores.append(0.8)
            else:
                return None
        else:
            lower_shadow_ratio = lower_shadow / body_size
            if lower_shadow_ratio >= 4.0:
                criteria_scores.append(1.0)
            elif lower_shadow_ratio >= 3.0:
                criteria_scores.append(0.9)
            elif lower_shadow_ratio >= 2.0:
                criteria_scores.append(0.7)
            else:
                return None  # Lower shadow too short
        
        # Criterion 3: Minimal upper shadow (<10% of lower shadow)
        if lower_shadow == 0:
            return None  # No lower shadow means not a dragonfly doji
            
        upper_shadow_ratio = upper_shadow / lower_shadow
        if upper_shadow_ratio <= 0.05:
            criteria_scores.append(1.0)
        elif upper_shadow_ratio <= 0.10:
            criteria_scores.append(0.8)
        elif upper_shadow_ratio <= 0.20:
            criteria_scores.append(0.5)
        else:
            return None  # Upper shadow too long
        
        # Criterion 4: Lower shadow should be significant portion of total range
        lower_shadow_range_ratio = lower_shadow / total_range
        if lower_shadow_range_ratio >= 0.7:
            criteria_scores.append(1.0)
        elif lower_shadow_range_ratio >= 0.5:
            criteria_scores.append(0.8)
        elif lower_shadow_range_ratio >= 0.3:
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.3)
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _get_pattern_description(self) -> str:
        """Get pattern description."""
        return ("Dragonfly Doji: A reversal pattern with a small body, long lower shadow, "
                "and minimal upper shadow, indicating potential bullish reversal")