"""
Evening Star pattern detector implementation.
"""

from typing import Optional
import pandas as pd
from .base import BasePatternDetector


class EveningStarDetector(BasePatternDetector):
    """
    Detector for Evening Star candlestick pattern.
    
    Characteristics:
    - Three-candle pattern
    - First: Long bullish candle
    - Second: Small body (doji/spinning top)
    - Third: Long bearish candle closing below first candle midpoint
    """
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Evening Star"
    
    def _get_min_periods(self) -> int:
        """Evening star needs exactly 3 periods."""
        return 3
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Evening Star pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check (should be the third candle)
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data) or index < 2:
            return None
            
        # Get the three candles (first, star, third)
        first_candle = data.iloc[index - 2]
        star_candle = data.iloc[index - 1]
        third_candle = data.iloc[index]
        
        # Criteria scores for confidence calculation
        criteria_scores = []
        
        # Criterion 1: First candle must be bullish and reasonably long
        if not self.is_bullish_candle(first_candle):
            return None
        
        first_body_size = self.calculate_body_size(first_candle)
        first_range = self.calculate_total_range(first_candle)
        
        if first_range == 0:
            return None
            
        first_body_ratio = first_body_size / first_range
        if first_body_ratio >= 0.7:
            criteria_scores.append(1.0)
        elif first_body_ratio >= 0.5:
            criteria_scores.append(0.8)
        elif first_body_ratio >= 0.3:
            criteria_scores.append(0.6)
        else:
            return None  # First candle body too small
        
        # Criterion 2: Star candle should have small body
        star_body_size = self.calculate_body_size(star_candle)
        star_range = self.calculate_total_range(star_candle)
        
        if star_range == 0:
            return None
            
        star_body_ratio = star_body_size / star_range
        if star_body_ratio <= 0.1:
            criteria_scores.append(1.0)  # Doji-like
        elif star_body_ratio <= 0.2:
            criteria_scores.append(0.9)  # Small body
        elif star_body_ratio <= 0.3:
            criteria_scores.append(0.7)  # Spinning top
        else:
            return None  # Star body too large
        
        # Criterion 3: Star should gap up from first candle
        star_gap_score = self._check_star_gap(first_candle, star_candle)
        criteria_scores.append(star_gap_score)
        
        # Criterion 4: Third candle must be bearish
        if not self.is_bearish_candle(third_candle):
            return None
        
        third_body_size = self.calculate_body_size(third_candle)
        third_range = self.calculate_total_range(third_candle)
        
        if third_range == 0:
            return None
            
        third_body_ratio = third_body_size / third_range
        if third_body_ratio >= 0.7:
            criteria_scores.append(1.0)
        elif third_body_ratio >= 0.5:
            criteria_scores.append(0.8)
        elif third_body_ratio >= 0.3:
            criteria_scores.append(0.6)
        else:
            return None  # Third candle body too small
        
        # Criterion 5: Third candle should close below first candle midpoint
        first_midpoint = (first_candle['open'] + first_candle['close']) / 2
        penetration_score = self._check_penetration(first_candle, third_candle, first_midpoint)
        if penetration_score == 0:
            return None  # No penetration
        criteria_scores.append(penetration_score)
        
        # Criterion 6: Volume confirmation (optional but preferred)
        volume_score = self._check_volume_pattern(data, index)
        criteria_scores.append(volume_score)
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _check_star_gap(self, first_candle: pd.Series, star_candle: pd.Series) -> float:
        """
        Check if star candle gaps up from first candle.
        
        Args:
            first_candle: First candle data
            star_candle: Star candle data
            
        Returns:
            Gap score (0.0 to 1.0)
        """
        first_high = first_candle['high']
        star_low = star_candle['low']
        
        if star_low > first_high:
            # True gap
            gap_size = star_low - first_high
            gap_percentage = gap_size / first_candle['close']
            
            if gap_percentage >= 0.02:
                return 1.0
            elif gap_percentage >= 0.01:
                return 0.9
            else:
                return 0.8
        elif star_candle['open'] > first_candle['close']:
            # Body gap but shadow overlap
            return 0.7
        else:
            # No gap but star should be higher
            if star_candle['close'] > first_candle['close']:
                return 0.5
            else:
                return 0.2
    
    def _check_penetration(self, first_candle: pd.Series, third_candle: pd.Series, 
                          first_midpoint: float) -> float:
        """
        Check how much the third candle penetrates into the first candle.
        
        Args:
            first_candle: First candle data
            third_candle: Third candle data
            first_midpoint: Midpoint of first candle
            
        Returns:
            Penetration score (0.0 to 1.0)
        """
        third_close = third_candle['close']
        first_open = first_candle['open']
        
        if third_close >= first_midpoint:
            return 0  # No significant penetration
        
        # Calculate penetration depth
        penetration = first_midpoint - third_close
        first_body_size = self.calculate_body_size(first_candle)
        
        if first_body_size == 0:
            return 0
            
        penetration_ratio = penetration / first_body_size
        
        if third_close <= first_open:
            return 1.0  # Complete penetration
        elif penetration_ratio >= 0.75:
            return 0.9
        elif penetration_ratio >= 0.5:
            return 0.8
        elif penetration_ratio >= 0.25:
            return 0.6
        else:
            return 0.3
    
    def _check_volume_pattern(self, data: pd.DataFrame, index: int) -> float:
        """
        Check volume pattern for evening star confirmation.
        
        Args:
            data: OHLCV DataFrame
            index: Current index (third candle)
            
        Returns:
            Volume pattern score (0.0 to 1.0)
        """
        if index < 2:
            return 0.5
        
        first_volume = data.iloc[index - 2]['volume']
        star_volume = data.iloc[index - 1]['volume']
        third_volume = data.iloc[index]['volume']
        
        if first_volume == 0 or star_volume == 0 or third_volume == 0:
            return 0.5  # Neutral if no volume data
        
        # Ideal pattern: high volume on first and third, lower on star
        score = 0.5
        
        # First candle should have good volume
        if first_volume >= star_volume:
            score += 0.1
        
        # Third candle should have high volume (selling pressure)
        if third_volume >= first_volume:
            score += 0.2
        elif third_volume >= star_volume:
            score += 0.1
        
        # Star should have relatively lower volume
        if star_volume <= min(first_volume, third_volume):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_pattern_description(self) -> str:
        """Get pattern description."""
        return ("Evening Star: A bearish reversal pattern consisting of three candles - "
                "a long bullish candle, a small-bodied star candle, and a long bearish candle")