"""
Three White Soldiers pattern detector implementation.
"""

from typing import Optional
import pandas as pd
from .base import BasePatternDetector


class ThreeWhiteSoldiersDetector(BasePatternDetector):
    """
    Detector for Three White Soldiers candlestick pattern.
    
    Characteristics:
    - Three consecutive bullish candles
    - Each opens within previous body
    - Each closes near session high
    - Increasing or stable volume
    """
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Three White Soldiers"
    
    def _get_min_periods(self) -> int:
        """Three White Soldiers needs exactly 3 periods."""
        return 3
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Three White Soldiers pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check (should be the third candle)
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data) or index < 2:
            return None
            
        # Get the three consecutive candles
        first_candle = data.iloc[index - 2]
        second_candle = data.iloc[index - 1]
        third_candle = data.iloc[index]
        
        candles = [first_candle, second_candle, third_candle]
        
        # Criteria scores for confidence calculation
        criteria_scores = []
        
        # Criterion 1: All three candles must be bullish
        for i, candle in enumerate(candles):
            if not self.is_bullish_candle(candle):
                return None
        
        # All bullish - add base score
        criteria_scores.append(1.0)
        
        # Criterion 2: Each candle should have substantial body
        body_scores = []
        for candle in candles:
            body_size = self.calculate_body_size(candle)
            total_range = self.calculate_total_range(candle)
            
            if total_range == 0:
                return None
                
            body_ratio = body_size / total_range
            if body_ratio >= 0.7:
                body_scores.append(1.0)
            elif body_ratio >= 0.5:
                body_scores.append(0.8)
            elif body_ratio >= 0.3:
                body_scores.append(0.6)
            else:
                body_scores.append(0.3)
        
        criteria_scores.append(sum(body_scores) / len(body_scores))
        
        # Criterion 3: Each candle opens within previous candle's body
        opening_scores = []
        
        # Second candle opens within first candle's body
        first_body_low = min(first_candle['open'], first_candle['close'])
        first_body_high = max(first_candle['open'], first_candle['close'])
        
        if first_body_low <= second_candle['open'] <= first_body_high:
            opening_scores.append(1.0)
        elif second_candle['open'] > first_candle['open']:
            opening_scores.append(0.7)  # Opens higher but outside body
        else:
            opening_scores.append(0.3)  # Opens lower
        
        # Third candle opens within second candle's body
        second_body_low = min(second_candle['open'], second_candle['close'])
        second_body_high = max(second_candle['open'], second_candle['close'])
        
        if second_body_low <= third_candle['open'] <= second_body_high:
            opening_scores.append(1.0)
        elif third_candle['open'] > second_candle['open']:
            opening_scores.append(0.7)  # Opens higher but outside body
        else:
            opening_scores.append(0.3)  # Opens lower
        
        criteria_scores.append(sum(opening_scores) / len(opening_scores))
        
        # Criterion 4: Each candle closes near its high
        close_high_scores = []
        for candle in candles:
            if candle['high'] == candle['low']:  # Avoid division by zero
                close_high_scores.append(0.5)
                continue
                
            close_position = (candle['close'] - candle['low']) / (candle['high'] - candle['low'])
            
            if close_position >= 0.9:
                close_high_scores.append(1.0)
            elif close_position >= 0.8:
                close_high_scores.append(0.9)
            elif close_position >= 0.7:
                close_high_scores.append(0.7)
            elif close_position >= 0.6:
                close_high_scores.append(0.5)
            else:
                close_high_scores.append(0.3)
        
        criteria_scores.append(sum(close_high_scores) / len(close_high_scores))
        
        # Criterion 5: Progressive price advancement
        advancement_score = self._check_price_advancement(candles)
        criteria_scores.append(advancement_score)
        
        # Criterion 6: Volume pattern (increasing or stable)
        volume_score = self._check_volume_pattern(candles)
        criteria_scores.append(volume_score)
        
        # Criterion 7: Reasonable candle sizes (not too small)
        size_scores = []
        for candle in candles:
            body_size = self.calculate_body_size(candle)
            # Compare to average of recent candles for context
            if index >= 5:
                recent_avg = self._get_average_body_size(data, index, 5)
                if recent_avg > 0:
                    size_ratio = body_size / recent_avg
                    if size_ratio >= 1.2:
                        size_scores.append(1.0)
                    elif size_ratio >= 1.0:
                        size_scores.append(0.8)
                    elif size_ratio >= 0.8:
                        size_scores.append(0.6)
                    else:
                        size_scores.append(0.4)
                else:
                    size_scores.append(0.7)
            else:
                size_scores.append(0.7)  # Neutral if insufficient history
        
        criteria_scores.append(sum(size_scores) / len(size_scores))
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _check_price_advancement(self, candles: list) -> float:
        """
        Check that each candle closes higher than the previous.
        
        Args:
            candles: List of three candle data
            
        Returns:
            Price advancement score (0.0 to 1.0)
        """
        if len(candles) != 3:
            return 0.0
        
        # Each close should be higher than previous
        if (candles[1]['close'] > candles[0]['close'] and 
            candles[2]['close'] > candles[1]['close']):
            
            # Calculate advancement strength
            first_advance = (candles[1]['close'] - candles[0]['close']) / candles[0]['close']
            second_advance = (candles[2]['close'] - candles[1]['close']) / candles[1]['close']
            
            avg_advance = (first_advance + second_advance) / 2
            
            if avg_advance >= 0.03:  # 3% average advance
                return 1.0
            elif avg_advance >= 0.02:  # 2% average advance
                return 0.9
            elif avg_advance >= 0.01:  # 1% average advance
                return 0.8
            else:
                return 0.6
        else:
            return 0.2  # Not advancing properly
    
    def _check_volume_pattern(self, candles: list) -> float:
        """
        Check volume pattern for three white soldiers.
        
        Args:
            candles: List of three candle data
            
        Returns:
            Volume pattern score (0.0 to 1.0)
        """
        if len(candles) != 3:
            return 0.5
        
        volumes = [candle['volume'] for candle in candles]
        
        # Check for zero volumes
        if any(vol == 0 for vol in volumes):
            return 0.5  # Neutral if no volume data
        
        # Ideal: increasing or stable volume
        if volumes[1] >= volumes[0] and volumes[2] >= volumes[1]:
            # Increasing volume
            if volumes[2] > volumes[0] * 1.2:
                return 1.0
            elif volumes[2] > volumes[0]:
                return 0.9
            else:
                return 0.8  # Stable volume
        elif volumes[1] >= volumes[0] * 0.8 and volumes[2] >= volumes[1] * 0.8:
            return 0.6  # Relatively stable
        else:
            return 0.3  # Declining volume
    
    def _get_average_body_size(self, data: pd.DataFrame, index: int, periods: int) -> float:
        """
        Calculate average body size for recent candles.
        
        Args:
            data: OHLCV DataFrame
            index: Current index
            periods: Number of periods to look back
            
        Returns:
            Average body size
        """
        start_idx = max(0, index - periods)
        recent_candles = data.iloc[start_idx:index]
        
        if len(recent_candles) == 0:
            return 0.0
        
        body_sizes = [self.calculate_body_size(candle) for _, candle in recent_candles.iterrows()]
        return sum(body_sizes) / len(body_sizes)
    
    def _get_pattern_description(self) -> str:
        """Get pattern description."""
        return ("Three White Soldiers: A bullish reversal pattern of three consecutive "
                "long bullish candles, each opening within the previous body and closing near the high")