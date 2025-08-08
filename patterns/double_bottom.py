"""
Double Bottom pattern detector implementation.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from .base import BasePatternDetector, PatternResult


class DoubleBottomDetector(BasePatternDetector):
    """
    Detector for Double Bottom reversal pattern.
    
    Characteristics:
    - Two distinct troughs at approximately the same level
    - Peak between troughs (at least 10-20% rally from troughs)
    - Second trough occurs after the first with intervening rally
    - Pattern confirms when price breaks above the peak high
    - Indicates potential trend reversal from bearish to bullish
    """
    
    def __init__(self, min_confidence: float = 0.5, trough_tolerance: float = 0.02, 
                 min_peak_rally: float = 0.10, lookback_periods: int = 50):
        """
        Initialize Double Bottom detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            trough_tolerance: Maximum percentage difference between troughs (default 2%)
            min_peak_rally: Minimum rally from trough to peak (default 10%)
            lookback_periods: Number of periods to look back for pattern formation
        """
        super().__init__(min_confidence)
        self.trough_tolerance = trough_tolerance
        self.min_peak_rally = min_peak_rally
        self.lookback_periods = lookback_periods
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Double Bottom"
    
    def _get_min_periods(self) -> int:
        """Double bottom needs sufficient periods to form."""
        return self.lookback_periods
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Double Bottom pattern at specific index.
        
        Args:
            data: OHLCV DataFrame
            index: Index position to check
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        if index >= len(data) or index < self.lookback_periods:
            return None
        
        # Look back from current index
        lookback_data = data.iloc[max(0, index - self.lookback_periods):index + 1]
        
        if len(lookback_data) < self.lookback_periods:
            return None
        
        # Find peaks and valleys in the lookback period
        peaks, valleys = self._find_peaks_and_valleys(lookback_data)
        
        if len(valleys) < 2 or len(peaks) < 1:
            return None
        
        # Check for double bottom pattern
        pattern_info = self._analyze_double_bottom_pattern(lookback_data, peaks, valleys)
        
        if pattern_info is None:
            return None
        
        trough1_idx, trough2_idx, peak_idx, confirmation_level = pattern_info
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_double_bottom_confidence(
            lookback_data, trough1_idx, trough2_idx, peak_idx, confirmation_level
        )
        
        return confidence if confidence >= self.min_confidence else None
    
    def _find_peaks_and_valleys(self, data: pd.DataFrame) -> Tuple[List[int], List[int]]:
        """
        Find local peaks and valleys in the price data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Tuple of (peaks_indices, valleys_indices)
        """
        highs = data['high'].values
        lows = data['low'].values
        
        peaks = []
        valleys = []
        
        # Use a simple peak/valley detection with minimum distance
        min_distance = max(5, len(data) // 20)  # At least 5 periods or 5% of data
        
        # Find peaks (local maxima)
        for i in range(min_distance, len(highs) - min_distance):
            is_peak = True
            current_high = highs[i]
            
            # Check if current point is higher than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and highs[j] >= current_high:
                    is_peak = False
                    break
            
            if is_peak:
                peaks.append(i)
        
        # Find valleys (local minima)
        for i in range(min_distance, len(lows) - min_distance):
            is_valley = True
            current_low = lows[i]
            
            # Check if current point is lower than surrounding points
            for j in range(i - min_distance, i + min_distance + 1):
                if j != i and lows[j] <= current_low:
                    is_valley = False
                    break
            
            if is_valley:
                valleys.append(i)
        
        return peaks, valleys
    
    def _analyze_double_bottom_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                     valleys: List[int]) -> Optional[Tuple[int, int, int, float]]:
        """
        Analyze if the peaks and valleys form a valid double bottom pattern.
        
        Args:
            data: OHLCV DataFrame
            peaks: List of peak indices
            valleys: List of valley indices
            
        Returns:
            Tuple of (trough1_idx, trough2_idx, peak_idx, confirmation_level) or None
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Look for the two lowest troughs
        valley_depths = [(idx, lows[idx]) for idx in valleys]
        valley_depths.sort(key=lambda x: x[1])  # Sort by depth (ascending)
        
        if len(valley_depths) < 2:
            return None
        
        # Try different combinations of the lowest valleys
        for i in range(len(valley_depths) - 1):
            for j in range(i + 1, min(i + 3, len(valley_depths))):  # Check top 3 combinations
                trough1_idx, trough1_depth = valley_depths[i]
                trough2_idx, trough2_depth = valley_depths[j]
                
                # Ensure proper chronological order
                if trough1_idx > trough2_idx:
                    trough1_idx, trough2_idx = trough2_idx, trough1_idx
                    trough1_depth, trough2_depth = trough2_depth, trough1_depth
                
                # Check if troughs are at similar levels (within tolerance)
                depth_diff = abs(trough1_depth - trough2_depth) / max(trough1_depth, trough2_depth)
                if depth_diff > self.trough_tolerance:
                    continue
                
                # Find peak between the troughs
                peaks_between = [p for p in peaks if trough1_idx < p < trough2_idx]
                if not peaks_between:
                    continue
                
                # Take the highest peak between troughs
                peak_idx = max(peaks_between, key=lambda p: highs[p])
                peak_high = highs[peak_idx]
                
                # Check minimum rally requirement
                trough_avg = (trough1_depth + trough2_depth) / 2
                rally_ratio = (peak_high - trough_avg) / trough_avg
                
                if rally_ratio < self.min_peak_rally:
                    continue
                
                # Check for pattern confirmation (price breaking above peak)
                confirmation_level = 0.0
                if len(data) > trough2_idx + 1:
                    post_trough2_highs = highs[trough2_idx + 1:]
                    if len(post_trough2_highs) > 0 and np.max(post_trough2_highs) > peak_high:
                        confirmation_level = 1.0
                    elif closes[-1] > peak_high:
                        confirmation_level = 0.8
                    elif closes[-1] > trough_avg * 1.05:  # Close to breaking resistance
                        confirmation_level = 0.6
                
                return trough1_idx, trough2_idx, peak_idx, confirmation_level
        
        return None
    
    def _calculate_double_bottom_confidence(self, data: pd.DataFrame, trough1_idx: int, 
                                          trough2_idx: int, peak_idx: int, 
                                          confirmation_level: float) -> float:
        """
        Calculate confidence score for the double bottom pattern.
        
        Args:
            data: OHLCV DataFrame
            trough1_idx: Index of first trough
            trough2_idx: Index of second trough
            peak_idx: Index of peak between troughs
            confirmation_level: Level of pattern confirmation
            
        Returns:
            Confidence score
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        criteria_scores = []
        
        # Criterion 1: Trough depth similarity
        trough1_depth = lows[trough1_idx]
        trough2_depth = lows[trough2_idx]
        depth_diff = abs(trough1_depth - trough2_depth) / max(trough1_depth, trough2_depth)
        
        if depth_diff <= 0.005:  # Within 0.5%
            criteria_scores.append(1.0)
        elif depth_diff <= 0.01:  # Within 1%
            criteria_scores.append(0.9)
        elif depth_diff <= 0.02:  # Within 2%
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 2: Peak height
        peak_high = highs[peak_idx]
        trough_avg = (trough1_depth + trough2_depth) / 2
        rally_ratio = (peak_high - trough_avg) / trough_avg
        
        if rally_ratio >= 0.20:  # 20% or more rally
            criteria_scores.append(1.0)
        elif rally_ratio >= 0.15:  # 15% rally
            criteria_scores.append(0.9)
        elif rally_ratio >= 0.10:  # 10% rally
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 3: Time separation between troughs
        time_separation = trough2_idx - trough1_idx
        min_separation = max(10, len(data) // 10)  # At least 10 periods or 10% of data
        
        if time_separation >= min_separation * 2:
            criteria_scores.append(1.0)
        elif time_separation >= min_separation:
            criteria_scores.append(0.8)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 4: Volume analysis (if available)
        if volumes is not None:
            try:
                # Higher volume on second trough recovery suggests stronger buying pressure
                trough2_to_end_volume = np.mean(volumes[trough2_idx:])
                overall_avg_volume = np.mean(volumes)
                
                volume_ratio = trough2_to_end_volume / overall_avg_volume
                if volume_ratio >= 1.5:
                    criteria_scores.append(1.0)
                elif volume_ratio >= 1.2:
                    criteria_scores.append(0.8)
                elif volume_ratio >= 1.0:
                    criteria_scores.append(0.6)
                else:
                    criteria_scores.append(0.4)
            except:
                criteria_scores.append(0.6)  # Neutral score if volume analysis fails
        else:
            criteria_scores.append(0.6)  # Neutral score when no volume data
        
        # Criterion 5: Pattern confirmation
        criteria_scores.append(confirmation_level)
        
        # Criterion 6: Trend context (should occur in downtrend)
        trend_score = self._analyze_trend_context(data, trough1_idx)
        criteria_scores.append(trend_score)
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _analyze_trend_context(self, data: pd.DataFrame, trough1_idx: int) -> float:
        """
        Analyze if the pattern occurs in proper trend context.
        
        Args:
            data: OHLCV DataFrame
            trough1_idx: Index of first trough
            
        Returns:
            Trend context score
        """
        if trough1_idx < 20:
            return 0.5  # Insufficient data
        
        # Look at trend before first trough
        pre_trough_data = data.iloc[:trough1_idx]
        
        # Calculate simple trend using linear regression on closes
        closes = pre_trough_data['close'].values
        x = np.arange(len(closes))
        
        try:
            # Simple linear regression
            slope = np.polyfit(x, closes, 1)[0]
            trend_strength = abs(slope) / np.mean(closes)
            
            # For double bottom, we want prior downtrend
            if slope < 0:  # Downtrend
                if trend_strength >= 0.01:  # Strong downtrend
                    return 1.0
                elif trend_strength >= 0.005:  # Moderate downtrend
                    return 0.8
                else:  # Weak downtrend
                    return 0.6
            else:  # Not in downtrend
                return 0.3
                
        except:
            return 0.5
