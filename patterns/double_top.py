"""
Double Top pattern detector implementation.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from .base import BasePatternDetector, PatternResult


class DoubleTopDetector(BasePatternDetector):
    """
    Detector for Double Top reversal pattern.
    
    Characteristics:
    - Two distinct peaks at approximately the same level
    - Valley between peaks (at least 10-20% decline from peaks)
    - Second peak occurs after the first with intervening decline
    - Pattern confirms when price breaks below the valley low
    - Indicates potential trend reversal from bullish to bearish
    """
    
    def __init__(self, min_confidence: float = 0.5, peak_tolerance: float = 0.02, 
                 min_valley_decline: float = 0.10, lookback_periods: int = 50):
        """
        Initialize Double Top detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            peak_tolerance: Maximum percentage difference between peaks (default 2%)
            min_valley_decline: Minimum decline from peak to valley (default 10%)
            lookback_periods: Number of periods to look back for pattern formation
        """
        super().__init__(min_confidence)
        self.peak_tolerance = peak_tolerance
        self.min_valley_decline = min_valley_decline
        self.lookback_periods = lookback_periods
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Double Top"
    
    def _get_min_periods(self) -> int:
        """Double top needs sufficient periods to form."""
        return self.lookback_periods
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Double Top pattern at specific index.
        
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
        
        if len(peaks) < 2 or len(valleys) < 1:
            return None
        
        # Check for double top pattern
        pattern_info = self._analyze_double_top_pattern(lookback_data, peaks, valleys)
        
        if pattern_info is None:
            return None
        
        peak1_idx, peak2_idx, valley_idx, confirmation_level = pattern_info
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_double_top_confidence(
            lookback_data, peak1_idx, peak2_idx, valley_idx, confirmation_level
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
    
    def _analyze_double_top_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                   valleys: List[int]) -> Optional[Tuple[int, int, int, float]]:
        """
        Analyze if the peaks and valleys form a valid double top pattern.
        
        Args:
            data: OHLCV DataFrame
            peaks: List of peak indices
            valleys: List of valley indices
            
        Returns:
            Tuple of (peak1_idx, peak2_idx, valley_idx, confirmation_level) or None
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Look for the two highest peaks
        peak_heights = [(idx, highs[idx]) for idx in peaks]
        peak_heights.sort(key=lambda x: x[1], reverse=True)
        
        if len(peak_heights) < 2:
            return None
        
        # Try different combinations of the highest peaks
        for i in range(len(peak_heights) - 1):
            for j in range(i + 1, min(i + 3, len(peak_heights))):  # Check top 3 combinations
                peak1_idx, peak1_height = peak_heights[i]
                peak2_idx, peak2_height = peak_heights[j]
                
                # Ensure proper chronological order
                if peak1_idx > peak2_idx:
                    peak1_idx, peak2_idx = peak2_idx, peak1_idx
                    peak1_height, peak2_height = peak2_height, peak1_height
                
                # Check if peaks are at similar levels (within tolerance)
                height_diff = abs(peak1_height - peak2_height) / max(peak1_height, peak2_height)
                if height_diff > self.peak_tolerance:
                    continue
                
                # Find valley between the peaks
                valleys_between = [v for v in valleys if peak1_idx < v < peak2_idx]
                if not valleys_between:
                    continue
                
                # Take the lowest valley between peaks
                valley_idx = min(valleys_between, key=lambda v: lows[v])
                valley_low = lows[valley_idx]
                
                # Check minimum decline requirement
                peak_avg = (peak1_height + peak2_height) / 2
                decline_ratio = (peak_avg - valley_low) / peak_avg
                
                if decline_ratio < self.min_valley_decline:
                    continue
                
                # Check for pattern confirmation (price breaking below valley)
                confirmation_level = 0.0
                if len(data) > peak2_idx + 1:
                    post_peak2_lows = lows[peak2_idx + 1:]
                    if len(post_peak2_lows) > 0 and np.min(post_peak2_lows) < valley_low:
                        confirmation_level = 1.0
                    elif closes[-1] < valley_low:
                        confirmation_level = 0.8
                    elif closes[-1] < peak_avg * 0.95:  # Close to breaking support
                        confirmation_level = 0.6
                
                return peak1_idx, peak2_idx, valley_idx, confirmation_level
        
        return None
    
    def _calculate_double_top_confidence(self, data: pd.DataFrame, peak1_idx: int, 
                                       peak2_idx: int, valley_idx: int, 
                                       confirmation_level: float) -> float:
        """
        Calculate confidence score for the double top pattern.
        
        Args:
            data: OHLCV DataFrame
            peak1_idx: Index of first peak
            peak2_idx: Index of second peak
            valley_idx: Index of valley between peaks
            confirmation_level: Level of pattern confirmation
            
        Returns:
            Confidence score
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        criteria_scores = []
        
        # Criterion 1: Peak height similarity
        peak1_height = highs[peak1_idx]
        peak2_height = highs[peak2_idx]
        height_diff = abs(peak1_height - peak2_height) / max(peak1_height, peak2_height)
        
        if height_diff <= 0.005:  # Within 0.5%
            criteria_scores.append(1.0)
        elif height_diff <= 0.01:  # Within 1%
            criteria_scores.append(0.9)
        elif height_diff <= 0.02:  # Within 2%
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 2: Valley depth
        valley_low = lows[valley_idx]
        peak_avg = (peak1_height + peak2_height) / 2
        decline_ratio = (peak_avg - valley_low) / peak_avg
        
        if decline_ratio >= 0.20:  # 20% or more decline
            criteria_scores.append(1.0)
        elif decline_ratio >= 0.15:  # 15% decline
            criteria_scores.append(0.9)
        elif decline_ratio >= 0.10:  # 10% decline
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 3: Time separation between peaks
        time_separation = peak2_idx - peak1_idx
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
                # Higher volume on second peak decline suggests stronger selling pressure
                peak2_to_end_volume = np.mean(volumes[peak2_idx:])
                overall_avg_volume = np.mean(volumes)
                
                volume_ratio = peak2_to_end_volume / overall_avg_volume
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
        
        # Criterion 6: Trend context (should occur in uptrend)
        trend_score = self._analyze_trend_context(data, peak1_idx)
        criteria_scores.append(trend_score)
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _analyze_trend_context(self, data: pd.DataFrame, peak1_idx: int) -> float:
        """
        Analyze if the pattern occurs in proper trend context.
        
        Args:
            data: OHLCV DataFrame
            peak1_idx: Index of first peak
            
        Returns:
            Trend context score
        """
        if peak1_idx < 20:
            return 0.5  # Insufficient data
        
        # Look at trend before first peak
        pre_peak_data = data.iloc[:peak1_idx]
        
        # Calculate simple trend using linear regression on closes
        closes = pre_peak_data['close'].values
        x = np.arange(len(closes))
        
        try:
            # Simple linear regression
            slope = np.polyfit(x, closes, 1)[0]
            trend_strength = abs(slope) / np.mean(closes)
            
            # For double top, we want prior uptrend
            if slope > 0:  # Uptrend
                if trend_strength >= 0.01:  # Strong uptrend
                    return 1.0
                elif trend_strength >= 0.005:  # Moderate uptrend
                    return 0.8
                else:  # Weak uptrend
                    return 0.6
            else:  # Not in uptrend
                return 0.3
                
        except:
            return 0.5
