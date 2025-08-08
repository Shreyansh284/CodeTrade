"""
Inverse Head and Shoulders pattern detector implementation.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from .base import BasePatternDetector, PatternResult


class InverseHeadAndShouldersDetector(BasePatternDetector):
    """
    Detector for Inverse Head and Shoulders reversal pattern.
    
    Characteristics:
    - Three troughs: left shoulder, head (lowest), right shoulder
    - Head is significantly lower than both shoulders
    - Shoulders are at approximately the same level
    - Neckline connects the two peaks between troughs
    - Pattern confirms when price breaks above neckline
    - Indicates potential trend reversal from bearish to bullish
    """
    
    def __init__(self, min_confidence: float = 0.5, shoulder_tolerance: float = 0.05, 
                 head_prominence: float = 0.10, lookback_periods: int = 100):
        """
        Initialize Inverse Head and Shoulders detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            shoulder_tolerance: Maximum percentage difference between shoulders (default 5%)
            head_prominence: Minimum depth difference of head vs shoulders (default 10%)
            lookback_periods: Number of periods to look back for pattern formation
        """
        super().__init__(min_confidence)
        self.shoulder_tolerance = shoulder_tolerance
        self.head_prominence = head_prominence
        self.lookback_periods = lookback_periods
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Inverse Head and Shoulders"
    
    def _get_min_periods(self) -> int:
        """Inverse head and shoulders needs sufficient periods to form."""
        return self.lookback_periods
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Inverse Head and Shoulders pattern at specific index.
        
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
        
        if len(valleys) < 3 or len(peaks) < 2:
            return None
        
        # Check for inverse head and shoulders pattern
        pattern_info = self._analyze_inverse_head_shoulders_pattern(lookback_data, peaks, valleys)
        
        if pattern_info is None:
            return None
        
        left_shoulder_idx, head_idx, right_shoulder_idx, left_peak_idx, right_peak_idx, confirmation_level = pattern_info
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_inverse_head_shoulders_confidence(
            lookback_data, left_shoulder_idx, head_idx, right_shoulder_idx, 
            left_peak_idx, right_peak_idx, confirmation_level
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
        
        # Use a minimum distance to avoid noise
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
    
    def _analyze_inverse_head_shoulders_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                              valleys: List[int]) -> Optional[Tuple[int, int, int, int, int, float]]:
        """
        Analyze if the peaks and valleys form a valid inverse head and shoulders pattern.
        
        Args:
            data: OHLCV DataFrame
            peaks: List of peak indices
            valleys: List of valley indices
            
        Returns:
            Tuple of (left_shoulder_idx, head_idx, right_shoulder_idx, left_peak_idx, right_peak_idx, confirmation_level) or None
        """
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        
        # Need at least 3 valleys for inverse head and shoulders
        if len(valleys) < 3:
            return None
        
        # Try different combinations of 3 consecutive valleys
        for i in range(len(valleys) - 2):
            left_shoulder_idx = valleys[i]
            head_idx = valleys[i + 1]
            right_shoulder_idx = valleys[i + 2]
            
            left_shoulder_depth = lows[left_shoulder_idx]
            head_depth = lows[head_idx]
            right_shoulder_depth = lows[right_shoulder_idx]
            
            # Check if head is lower than both shoulders
            if head_depth >= left_shoulder_depth or head_depth >= right_shoulder_depth:
                continue
            
            # Check head prominence (how much deeper the head is)
            shoulder_avg = (left_shoulder_depth + right_shoulder_depth) / 2
            head_prominence_ratio = (shoulder_avg - head_depth) / shoulder_avg
            
            if head_prominence_ratio < self.head_prominence:
                continue
            
            # Check shoulder similarity
            shoulder_diff = abs(left_shoulder_depth - right_shoulder_depth) / max(left_shoulder_depth, right_shoulder_depth)
            if shoulder_diff > self.shoulder_tolerance:
                continue
            
            # Find peaks between valleys
            left_peak_candidates = [p for p in peaks if left_shoulder_idx < p < head_idx]
            right_peak_candidates = [p for p in peaks if head_idx < p < right_shoulder_idx]
            
            if not left_peak_candidates or not right_peak_candidates:
                continue
            
            # Take the highest peak between each pair of valleys
            left_peak_idx = max(left_peak_candidates, key=lambda p: highs[p])
            right_peak_idx = max(right_peak_candidates, key=lambda p: highs[p])
            
            # Calculate neckline level (average of the two peak highs)
            neckline_level = (highs[left_peak_idx] + highs[right_peak_idx]) / 2
            
            # Check for pattern confirmation (price breaking above neckline)
            confirmation_level = 0.0
            if len(data) > right_shoulder_idx + 1:
                post_pattern_highs = highs[right_shoulder_idx + 1:]
                post_pattern_closes = closes[right_shoulder_idx + 1:]
                
                if len(post_pattern_highs) > 0:
                    if np.max(post_pattern_highs) > neckline_level:
                        confirmation_level = 1.0
                    elif len(post_pattern_closes) > 0 and closes[-1] > neckline_level:
                        confirmation_level = 0.8
                    elif len(post_pattern_closes) > 0 and closes[-1] > neckline_level * 0.98:  # Close to breaking
                        confirmation_level = 0.6
            
            return left_shoulder_idx, head_idx, right_shoulder_idx, left_peak_idx, right_peak_idx, confirmation_level
        
        return None
    
    def _calculate_inverse_head_shoulders_confidence(self, data: pd.DataFrame, left_shoulder_idx: int, 
                                                   head_idx: int, right_shoulder_idx: int, 
                                                   left_peak_idx: int, right_peak_idx: int, 
                                                   confirmation_level: float) -> float:
        """
        Calculate confidence score for the inverse head and shoulders pattern.
        
        Args:
            data: OHLCV DataFrame
            left_shoulder_idx: Index of left shoulder valley
            head_idx: Index of head valley
            right_shoulder_idx: Index of right shoulder valley
            left_peak_idx: Index of left peak
            right_peak_idx: Index of right peak
            confirmation_level: Level of pattern confirmation
            
        Returns:
            Confidence score
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        criteria_scores = []
        
        # Criterion 1: Shoulder symmetry
        left_shoulder_depth = lows[left_shoulder_idx]
        right_shoulder_depth = lows[right_shoulder_idx]
        shoulder_diff = abs(left_shoulder_depth - right_shoulder_depth) / max(left_shoulder_depth, right_shoulder_depth)
        
        if shoulder_diff <= 0.01:  # Within 1%
            criteria_scores.append(1.0)
        elif shoulder_diff <= 0.03:  # Within 3%
            criteria_scores.append(0.9)
        elif shoulder_diff <= 0.05:  # Within 5%
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 2: Head prominence
        head_depth = lows[head_idx]
        shoulder_avg = (left_shoulder_depth + right_shoulder_depth) / 2
        prominence_ratio = (shoulder_avg - head_depth) / shoulder_avg
        
        if prominence_ratio >= 0.20:  # 20% or more deeper
            criteria_scores.append(1.0)
        elif prominence_ratio >= 0.15:  # 15% deeper
            criteria_scores.append(0.9)
        elif prominence_ratio >= 0.10:  # 10% deeper
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 3: Peak symmetry (neckline quality)
        left_peak_high = highs[left_peak_idx]
        right_peak_high = highs[right_peak_idx]
        peak_diff = abs(left_peak_high - right_peak_high) / max(left_peak_high, right_peak_high)
        
        if peak_diff <= 0.02:  # Within 2%
            criteria_scores.append(1.0)
        elif peak_diff <= 0.05:  # Within 5%
            criteria_scores.append(0.8)
        elif peak_diff <= 0.10:  # Within 10%
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.4)
        
        # Criterion 4: Time symmetry
        left_time_span = head_idx - left_shoulder_idx
        right_time_span = right_shoulder_idx - head_idx
        time_diff = abs(left_time_span - right_time_span) / max(left_time_span, right_time_span)
        
        if time_diff <= 0.20:  # Within 20%
            criteria_scores.append(1.0)
        elif time_diff <= 0.40:  # Within 40%
            criteria_scores.append(0.8)
        elif time_diff <= 0.60:  # Within 60%
            criteria_scores.append(0.6)
        else:
            criteria_scores.append(0.4)
        
        # Criterion 5: Volume analysis (if available)
        if volumes is not None:
            try:
                # Typically, volume should be higher on head formation and increase on right shoulder
                head_volume = np.mean(volumes[max(0, head_idx-2):head_idx+3])
                right_shoulder_volume = np.mean(volumes[max(0, right_shoulder_idx-2):right_shoulder_idx+3])
                overall_avg_volume = np.mean(volumes)
                
                head_volume_ratio = head_volume / overall_avg_volume
                right_shoulder_volume_ratio = right_shoulder_volume / overall_avg_volume
                
                # Good pattern: high volume on head, increasing volume on right shoulder
                if head_volume_ratio >= 1.2 and right_shoulder_volume_ratio >= 1.1:
                    criteria_scores.append(1.0)
                elif head_volume_ratio >= 1.1:
                    criteria_scores.append(0.8)
                elif head_volume_ratio >= 1.0:
                    criteria_scores.append(0.6)
                else:
                    criteria_scores.append(0.4)
            except:
                criteria_scores.append(0.6)  # Neutral score if volume analysis fails
        else:
            criteria_scores.append(0.6)  # Neutral score when no volume data
        
        # Criterion 6: Pattern confirmation
        criteria_scores.append(confirmation_level)
        
        # Criterion 7: Trend context (should occur after downtrend)
        trend_score = self._analyze_trend_context(data, left_shoulder_idx)
        criteria_scores.append(trend_score)
        
        return self.calculate_confidence_score(criteria_scores)
    
    def _analyze_trend_context(self, data: pd.DataFrame, left_shoulder_idx: int) -> float:
        """
        Analyze if the pattern occurs in proper trend context.
        
        Args:
            data: OHLCV DataFrame
            left_shoulder_idx: Index of left shoulder
            
        Returns:
            Trend context score
        """
        if left_shoulder_idx < 20:
            return 0.5  # Insufficient data
        
        # Look at trend before left shoulder
        pre_pattern_data = data.iloc[:left_shoulder_idx]
        
        # Calculate simple trend using linear regression on closes
        closes = pre_pattern_data['close'].values
        x = np.arange(len(closes))
        
        try:
            # Simple linear regression
            slope = np.polyfit(x, closes, 1)[0]
            trend_strength = abs(slope) / np.mean(closes)
            
            # For inverse head and shoulders, we want prior downtrend
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
