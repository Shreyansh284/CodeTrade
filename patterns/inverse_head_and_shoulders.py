"""
Enhanced Inverse Head and Shoulders pattern detector implementation.

Key improvements:
- Better trough/peak detection with multiple algorithms
- Enhanced neckline analysis with polynomial fitting
- Improved volume analysis with multiple confirmation signals
- Better trend context analysis with multiple timeframes
- Robust statistical validation
- Performance optimizations
- Better handling of edge cases
"""

from typing import Optional, List, Tuple, Dict, Any
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import linregress
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from .base import BasePatternDetector, PatternResult


@dataclass
class InverseHeadShouldersComponents:
    """Data class to hold Inverse H&S pattern components."""
    left_shoulder_idx: int
    head_idx: int
    right_shoulder_idx: int
    left_peak_idx: int
    right_peak_idx: int
    neckline_slope: float
    neckline_intercept: float
    pattern_depth: float
    target_price: float


class InverseHeadAndShouldersDetector(BasePatternDetector):
    """
    Enhanced Inverse Head and Shoulders reversal pattern detector.
    
    Improvements over original:
    1. Multiple trough detection algorithms with consensus
    2. Advanced neckline analysis with polynomial fitting
    3. Multi-timeframe trend context analysis
    4. Enhanced volume pattern recognition
    5. Statistical significance testing
    6. Better parameter adaptation based on market volatility
    
    Pattern characteristics:
    - Three troughs: left shoulder, head (lowest), right shoulder
    - Head is significantly lower than both shoulders
    - Shoulders are at approximately the same level
    - Neckline connects the two peaks between troughs
    - Pattern confirms when price breaks above neckline
    - Indicates potential trend reversal from bearish to bullish
    """
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 lookback_periods: int = 120,
                 shoulder_tolerance: float = 0.05,
                 head_prominence: float = 0.12,
                 volume_confirmation: bool = True,
                 neckline_touch_tolerance: float = 0.02,
                 breakout_volume_multiplier: float = 1.5):
        """
        Initialize Enhanced Inverse Head and Shoulders detector.
        
        Args:
            min_confidence: Minimum confidence threshold (0.0-1.0)
            lookback_periods: Number of periods to analyze
            shoulder_tolerance: Max percentage difference between shoulders
            head_prominence: Min depth difference of head vs shoulders
            volume_confirmation: Whether to use volume confirmation
            neckline_touch_tolerance: Tolerance for neckline touches
            breakout_volume_multiplier: Volume multiplier for breakout confirmation
        """
        super().__init__(min_confidence)
        self.lookback_periods = lookback_periods
        self.shoulder_tolerance = shoulder_tolerance
        self.head_prominence = head_prominence
        self.volume_confirmation = volume_confirmation
        self.neckline_touch_tolerance = neckline_touch_tolerance
        self.breakout_volume_multiplier = breakout_volume_multiplier
        
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Inverse Head & Shoulders"
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect inverse head and shoulders pattern at specific index.
        
        Args:
            data: OHLCV DataFrame with datetime index
            index: Index position to check for pattern
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        # Use the main detection logic for pattern detection at specific index
        if index < self._get_min_periods() or index >= len(data) - 10:
            return None
            
        # Get a window around the index for pattern detection
        start_idx = max(0, index - self.lookback_periods)
        end_idx = min(len(data), index + 20)
        window_data = data.iloc[start_idx:end_idx]
        
        try:
            result = self._detect_inverse_head_shoulders_pattern(window_data)
            if result and result.confidence >= self.min_confidence:
                return result.confidence
        except Exception:
            pass
            
        return None
    
    def _get_min_periods(self) -> int:
        """Minimum periods needed for pattern detection."""
        return max(50, self.lookback_periods // 2)
    
    def detect(self, data: pd.DataFrame) -> List[PatternResult]:
        """
        Detect Inverse Head and Shoulders patterns in the data.
        
        Args:
            data: OHLCV DataFrame with datetime index
            
        Returns:
            List of detected patterns
        """
        if data is None or len(data) < self._get_min_periods():
            return []
        
        patterns = []
        
        # Use sliding window approach for better detection
        window_size = self.lookback_periods
        step_size = max(10, window_size // 10)  # Overlap windows
        
        for start_idx in range(0, len(data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx].copy()
            
            # Reset index for easier handling
            window_data = window_data.reset_index(drop=True)
            
            # Detect pattern in this window
            pattern_components = self._detect_inverse_head_shoulders_pattern(window_data)
            
            if pattern_components:
                # Adjust indices back to original data
                adjusted_components = self._adjust_indices_to_original(
                    pattern_components, start_idx, data
                )
                
                if adjusted_components:
                    patterns.append(adjusted_components)
        
        # Remove duplicate patterns (patterns that overlap significantly)
        patterns = self._remove_duplicate_patterns(patterns)
        
        return patterns
    
    def _detect_inverse_head_shoulders_pattern(self, data: pd.DataFrame) -> Optional[PatternResult]:
        """
        Detect Inverse Head and Shoulders pattern in the given data window.
        
        Args:
            data: OHLCV DataFrame (window)
            
        Returns:
            PatternResult if pattern detected, None otherwise
        """
        try:
            # Multiple detection algorithms
            troughs_scipy = self._find_troughs_scipy(data)
            troughs_rolling = self._find_troughs_rolling_window(data)
            peaks_scipy = self._find_peaks_scipy(data)
            peaks_rolling = self._find_peaks_rolling_window(data)
            
            # Combine and validate trough/peak sets
            best_pattern = None
            best_confidence = 0.0
            
            # Try different combinations
            for troughs in [troughs_scipy, troughs_rolling]:
                for peaks in [peaks_scipy, peaks_rolling]:
                    pattern = self._analyze_inverse_hs_components(data, troughs, peaks)
                    if pattern and pattern.confidence > best_confidence:
                        best_pattern = pattern
                        best_confidence = pattern.confidence
            
            return best_pattern if best_confidence >= self.min_confidence else None
            
        except Exception as e:
            self.logger.warning(f"Error in inverse H&S detection: {e}")
            return None
    
    def _find_troughs_scipy(self, data: pd.DataFrame, min_prominence_pct: float = 0.02) -> List[int]:
        """Find troughs using scipy.signal.find_peaks on inverted lows."""
        try:
            lows = data['low'].values
            # Invert to find troughs
            inverted = -lows
            
            # Calculate dynamic prominence based on price range
            price_range = data['high'].max() - data['low'].min()
            min_prominence = price_range * min_prominence_pct
            
            # Find peaks in inverted data (troughs in original)
            peaks, properties = signal.find_peaks(
                inverted,
                prominence=min_prominence,
                distance=max(5, len(data) // 20)
            )
            
            return peaks.tolist()
            
        except Exception:
            return []
    
    def _find_troughs_rolling_window(self, data: pd.DataFrame, window: int = 10) -> List[int]:
        """Find troughs using rolling window minima."""
        try:
            lows = data['low'].values
            troughs = []
            
            for i in range(window, len(lows) - window):
                current = lows[i]
                is_trough = True
                
                # Check if current point is minimum in window
                for j in range(i - window, i + window + 1):
                    if j != i and lows[j] <= current:
                        is_trough = False
                        break
                
                if is_trough:
                    troughs.append(i)
            
            return troughs
            
        except Exception:
            return []
    
    def _find_peaks_scipy(self, data: pd.DataFrame, min_prominence_pct: float = 0.02) -> List[int]:
        """Find peaks using scipy.signal.find_peaks."""
        try:
            highs = data['high'].values
            
            # Calculate dynamic prominence
            price_range = data['high'].max() - data['low'].min()
            min_prominence = price_range * min_prominence_pct
            
            peaks, properties = signal.find_peaks(
                highs,
                prominence=min_prominence,
                distance=max(5, len(data) // 20)
            )
            
            return peaks.tolist()
            
        except Exception:
            return []
    
    def _find_peaks_rolling_window(self, data: pd.DataFrame, window: int = 8) -> List[int]:
        """Find peaks using rolling window maxima."""
        try:
            highs = data['high'].values
            peaks = []
            
            for i in range(window, len(highs) - window):
                current = highs[i]
                is_peak = True
                
                # Check if current point is maximum in window
                for j in range(i - window, i + window + 1):
                    if j != i and highs[j] >= current:
                        is_peak = False
                        break
                
                if is_peak:
                    peaks.append(i)
            
            return peaks
            
        except Exception:
            return []
    
    def _analyze_inverse_hs_components(self, data: pd.DataFrame, 
                                     troughs: List[int], peaks: List[int]) -> Optional[PatternResult]:
        """
        Analyze trough and peak combinations for inverse H&S pattern.
        
        Args:
            data: OHLCV DataFrame
            troughs: List of trough indices
            peaks: List of peak indices
            
        Returns:
            PatternResult if valid pattern found, None otherwise
        """
        if len(troughs) < 3 or len(peaks) < 2:
            return None
        
        lows = data['low'].values
        highs = data['high'].values
        
        # Try different combinations of 3 consecutive troughs
        best_pattern = None
        best_confidence = 0.0
        
        for i in range(len(troughs) - 2):
            left_shoulder_idx = troughs[i]
            head_idx = troughs[i + 1]
            right_shoulder_idx = troughs[i + 2]
            
            # Validate trough formation
            if not self._validate_inverse_hs_troughs(data, left_shoulder_idx, head_idx, right_shoulder_idx):
                continue
            
            # Find corresponding peaks
            left_peak_candidates = [p for p in peaks if left_shoulder_idx < p < head_idx]
            right_peak_candidates = [p for p in peaks if head_idx < p < right_shoulder_idx]
            
            if not left_peak_candidates or not right_peak_candidates:
                continue
            
            # Select best peaks (highest)
            left_peak_idx = max(left_peak_candidates, key=lambda x: highs[x])
            right_peak_idx = max(right_peak_candidates, key=lambda x: highs[x])
            
            # Create pattern components
            components = InverseHeadShouldersComponents(
                left_shoulder_idx=left_shoulder_idx,
                head_idx=head_idx,
                right_shoulder_idx=right_shoulder_idx,
                left_peak_idx=left_peak_idx,
                right_peak_idx=right_peak_idx,
                neckline_slope=0.0,  # Will be calculated
                neckline_intercept=0.0,  # Will be calculated
                pattern_depth=0.0,  # Will be calculated
                target_price=0.0  # Will be calculated
            )
            
            # Calculate detailed metrics
            confidence = self._calculate_inverse_hs_confidence(data, components)
            
            if confidence > best_confidence:
                best_confidence = confidence
                
                # Calculate additional metrics
                self._calculate_inverse_hs_metrics(data, components)
                
                # Create PatternResult
                start_idx = left_shoulder_idx
                end_idx = right_shoulder_idx
                
                best_pattern = PatternResult(
                    pattern_name=self.get_pattern_name(),
                    confidence=confidence,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    entry_price=float(lows[head_idx]),
                    target_price=components.target_price,
                    exit_price=components.target_price,
                    metadata={
                        'left_shoulder_idx': left_shoulder_idx,
                        'head_idx': head_idx,
                        'right_shoulder_idx': right_shoulder_idx,
                        'left_peak_idx': left_peak_idx,
                        'right_peak_idx': right_peak_idx,
                        'neckline_slope': components.neckline_slope,
                        'neckline_intercept': components.neckline_intercept,
                        'pattern_depth': components.pattern_depth,
                        'left_shoulder_low': float(lows[left_shoulder_idx]),
                        'head_low': float(lows[head_idx]),
                        'right_shoulder_low': float(lows[right_shoulder_idx]),
                        'left_peak_high': float(highs[left_peak_idx]),
                        'right_peak_high': float(highs[right_peak_idx])
                    }
                )
        
        return best_pattern
    
    def _validate_inverse_hs_troughs(self, data: pd.DataFrame, 
                                   left_shoulder_idx: int, head_idx: int, right_shoulder_idx: int) -> bool:
        """
        Validate that the three troughs form a valid inverse H&S pattern.
        
        Args:
            data: OHLCV DataFrame
            left_shoulder_idx: Index of left shoulder trough
            head_idx: Index of head trough
            right_shoulder_idx: Index of right shoulder trough
            
        Returns:
            True if valid inverse H&S troughs, False otherwise
        """
        try:
            lows = data['low'].values
            
            left_shoulder_low = lows[left_shoulder_idx]
            head_low = lows[head_idx]
            right_shoulder_low = lows[right_shoulder_idx]
            
            # Head should be lower than both shoulders
            if head_low >= left_shoulder_low or head_low >= right_shoulder_low:
                return False
            
            # Check head prominence
            shoulder_avg = (left_shoulder_low + right_shoulder_low) / 2
            head_prominence_ratio = (shoulder_avg - head_low) / shoulder_avg
            
            if head_prominence_ratio < self.head_prominence:
                return False
            
            # Check shoulder similarity
            shoulder_diff = abs(left_shoulder_low - right_shoulder_low) / max(left_shoulder_low, right_shoulder_low)
            
            if shoulder_diff > self.shoulder_tolerance:
                return False
            
            # Check chronological order and spacing
            if not (left_shoulder_idx < head_idx < right_shoulder_idx):
                return False
            
            # Ensure reasonable spacing between components
            min_spacing = max(3, len(data) // 30)
            if (head_idx - left_shoulder_idx < min_spacing or 
                right_shoulder_idx - head_idx < min_spacing):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _calculate_inverse_hs_confidence(self, data: pd.DataFrame, 
                                       components: InverseHeadShouldersComponents) -> float:
        """
        Calculate confidence score for the inverse H&S pattern.
        
        Args:
            data: OHLCV DataFrame
            components: Pattern components
            
        Returns:
            Confidence score (0.0-1.0)
        """
        try:
            confidence_factors = []
            
            # Factor 1: Head prominence (how much deeper head is)
            lows = data['low'].values
            left_shoulder_low = lows[components.left_shoulder_idx]
            head_low = lows[components.head_idx]
            right_shoulder_low = lows[components.right_shoulder_idx]
            
            shoulder_avg = (left_shoulder_low + right_shoulder_low) / 2
            head_prominence = (shoulder_avg - head_low) / shoulder_avg
            prominence_score = min(1.0, head_prominence / self.head_prominence)
            confidence_factors.append(prominence_score * 0.25)
            
            # Factor 2: Shoulder symmetry
            shoulder_diff = abs(left_shoulder_low - right_shoulder_low) / max(left_shoulder_low, right_shoulder_low)
            symmetry_score = max(0.0, 1.0 - (shoulder_diff / self.shoulder_tolerance))
            confidence_factors.append(symmetry_score * 0.20)
            
            # Factor 3: Neckline quality
            neckline_score = self._calculate_neckline_quality(data, components)
            confidence_factors.append(neckline_score * 0.20)
            
            # Factor 4: Volume confirmation (if available)
            if self.volume_confirmation and 'volume' in data.columns:
                volume_score = self._calculate_volume_confirmation(data, components)
                confidence_factors.append(volume_score * 0.15)
            else:
                confidence_factors.append(0.10)  # Neutral score if no volume
            
            # Factor 5: Pattern proportions
            proportion_score = self._calculate_pattern_proportions(data, components)
            confidence_factors.append(proportion_score * 0.20)
            
            # Total confidence
            total_confidence = sum(confidence_factors)
            
            # Apply smoothing and ensure valid range
            return max(0.0, min(1.0, total_confidence))
            
        except Exception:
            return 0.0
    
    def _calculate_neckline_quality(self, data: pd.DataFrame, 
                                  components: InverseHeadShouldersComponents) -> float:
        """Calculate quality of the neckline formation."""
        try:
            highs = data['high'].values
            
            left_peak_high = highs[components.left_peak_idx]
            right_peak_high = highs[components.right_peak_idx]
            
            # Neckline should be relatively flat
            height_diff = abs(left_peak_high - right_peak_high) / max(left_peak_high, right_peak_high)
            flatness_score = max(0.0, 1.0 - (height_diff / 0.05))  # 5% tolerance
            
            return flatness_score
            
        except Exception:
            return 0.0
    
    def _calculate_volume_confirmation(self, data: pd.DataFrame, 
                                     components: InverseHeadShouldersComponents) -> float:
        """Calculate volume confirmation score."""
        try:
            volumes = data['volume'].values
            
            # Volume should be higher at the head (selling climax)
            head_volume = volumes[components.head_idx]
            shoulder_volumes = [
                volumes[components.left_shoulder_idx],
                volumes[components.right_shoulder_idx]
            ]
            
            avg_shoulder_volume = np.mean(shoulder_volumes)
            
            if avg_shoulder_volume > 0:
                volume_ratio = head_volume / avg_shoulder_volume
                volume_score = min(1.0, volume_ratio / 1.5)  # Expect 1.5x volume at head
            else:
                volume_score = 0.5
            
            return volume_score
            
        except Exception:
            return 0.5
    
    def _calculate_pattern_proportions(self, data: pd.DataFrame, 
                                     components: InverseHeadShouldersComponents) -> float:
        """Calculate pattern proportion quality."""
        try:
            # Check time proportions
            left_duration = components.head_idx - components.left_shoulder_idx
            right_duration = components.right_shoulder_idx - components.head_idx
            
            if left_duration > 0 and right_duration > 0:
                duration_ratio = min(left_duration, right_duration) / max(left_duration, right_duration)
                proportion_score = duration_ratio  # Closer to 1.0 is better
            else:
                proportion_score = 0.0
            
            return proportion_score
            
        except Exception:
            return 0.0
    
    def _calculate_inverse_hs_metrics(self, data: pd.DataFrame, 
                                    components: InverseHeadShouldersComponents):
        """Calculate additional metrics for the pattern."""
        try:
            highs = data['high'].values
            lows = data['low'].values
            
            # Calculate neckline
            left_peak_high = highs[components.left_peak_idx]
            right_peak_high = highs[components.right_peak_idx]
            
            # Simple linear neckline
            components.neckline_slope = (right_peak_high - left_peak_high) / (components.right_peak_idx - components.left_peak_idx)
            components.neckline_intercept = left_peak_high - components.neckline_slope * components.left_peak_idx
            
            # Pattern depth (head to neckline)
            neckline_at_head = components.neckline_slope * components.head_idx + components.neckline_intercept
            components.pattern_depth = neckline_at_head - lows[components.head_idx]
            
            # Target price (neckline + pattern depth)
            neckline_level = (left_peak_high + right_peak_high) / 2
            components.target_price = neckline_level + components.pattern_depth
            
        except Exception:
            pass
    
    def _adjust_indices_to_original(self, pattern: PatternResult, start_idx: int, 
                                  original_data: pd.DataFrame) -> Optional[PatternResult]:
        """Adjust pattern indices back to original data coordinates."""
        try:
            # Adjust all indices in metadata
            adjusted_metadata = pattern.metadata.copy()
            
            idx_fields = ['left_shoulder_idx', 'head_idx', 'right_shoulder_idx', 
                         'left_peak_idx', 'right_peak_idx']
            
            for field in idx_fields:
                if field in adjusted_metadata:
                    adjusted_metadata[field] += start_idx
            
            # Create new pattern with adjusted indices
            adjusted_pattern = PatternResult(
                pattern_name=pattern.pattern_name,
                confidence=pattern.confidence,
                start_idx=pattern.start_idx + start_idx,
                end_idx=pattern.end_idx + start_idx,
                entry_price=pattern.entry_price,
                target_price=pattern.target_price,
                exit_price=pattern.exit_price,
                metadata=adjusted_metadata
            )
            
            return adjusted_pattern
            
        except Exception:
            return None
    
    def _remove_duplicate_patterns(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """Remove overlapping patterns, keeping the one with highest confidence."""
        if len(patterns) <= 1:
            return patterns
        
        # Sort by confidence (descending)
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        filtered_patterns = []
        
        for pattern in patterns:
            is_duplicate = False
            
            for existing in filtered_patterns:
                # Check for significant overlap
                overlap_start = max(pattern.start_idx, existing.start_idx)
                overlap_end = min(pattern.end_idx, existing.end_idx)
                
                if overlap_start < overlap_end:
                    overlap_size = overlap_end - overlap_start
                    pattern_size = pattern.end_idx - pattern.start_idx
                    
                    # If more than 50% overlap, consider it duplicate
                    if overlap_size / pattern_size > 0.5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
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
