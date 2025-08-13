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
                 min_valley_decline: float = 0.10, lookback_periods: int = 50,
                 # New tunables
                 min_separation_candles: int = 8,
                 max_peak_gap_candles: int = 60,
                 atr_period: int = 14,
                 atr_prominence_mult: float = 1.0,
                 atr_tolerance_mult: float = 1.5,
                 atr_move_mult: float = 1.5):
        """
        Initialize Double Top detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            peak_tolerance: Maximum percentage difference between peaks (default 2%)
            min_valley_decline: Minimum decline from peak to valley (default 10%)
            lookback_periods: Number of periods to look back for pattern formation
            min_separation_candles: Minimum separation between the two peaks
            max_peak_gap_candles: Maximum allowed candles between two peaks
            atr_period: ATR period for adaptive thresholds
            atr_prominence_mult: Min swing prominence in ATR units
            atr_tolerance_mult: Peak equality tolerance in ATR units
            atr_move_mult: Required move in ATR units
        """
        super().__init__(min_confidence)
        self.peak_tolerance = peak_tolerance
        self.min_valley_decline = min_valley_decline
        self.lookback_periods = lookback_periods
        self.min_separation_candles = min_separation_candles
        self.max_peak_gap_candles = max_peak_gap_candles
        self.atr_period = atr_period
        self.atr_prominence_mult = atr_prominence_mult
        self.atr_tolerance_mult = atr_tolerance_mult
        self.atr_move_mult = atr_move_mult
    
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
        
        # Compute ATR for adaptive thresholds
        atr = self._compute_atr(lookback_data, self.atr_period)
        
        # Find peaks and valleys in the lookback period (ATR-aware prominence)
        peaks, valleys = self._find_peaks_and_valleys(lookback_data, atr)
        
        if len(peaks) < 2 or len(valleys) < 1:
            return None
        
        # Check for double top pattern
        pattern_info = self._analyze_double_top_pattern(lookback_data, peaks, valleys, atr)
        
        if pattern_info is None:
            return None
        
        peak1_idx, peak2_idx, valley_idx, confirmation_level = pattern_info
        
        # Calculate confidence based on pattern quality
        confidence = self._calculate_double_top_confidence(
            lookback_data, peak1_idx, peak2_idx, valley_idx, confirmation_level
        )
        
        return confidence if confidence >= self.min_confidence else None
    
    def _find_peaks_and_valleys(self, data: pd.DataFrame, atr: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find local peaks and valleys in the price data using ATR-aware prominence.
        """
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        n = len(data)
        peaks: List[int] = []
        valleys: List[int] = []
        
        min_distance = max(5, n // 20)
        
        for i in range(min_distance, n - min_distance):
            h = highs[i]
            # Local window max excluding i
            local_max = max(highs[i - min_distance:i].max(initial=h), highs[i+1:i+1+min_distance].max(initial=h))
            if h >= highs[i - min_distance:i + min_distance + 1].max() and (h - local_max) >= self.atr_prominence_mult * atr[i]:
                peaks.append(i)
        
        for i in range(min_distance, n - min_distance):
            l = lows[i]
            local_min = min(lows[i - min_distance:i].min(initial=l), lows[i+1:i+1+min_distance].min(initial=l))
            if l <= lows[i - min_distance:i + min_distance + 1].min() and (local_min - l) >= self.atr_prominence_mult * atr[i]:
                valleys.append(i)
        
        return peaks, valleys
    
    def _analyze_double_top_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                   valleys: List[int], atr: np.ndarray) -> Optional[Tuple[int, int, int, float]]:
        """
        Analyze if the peaks and valleys form a valid double top pattern using adaptive tolerances.
        """
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        closes = data['close'].values.astype(float)
        
        # Consider combinations of peaks (allow non-adjacent within gap constraints)
        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1, p2 = peaks[i], peaks[j]
                gap = p2 - p1
                if gap < self.min_separation_candles or gap > self.max_peak_gap_candles:
                    continue
                peak1_height, peak2_height = highs[p1], highs[p2]
                peak_avg = (peak1_height + peak2_height) / 2.0
                # Equality tolerance by % or ATR
                if abs(peak1_height - peak2_height) > max(self.peak_tolerance * peak_avg, self.atr_tolerance_mult * atr[p2]):
                    continue
                # Valley between peaks
                mids = [v for v in valleys if p1 < v < p2]
                if not mids:
                    continue
                v = min(mids, key=lambda idx: lows[idx])
                valley_low = lows[v]
                # Sufficient decline by % or ATR
                decline_ok = ((peak_avg - valley_low) / max(peak_avg, 1e-9) >= self.min_valley_decline) or ((peak_avg - valley_low) >= self.atr_move_mult * atr[v])
                if not decline_ok:
                    continue
                # Confirmation: close below valley low after p2
                confirmation_level = 0.0
                post = closes[p2+1:] if (p2 + 1) < len(closes) else np.array([closes[-1]])
                if post.size > 0 and np.min(post) < valley_low:
                    confirmation_level = 1.0
                elif closes[-1] < valley_low:
                    confirmation_level = 0.8
                elif closes[-1] < peak_avg * (1 - 0.05):
                    confirmation_level = 0.6
                
                return p1, p2, v, confirmation_level
        
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
    
    def _compute_atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Compute Wilder's ATR for adaptive thresholds."""
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        prev_close = np.roll(close, 1)
        prev_close[0] = close[0]
        tr = np.maximum.reduce([
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close)
        ])
        atr = np.empty_like(tr)
        atr[:] = np.nan
        if len(tr) == 0:
            return np.zeros(0)
        # Wilder smoothing
        start = min(period - 1, len(tr) - 1)
        atr[start] = np.nanmean(tr[:start+1])
        for i in range(start + 1, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        # Fill initial values
        first_valid = np.where(~np.isnan(atr))[0]
        if first_valid.size:
            atr[:first_valid[0]] = atr[first_valid[0]]
        else:
            atr[:] = np.nanmean(tr) if len(tr) else 0.0
        return atr
