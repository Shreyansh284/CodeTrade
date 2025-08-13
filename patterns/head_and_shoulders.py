"""
Head and Shoulders pattern detector implementation.
"""

from typing import Optional, List, Tuple
import pandas as pd
import numpy as np
from .base import BasePatternDetector, PatternResult


class HeadAndShouldersDetector(BasePatternDetector):
    """
    Detector for Head and Shoulders reversal pattern.
    
    Characteristics:
    - Three peaks: left shoulder, head (highest), right shoulder
    - Head is significantly higher than both shoulders
    - Shoulders are at approximately the same level
    - Neckline connects the two valleys between peaks
    - Pattern confirms when price breaks below neckline
    - Indicates potential trend reversal from bullish to bearish
    """
    
    def __init__(self, min_confidence: float = 0.5, shoulder_tolerance: float = 0.05, 
                 head_prominence: float = 0.10, lookback_periods: int = 100,
                 # New tunables
                 min_separation_candles: int = 8,
                 max_triplet_span: int = 120,
                 atr_period: int = 14,
                 atr_prominence_mult: float = 1.0,
                 atr_tolerance_mult: float = 1.5,
                 neckline_slope_tol: float = 0.15):
        """
        Initialize Head and Shoulders detector.
        
        Args:
            min_confidence: Minimum confidence threshold
            shoulder_tolerance: Max percentage difference between shoulders (default 5%)
            head_prominence: Min height difference of head vs shoulders (default 10%)
            lookback_periods: Number of periods to look back for pattern formation
            min_separation_candles: Minimum separation between peaks in triplet
            max_triplet_span: Maximum span from left shoulder to right shoulder
            atr_period: ATR period for adaptive thresholds
            atr_prominence_mult: Min swing prominence in ATR units
            atr_tolerance_mult: Shoulder/valley equality tolerance in ATR units
            neckline_slope_tol: Max relative slope allowed for neckline
        """
        super().__init__(min_confidence)
        self.shoulder_tolerance = shoulder_tolerance
        self.head_prominence = head_prominence
        self.lookback_periods = lookback_periods
        self.min_separation_candles = min_separation_candles
        self.max_triplet_span = max_triplet_span
        self.atr_period = atr_period
        self.atr_prominence_mult = atr_prominence_mult
        self.atr_tolerance_mult = atr_tolerance_mult
        self.neckline_slope_tol = neckline_slope_tol
    
    def get_pattern_name(self) -> str:
        """Return the pattern name."""
        return "Head and Shoulders"
    
    def _get_min_periods(self) -> int:
        """Head and shoulders needs sufficient periods to form."""
        return self.lookback_periods
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect Head and Shoulders pattern at specific index.
        """
        if index >= len(data) or index < self.lookback_periods:
            return None
        
        lookback_data = data.iloc[max(0, index - self.lookback_periods):index + 1]
        if len(lookback_data) < self.lookback_periods:
            return None
        
        atr = self._compute_atr(lookback_data, self.atr_period)
        peaks, valleys = self._find_peaks_and_valleys(lookback_data, atr)
        if len(peaks) < 3 or len(valleys) < 2:
            return None
        
        pattern_info = self._analyze_head_shoulders_pattern(lookback_data, peaks, valleys, atr)
        if pattern_info is None:
            return None
        
        ls, hd, rs, lv, rv, confirmation_level = pattern_info
        confidence = self._calculate_head_shoulders_confidence(
            lookback_data, ls, hd, rs, lv, rv, confirmation_level
        )
        return confidence if confidence >= self.min_confidence else None
    
    def _find_peaks_and_valleys(self, data: pd.DataFrame, atr: np.ndarray) -> Tuple[List[int], List[int]]:
        """Find peaks/valleys using ATR-aware prominence to reduce noise."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        n = len(data)
        min_distance = max(5, n // 20)
        peaks: List[int] = []
        valleys: List[int] = []
        
        for i in range(min_distance, n - min_distance):
            h = highs[i]
            local_max = max(highs[i - min_distance:i].max(initial=h), highs[i+1:i+1+min_distance].max(initial=h))
            if h >= highs[i - min_distance:i + min_distance + 1].max() and (h - local_max) >= self.atr_prominence_mult * atr[i]:
                peaks.append(i)
        
        for i in range(min_distance, n - min_distance):
            l = lows[i]
            local_min = min(lows[i - min_distance:i].min(initial=l), lows[i+1:i+1+min_distance].min(initial=l))
            if l <= lows[i - min_distance:i + min_distance + 1].min() and (local_min - l) >= self.atr_prominence_mult * atr[i]:
                valleys.append(i)
        
        return peaks, valleys
    
    def _analyze_head_shoulders_pattern(self, data: pd.DataFrame, peaks: List[int], 
                                       valleys: List[int], atr: np.ndarray) -> Optional[Tuple[int, int, int, int, int, float]]:
        """Analyze if peaks/valleys form H&S using ATR-aware tolerances and slope checks."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        closes = data['close'].values.astype(float)
        
        for i in range(len(peaks) - 2):
            ls, hd, rs = peaks[i], peaks[i+1], peaks[i+2]
            if (hd - ls) < self.min_separation_candles or (rs - hd) < self.min_separation_candles:
                continue
            if (rs - ls) > self.max_triplet_span:
                continue
            ls_h, hd_h, rs_h = highs[ls], highs[hd], highs[rs]
            if not (hd_h > ls_h and hd_h > rs_h):
                continue
            # Shoulder similarity with ATR tolerance
            shoulder_avg = (ls_h + rs_h) / 2.0
            if abs(ls_h - rs_h) > max(self.shoulder_tolerance * shoulder_avg, self.atr_tolerance_mult * atr[rs]):
                continue
            # Head prominence
            head_prom = (hd_h - shoulder_avg) / max(shoulder_avg, 1e-9)
            if head_prom < self.head_prominence:
                continue
            # Valleys between
            left_vs = [v for v in valleys if ls < v < hd]
            right_vs = [v for v in valleys if hd < v < rs]
            if not left_vs or not right_vs:
                continue
            lv = min(left_vs, key=lambda idx: lows[idx])
            rv = min(right_vs, key=lambda idx: lows[idx])
            lv_low, rv_low = lows[lv], lows[rv]
            neckline = (lv_low + rv_low) / 2.0
            # Neckline slope tolerance (relative)
            slope_rel = abs(rv_low - lv_low) / max(neckline, 1e-9)
            if slope_rel > self.neckline_slope_tol:
                continue
            # Confirmation on closes
            post = closes[rs+1:] if (rs + 1) < len(closes) else np.array([closes[-1]])
            confirmation_level = 0.0
            if post.size > 0 and np.min(post) < neckline:
                confirmation_level = 1.0
            elif closes[-1] < neckline:
                confirmation_level = 0.8
            elif closes[-1] < neckline * 1.02:
                confirmation_level = 0.6
            
            return ls, hd, rs, lv, rv, confirmation_level
        
        return None
    
    def _calculate_head_shoulders_confidence(self, data: pd.DataFrame, left_shoulder_idx: int, 
                                           head_idx: int, right_shoulder_idx: int, 
                                           left_valley_idx: int, right_valley_idx: int, 
                                           confirmation_level: float) -> float:
        """
        Calculate confidence score for the head and shoulders pattern.
        
        Args:
            data: OHLCV DataFrame
            left_shoulder_idx: Index of left shoulder peak
            head_idx: Index of head peak
            right_shoulder_idx: Index of right shoulder peak
            left_valley_idx: Index of left valley
            right_valley_idx: Index of right valley
            confirmation_level: Level of pattern confirmation
            
        Returns:
            Confidence score
        """
        highs = data['high'].values
        lows = data['low'].values
        volumes = data['volume'].values if 'volume' in data.columns else None
        
        criteria_scores = []
        
        # Criterion 1: Shoulder symmetry
        left_shoulder_height = highs[left_shoulder_idx]
        right_shoulder_height = highs[right_shoulder_idx]
        shoulder_diff = abs(left_shoulder_height - right_shoulder_height) / max(left_shoulder_height, right_shoulder_height)
        
        if shoulder_diff <= 0.01:  # Within 1%
            criteria_scores.append(1.0)
        elif shoulder_diff <= 0.03:  # Within 3%
            criteria_scores.append(0.9)
        elif shoulder_diff <= 0.05:  # Within 5%
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 2: Head prominence
        head_height = highs[head_idx]
        shoulder_avg = (left_shoulder_height + right_shoulder_height) / 2
        prominence_ratio = (head_height - shoulder_avg) / shoulder_avg
        
        if prominence_ratio >= 0.20:  # 20% or more higher
            criteria_scores.append(1.0)
        elif prominence_ratio >= 0.15:  # 15% higher
            criteria_scores.append(0.9)
        elif prominence_ratio >= 0.10:  # 10% higher
            criteria_scores.append(0.7)
        else:
            criteria_scores.append(0.5)
        
        # Criterion 3: Valley symmetry (neckline quality)
        left_valley_low = lows[left_valley_idx]
        right_valley_low = lows[right_valley_idx]
        valley_diff = abs(left_valley_low - right_valley_low) / max(left_valley_low, right_valley_low)
        
        if valley_diff <= 0.02:  # Within 2%
            criteria_scores.append(1.0)
        elif valley_diff <= 0.05:  # Within 5%
            criteria_scores.append(0.8)
        elif valley_diff <= 0.10:  # Within 10%
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
                # Typically, volume should be higher on head formation and decline on right shoulder
                head_volume = np.mean(volumes[max(0, head_idx-2):head_idx+3])
                right_shoulder_volume = np.mean(volumes[max(0, right_shoulder_idx-2):right_shoulder_idx+3])
                overall_avg_volume = np.mean(volumes)
                
                head_volume_ratio = head_volume / overall_avg_volume
                right_shoulder_volume_ratio = right_shoulder_volume / overall_avg_volume
                
                # Good pattern: high volume on head, declining volume on right shoulder
                if head_volume_ratio >= 1.2 and right_shoulder_volume_ratio <= 0.8:
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
        
        # Criterion 7: Trend context (should occur after uptrend)
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
            
            # For head and shoulders, we want prior uptrend
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
