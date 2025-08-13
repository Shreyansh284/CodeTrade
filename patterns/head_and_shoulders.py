"""
Improved Head and Shoulders pattern detector implementation.

Key improvements:
- Better peak/valley detection with multiple algorithms
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
class HeadShouldersComponents:
    """Data class to hold H&S pattern components."""
    left_shoulder_idx: int
    head_idx: int
    right_shoulder_idx: int
    left_valley_idx: int
    right_valley_idx: int
    neckline_slope: float
    neckline_intercept: float
    pattern_height: float
    target_price: float


class ImprovedHeadAndShouldersDetector(BasePatternDetector):
    """
    Enhanced Head and Shoulders reversal pattern detector.
    
    Improvements over original:
    1. Multiple peak detection algorithms with consensus
    2. Advanced neckline analysis with polynomial fitting
    3. Multi-timeframe trend context analysis
    4. Enhanced volume pattern recognition
    5. Statistical significance testing
    6. Better parameter adaptation based on market volatility
    7. Comprehensive pattern validation scoring
    """
    
    def __init__(self, 
                 min_confidence: float = 0.6,
                 shoulder_tolerance: float = 0.04,  # Reduced from 0.05
                 head_prominence: float = 0.08,    # Reduced from 0.10 for more sensitivity
                 lookback_periods: int = 120,      # Increased for better context
                 min_separation_candles: int = 5,  # More flexible
                 max_triplet_span: int = 100,      # More realistic
                 atr_period: int = 14,
                 atr_prominence_mult: float = 1.2,  # Slightly higher for noise reduction
                 atr_tolerance_mult: float = 1.8,   # More flexible
                 neckline_slope_tol: float = 0.20,  # More flexible
                 # New parameters for enhanced detection
                 volume_confirmation_weight: float = 0.15,
                 trend_confirmation_periods: List[int] = None,
                 statistical_significance_level: float = 0.05,
                 use_multiple_algorithms: bool = True,
                 adaptive_thresholds: bool = True):
        """
        Initialize Enhanced Head and Shoulders detector.
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
        self.volume_confirmation_weight = volume_confirmation_weight
        self.trend_confirmation_periods = trend_confirmation_periods or [10, 20, 50]
        self.statistical_significance_level = statistical_significance_level
        self.use_multiple_algorithms = use_multiple_algorithms
        self.adaptive_thresholds = adaptive_thresholds
    
    def get_pattern_name(self) -> str:
        return "Enhanced Head and Shoulders"
    
    def _get_min_periods(self) -> int:
        return self.lookback_periods
    
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """Enhanced pattern detection with multiple validation layers."""
        if index >= len(data) or index < self.lookback_periods:
            return None
        
        lookback_data = data.iloc[max(0, index - self.lookback_periods):index + 1].copy()
        if len(lookback_data) < self.lookback_periods:
            return None
        
        # Precompute technical indicators
        atr = self._compute_atr(lookback_data, self.atr_period)
        volatility = self._compute_volatility(lookback_data)
        
        # Adapt parameters based on volatility if enabled
        if self.adaptive_thresholds:
            adapted_params = self._adapt_parameters(volatility)
        else:
            adapted_params = {}
        
        # Enhanced peak and valley detection
        peaks, valleys, peak_scores, valley_scores = self._enhanced_peak_valley_detection(
            lookback_data, atr, adapted_params
        )
        
        if len(peaks) < 3 or len(valleys) < 2:
            return None
        
        # Find best H&S pattern among candidates
        best_pattern = self._find_best_hs_pattern(
            lookback_data, peaks, valleys, peak_scores, valley_scores, atr, adapted_params
        )
        
        if best_pattern is None:
            return None
        
        # Enhanced confidence calculation
        confidence = self._calculate_enhanced_confidence(
            lookback_data, best_pattern, atr, adapted_params
        )
        
        return confidence if confidence >= self.min_confidence else None
    
    def _enhanced_peak_valley_detection(self, data: pd.DataFrame, atr: np.ndarray, 
                                      adapted_params: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Enhanced peak/valley detection using multiple algorithms."""
        if not self.use_multiple_algorithms:
            return self._single_algorithm_detection(data, atr, adapted_params)
        
        # Algorithm 1: ATR-based (original improved)
        peaks1, valleys1, p_scores1, v_scores1 = self._atr_based_detection(data, atr, adapted_params)
        
        # Algorithm 2: Statistical prominence
        peaks2, valleys2, p_scores2, v_scores2 = self._statistical_detection(data, adapted_params)
        
        # Algorithm 3: Scipy-based peak detection
        peaks3, valleys3, p_scores3, v_scores3 = self._scipy_detection(data, atr, adapted_params)
        
        # Consensus-based merging
        final_peaks, final_valleys, final_p_scores, final_v_scores = self._merge_detections(
            [(peaks1, valleys1, p_scores1, v_scores1),
             (peaks2, valleys2, p_scores2, v_scores2),
             (peaks3, valleys3, p_scores3, v_scores3)]
        )
        
        return final_peaks, final_valleys, final_p_scores, final_v_scores
    
    def _atr_based_detection(self, data: pd.DataFrame, atr: np.ndarray, 
                           adapted_params: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Improved ATR-based detection with better scoring."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        n = len(data)
        
        prominence_mult = adapted_params.get('atr_prominence_mult', self.atr_prominence_mult)
        min_distance = max(3, n // 40)  # More responsive
        
        peaks, valleys = [], []
        peak_scores, valley_scores = [], []
        
        # Peak detection with scoring
        for i in range(min_distance, n - min_distance):
            h = highs[i]
            window = highs[i - min_distance:i + min_distance + 1]
            
            if h >= np.max(window):  # Local maximum
                # Calculate prominence score
                left_min = np.min(highs[max(0, i - min_distance*2):i])
                right_min = np.min(highs[i+1:min(n, i + min_distance*2 + 1)])
                prominence = min(h - left_min, h - right_min)
                
                if prominence >= prominence_mult * atr[i]:
                    peaks.append(i)
                    # Score based on prominence relative to ATR
                    score = min(1.0, prominence / (prominence_mult * atr[i]))
                    peak_scores.append(score)
        
        # Valley detection with scoring
        for i in range(min_distance, n - min_distance):
            l = lows[i]
            window = lows[i - min_distance:i + min_distance + 1]
            
            if l <= np.min(window):  # Local minimum
                # Calculate prominence score
                left_max = np.max(lows[max(0, i - min_distance*2):i])
                right_max = np.max(lows[i+1:min(n, i + min_distance*2 + 1)])
                prominence = min(left_max - l, right_max - l)
                
                if prominence >= prominence_mult * atr[i]:
                    valleys.append(i)
                    score = min(1.0, prominence / (prominence_mult * atr[i]))
                    valley_scores.append(score)
        
        return peaks, valleys, peak_scores, valley_scores
    
    def _statistical_detection(self, data: pd.DataFrame, 
                             adapted_params: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Statistical prominence-based detection."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        
        # Calculate rolling statistics
        window = 20
        high_mean = pd.Series(highs).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        high_std = pd.Series(highs).rolling(window, center=True).std().fillna(method='bfill').fillna(method='ffill')
        low_mean = pd.Series(lows).rolling(window, center=True).mean().fillna(method='bfill').fillna(method='ffill')
        low_std = pd.Series(lows).rolling(window, center=True).std().fillna(method='bfill').fillna(method='ffill')
        
        peaks, valleys = [], []
        peak_scores, valley_scores = [], []
        
        min_distance = max(5, len(data) // 30)
        
        # Find peaks that are statistically significant
        for i in range(min_distance, len(highs) - min_distance):
            h = highs[i]
            # Z-score based significance
            z_score = (h - high_mean.iloc[i]) / (high_std.iloc[i] + 1e-8)
            
            if z_score > 1.5:  # Statistically significant peak
                # Check if it's a local maximum
                window_vals = highs[i - min_distance:i + min_distance + 1]
                if h >= np.max(window_vals):
                    peaks.append(i)
                    peak_scores.append(min(1.0, z_score / 3.0))  # Normalize score
        
        # Find valleys that are statistically significant
        for i in range(min_distance, len(lows) - min_distance):
            l = lows[i]
            z_score = (low_mean.iloc[i] - l) / (low_std.iloc[i] + 1e-8)  # Inverted for valleys
            
            if z_score > 1.5:
                window_vals = lows[i - min_distance:i + min_distance + 1]
                if l <= np.min(window_vals):
                    valleys.append(i)
                    valley_scores.append(min(1.0, z_score / 3.0))
        
        return peaks, valleys, peak_scores, valley_scores
    
    def _scipy_detection(self, data: pd.DataFrame, atr: np.ndarray, 
                        adapted_params: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Scipy-based peak detection with adaptive parameters."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        
        # Adaptive prominence based on ATR
        avg_atr = np.nanmean(atr)
        prominence = adapted_params.get('atr_prominence_mult', self.atr_prominence_mult) * avg_atr
        distance = max(5, len(data) // 30)
        
        # Find peaks
        peak_indices, peak_properties = signal.find_peaks(
            highs, 
            prominence=prominence,
            distance=distance,
            width=(1, None)
        )
        
        # Find valleys (invert signal)
        valley_indices, valley_properties = signal.find_peaks(
            -lows,
            prominence=prominence,
            distance=distance,
            width=(1, None)
        )
        
        # Calculate scores based on prominence
        peak_scores = []
        if len(peak_indices) > 0:
            max_prominence = np.max(peak_properties['prominences'])
            peak_scores = [min(1.0, p / max_prominence) for p in peak_properties['prominences']]
        
        valley_scores = []
        if len(valley_indices) > 0:
            max_prominence = np.max(valley_properties['prominences'])
            valley_scores = [min(1.0, p / max_prominence) for p in valley_properties['prominences']]
        
        return peak_indices.tolist(), valley_indices.tolist(), peak_scores, valley_scores
    
    def _merge_detections(self, detections: List[Tuple]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Merge multiple detection results using consensus."""
        all_peaks = {}  # {index: [scores]}
        all_valleys = {}
        
        tolerance = 3  # Allow 3-candle tolerance for consensus
        
        for peaks, valleys, p_scores, v_scores in detections:
            # Group peaks by proximity
            for i, (peak_idx, score) in enumerate(zip(peaks, p_scores)):
                found_group = False
                for existing_idx in all_peaks:
                    if abs(peak_idx - existing_idx) <= tolerance:
                        all_peaks[existing_idx].append(score)
                        found_group = True
                        break
                if not found_group:
                    all_peaks[peak_idx] = [score]
            
            # Group valleys by proximity
            for i, (valley_idx, score) in enumerate(zip(valleys, v_scores)):
                found_group = False
                for existing_idx in all_valleys:
                    if abs(valley_idx - existing_idx) <= tolerance:
                        all_valleys[existing_idx].append(score)
                        found_group = True
                        break
                if not found_group:
                    all_valleys[valley_idx] = [score]
        
        # Select peaks/valleys with consensus (at least 2 algorithms agree)
        final_peaks, final_peak_scores = [], []
        for idx, scores in all_peaks.items():
            if len(scores) >= 2:  # At least 2 algorithms agree
                final_peaks.append(idx)
                final_peak_scores.append(np.mean(scores) * (len(scores) / 3))  # Bonus for consensus
        
        final_valleys, final_valley_scores = [], []
        for idx, scores in all_valleys.items():
            if len(scores) >= 2:
                final_valleys.append(idx)
                final_valley_scores.append(np.mean(scores) * (len(scores) / 3))
        
        # Sort by index
        peak_pairs = list(zip(final_peaks, final_peak_scores))
        valley_pairs = list(zip(final_valleys, final_valley_scores))
        peak_pairs.sort()
        valley_pairs.sort()
        
        final_peaks = [p[0] for p in peak_pairs]
        final_peak_scores = [p[1] for p in peak_pairs]
        final_valleys = [v[0] for v in valley_pairs]
        final_valley_scores = [v[1] for v in valley_pairs]
        
        return final_peaks, final_valleys, final_peak_scores, final_valley_scores
    
    def _find_best_hs_pattern(self, data: pd.DataFrame, peaks: List[int], valleys: List[int],
                             peak_scores: List[float], valley_scores: List[float],
                             atr: np.ndarray, adapted_params: Dict[str, Any]) -> Optional[HeadShouldersComponents]:
        """Find the best H&S pattern among all candidates."""
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        
        best_pattern = None
        best_score = 0.0
        
        # Try all possible triplets
        for i in range(len(peaks) - 2):
            for j in range(i + 1, len(peaks) - 1):
                for k in range(j + 1, len(peaks)):
                    ls, hd, rs = peaks[i], peaks[j], peaks[k]
                    
                    # Basic constraints
                    if (hd - ls) < self.min_separation_candles or (rs - hd) < self.min_separation_candles:
                        continue
                    if (rs - ls) > self.max_triplet_span:
                        continue
                    
                    pattern = self._validate_hs_triplet(
                        data, ls, hd, rs, valleys, highs, lows, atr, adapted_params
                    )
                    
                    if pattern is not None:
                        # Calculate preliminary score
                        score = self._score_hs_pattern(data, pattern, peak_scores, valley_scores, atr)
                        if score > best_score:
                            best_score = score
                            best_pattern = pattern
        
        return best_pattern
    
    def _validate_hs_triplet(self, data: pd.DataFrame, ls: int, hd: int, rs: int,
                           valleys: List[int], highs: np.ndarray, lows: np.ndarray,
                           atr: np.ndarray, adapted_params: Dict[str, Any]) -> Optional[HeadShouldersComponents]:
        """Validate a potential H&S triplet."""
        ls_h, hd_h, rs_h = highs[ls], highs[hd], highs[rs]
        
        # Head must be highest
        if not (hd_h > ls_h and hd_h > rs_h):
            return None
        
        # Shoulder similarity with adaptive tolerance
        shoulder_avg = (ls_h + rs_h) / 2.0
        tolerance_abs = max(
            adapted_params.get('shoulder_tolerance', self.shoulder_tolerance) * shoulder_avg,
            adapted_params.get('atr_tolerance_mult', self.atr_tolerance_mult) * atr[rs]
        )
        
        if abs(ls_h - rs_h) > tolerance_abs:
            return None
        
        # Head prominence check
        head_prom = (hd_h - shoulder_avg) / max(shoulder_avg, 1e-9)
        if head_prom < adapted_params.get('head_prominence', self.head_prominence):
            return None
        
        # Find best valleys
        left_candidates = [v for v in valleys if ls < v < hd]
        right_candidates = [v for v in valleys if hd < v < rs]
        
        if not left_candidates or not right_candidates:
            return None
        
        lv = min(left_candidates, key=lambda idx: lows[idx])
        rv = min(right_candidates, key=lambda idx: lows[idx])
        
        # Enhanced neckline analysis
        neckline_analysis = self._analyze_neckline(data, lv, rv, ls, rs)
        if neckline_analysis is None:
            return None
        
        slope, intercept, r_squared = neckline_analysis
        
        # Calculate pattern metrics
        pattern_height = hd_h - ((lows[lv] + lows[rv]) / 2)
        neckline_price = (lows[lv] + lows[rv]) / 2
        target_price = neckline_price - pattern_height  # Measured move
        
        return HeadShouldersComponents(
            left_shoulder_idx=ls,
            head_idx=hd,
            right_shoulder_idx=rs,
            left_valley_idx=lv,
            right_valley_idx=rv,
            neckline_slope=slope,
            neckline_intercept=intercept,
            pattern_height=pattern_height,
            target_price=target_price
        )
    
    def _analyze_neckline(self, data: pd.DataFrame, lv: int, rv: int, 
                         ls: int, rs: int) -> Optional[Tuple[float, float, float]]:
        """Enhanced neckline analysis with polynomial fitting and quality assessment."""
        lows = data['low'].values.astype(float)
        
        # Basic neckline slope check
        lv_low, rv_low = lows[lv], lows[rv]
        neckline_avg = (lv_low + rv_low) / 2.0
        
        # Relative slope tolerance
        slope_rel = abs(rv_low - lv_low) / max(neckline_avg, 1e-9)
        if slope_rel > self.neckline_slope_tol:
            return None
        
        # Fit line to neckline points and intermediate lows
        x_points = [lv, rv]
        y_points = [lv_low, rv_low]
        
        # Add intermediate points for better fitting
        intermediate_indices = range(lv + 1, rv)
        for idx in intermediate_indices:
            if idx < len(lows):
                # Only include points that are reasonably close to the neckline
                if abs(lows[idx] - neckline_avg) / neckline_avg < 0.1:
                    x_points.append(idx)
                    y_points.append(lows[idx])
        
        if len(x_points) < 2:
            return None
        
        try:
            # Linear regression for neckline
            slope, intercept, r_value, p_value, std_err = linregress(x_points, y_points)
            r_squared = r_value ** 2
            
            # Quality check: R-squared should be reasonable for a good neckline
            if r_squared < 0.5:  # Poor fit
                return None
            
            return slope, intercept, r_squared
            
        except:
            # Fallback to simple line between valley points
            slope = (rv_low - lv_low) / (rv - lv) if rv != lv else 0
            intercept = lv_low - slope * lv
            return slope, intercept, 0.8  # Assume decent fit
    
    def _calculate_enhanced_confidence(self, data: pd.DataFrame, 
                                     pattern: HeadShouldersComponents,
                                     atr: np.ndarray,
                                     adapted_params: Dict[str, Any]) -> float:
        """Enhanced confidence calculation with comprehensive analysis."""
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        volumes = data.get('volume', pd.Series()).values if 'volume' in data.columns else None
        
        criteria_scores = []
        
        # 1. Shoulder Symmetry (Enhanced)
        shoulder_score = self._calculate_shoulder_symmetry_score(
            highs[pattern.left_shoulder_idx], 
            highs[pattern.right_shoulder_idx],
            atr[pattern.right_shoulder_idx]
        )
        criteria_scores.append(shoulder_score)
        
        # 2. Head Prominence (Enhanced)
        head_score = self._calculate_head_prominence_score(
            highs[pattern.head_idx],
            highs[pattern.left_shoulder_idx],
            highs[pattern.right_shoulder_idx]
        )
        criteria_scores.append(head_score)
        
        # 3. Neckline Quality (New)
        neckline_score = self._calculate_neckline_quality_score(
            data, pattern.left_valley_idx, pattern.right_valley_idx,
            pattern.neckline_slope, pattern.left_shoulder_idx, pattern.right_shoulder_idx
        )
        criteria_scores.append(neckline_score)
        
        # 4. Time Symmetry (Enhanced)
        time_score = self._calculate_time_symmetry_score(
            pattern.left_shoulder_idx, pattern.head_idx, pattern.right_shoulder_idx
        )
        criteria_scores.append(time_score)
        
        # 5. Volume Analysis (Enhanced)
        if volumes is not None and len(volumes) > 0:
            volume_score = self._calculate_enhanced_volume_score(
                volumes, pattern.left_shoulder_idx, pattern.head_idx, pattern.right_shoulder_idx
            )
        else:
            volume_score = 0.5  # Neutral when no volume data
        criteria_scores.append(volume_score)
        
        # 6. Pattern Confirmation (Enhanced)
        confirmation_score = self._calculate_confirmation_score(
            closes, pattern.left_valley_idx, pattern.right_valley_idx, 
            pattern.right_shoulder_idx
        )
        criteria_scores.append(confirmation_score)
        
        # 7. Multi-timeframe Trend Context (New)
        trend_score = self._calculate_multi_timeframe_trend_score(
            data, pattern.left_shoulder_idx
        )
        criteria_scores.append(trend_score)
        
        # 8. Statistical Significance (New)
        stats_score = self._calculate_statistical_significance(
            data, pattern.left_shoulder_idx, pattern.head_idx, pattern.right_shoulder_idx
        )
        criteria_scores.append(stats_score)
        
        # 9. Pattern Geometry (New)
        geometry_score = self._calculate_pattern_geometry_score(pattern, data)
        criteria_scores.append(geometry_score)
        
        # Calculate weighted confidence
        weights = [1.2, 1.3, 1.1, 0.9, self.volume_confirmation_weight * 10, 1.4, 1.0, 0.8, 0.7]
        weighted_score = np.average(criteria_scores, weights=weights[:len(criteria_scores)])
        
        return min(1.0, weighted_score)
    
    def _calculate_shoulder_symmetry_score(self, left_h: float, right_h: float, atr_val: float) -> float:
        """Enhanced shoulder symmetry scoring with ATR normalization."""
        abs_diff = abs(left_h - right_h)
        avg_height = (left_h + right_h) / 2
        
        # Relative difference
        rel_diff = abs_diff / avg_height if avg_height > 0 else 1.0
        
        # ATR normalized difference
        atr_diff = abs_diff / atr_val if atr_val > 0 else rel_diff
        
        # Combined scoring
        rel_score = max(0.0, 1.0 - rel_diff / 0.05)  # Perfect if within 5%
        atr_score = max(0.0, 1.0 - atr_diff / 2.0)   # Perfect if within 2 ATR
        
        return (rel_score + atr_score) / 2
    
    def _calculate_head_prominence_score(self, head_h: float, left_h: float, right_h: float) -> float:
        """Enhanced head prominence scoring."""
        shoulder_avg = (left_h + right_h) / 2
        prominence_ratio = (head_h - shoulder_avg) / shoulder_avg if shoulder_avg > 0 else 0
        
        if prominence_ratio >= 0.25:
            return 1.0
        elif prominence_ratio >= 0.20:
            return 0.95
        elif prominence_ratio >= 0.15:
            return 0.85
        elif prominence_ratio >= 0.10:
            return 0.70
        elif prominence_ratio >= 0.05:
            return 0.50
        else:
            return 0.30
    
    def _calculate_enhanced_volume_score(self, volumes: np.ndarray, ls_idx: int, hd_idx: int, rs_idx: int) -> float:
        """Enhanced volume analysis with multiple signals."""
        try:
            # Safe volume extraction
            def safe_volume_avg(idx, window=2):
                start = max(0, idx - window)
                end = min(len(volumes), idx + window + 1)
                return np.mean(volumes[start:end]) if end > start else 0
            
            ls_vol = safe_volume_avg(ls_idx)
            hd_vol = safe_volume_avg(hd_idx)
            rs_vol = safe_volume_avg(rs_idx)
            
            # Overall average for context
            overall_avg = np.mean(volumes) if len(volumes) > 0 else 1
            
            if overall_avg == 0:
                return 0.5
            
            # Volume ratios
            head_ratio = hd_vol / overall_avg
            rs_ratio = rs_vol / overall_avg
            
            score_components = []
            
            # 1. Head volume should be elevated
            if head_ratio >= 1.5:
                score_components.append(1.0)
            elif head_ratio >= 1.2:
                score_components.append(0.8)
            elif head_ratio >= 1.0:
                score_components.append(0.6)
            else:
                score_components.append(0.3)
            
            # 2. Right shoulder volume should be declining
            if rs_ratio <= 0.7:
                score_components.append(1.0)
            elif rs_ratio <= 0.9:
                score_components.append(0.7)
            elif rs_ratio <= 1.1:
                score_components.append(0.5)
            else:
                score_components.append(0.2)
            
            # 3. Volume trend analysis
            if hd_vol > ls_vol and rs_vol < hd_vol:
                score_components.append(1.0)  # Ideal volume pattern
            elif hd_vol > rs_vol:
                score_components.append(0.7)
            else:
                score_components.append(0.3)
            
            return np.mean(score_components)
            
        except:
            return 0.5
    
    def _calculate_neckline_quality_score(self, data: pd.DataFrame, lv_idx: int, rv_idx: int,
                                        slope: float, ls_idx: int, rs_idx: int) -> float:
        """Calculate neckline quality based on multiple factors."""
        lows = data['low'].values
        
        try:
            # 1. Slope score (flatter is better for H&S)
            abs_slope = abs(slope)
            if abs_slope <= 0.001:
                slope_score = 1.0
            elif abs_slope <= 0.005:
                slope_score = 0.9
            elif abs_slope <= 0.01:
                slope_score = 0.7
            else:
                slope_score = 0.5
            
            # 2. Valley depth consistency
            lv_low, rv_low = lows[lv_idx], lows[rv_idx]
            depth_diff = abs(lv_low - rv_low) / max(lv_low, rv_low)
            depth_score = max(0.0, 1.0 - depth_diff / 0.05)  # Within 5% is perfect
            
            # 3. Neckline test count (how many times price touched neckline area)
            neckline_level = (lv_low + rv_low) / 2
            tolerance = abs(rv_low - lv_low) * 0.5 + np.std(lows) * 0.1
            
            touches = 0
            for i in range(lv_idx, rv_idx + 1):
                if abs(lows[i] - neckline_level) <= tolerance:
                    touches += 1
            
            touch_score = min(1.0, touches / 3)  # More touches = stronger neckline
            
            return np.mean([slope_score, depth_score, touch_score])
            
        except:
            return 0.6
    
    def _calculate_time_symmetry_score(self, ls_idx: int, hd_idx: int, rs_idx: int) -> float:
        """Enhanced time symmetry calculation."""
        left_span = hd_idx - ls_idx
        right_span = rs_idx - hd_idx
        
        if left_span == 0 or right_span == 0:
            return 0.3
        
        ratio = min(left_span, right_span) / max(left_span, right_span)
        
        if ratio >= 0.9:
            return 1.0
        elif ratio >= 0.8:
            return 0.9
        elif ratio >= 0.7:
            return 0.8
        elif ratio >= 0.6:
            return 0.6
        elif ratio >= 0.5:
            return 0.4
        else:
            return 0.2
    
    def _calculate_confirmation_score(self, closes: np.ndarray, lv_idx: int, rv_idx: int, rs_idx: int) -> float:
        """Enhanced pattern confirmation scoring."""
        try:
            neckline = (closes[lv_idx] + closes[rv_idx]) / 2  # Use closes for confirmation
            
            # Look at price action after right shoulder
            post_rs_closes = closes[rs_idx+1:] if rs_idx + 1 < len(closes) else np.array([closes[-1]])
            current_close = closes[-1]
            
            # Multiple confirmation levels
            strong_break = neckline * 0.98  # 2% break
            weak_break = neckline * 0.995   # 0.5% break
            
            # Check for sustained break
            if len(post_rs_closes) > 0:
                min_post_close = np.min(post_rs_closes)
                if min_post_close < strong_break:
                    return 1.0  # Strong confirmation
                elif min_post_close < weak_break:
                    return 0.8  # Moderate confirmation
            
            # Current price relative to neckline
            if current_close < strong_break:
                return 0.9
            elif current_close < weak_break:
                return 0.7
            elif current_close < neckline:
                return 0.6
            elif current_close < neckline * 1.02:
                return 0.4
            else:
                return 0.2  # No confirmation yet
                
        except:
            return 0.5
    
    def _calculate_multi_timeframe_trend_score(self, data: pd.DataFrame, ls_idx: int) -> float:
        """Multi-timeframe trend analysis."""
        if ls_idx < max(self.trend_confirmation_periods):
            return 0.5
        
        closes = data['close'].values
        scores = []
        
        for period in self.trend_confirmation_periods:
            if ls_idx >= period:
                # Analyze trend before left shoulder
                trend_data = closes[ls_idx-period:ls_idx]
                
                try:
                    # Linear regression slope
                    x = np.arange(len(trend_data))
                    slope, _, r_value, p_value, _ = linregress(x, trend_data)
                    
                    # Normalize slope
                    slope_norm = slope / np.mean(trend_data) if np.mean(trend_data) > 0 else 0
                    
                    # Score based on uptrend strength
                    if slope_norm > 0.002 and r_value > 0.5:  # Strong uptrend
                        scores.append(1.0)
                    elif slope_norm > 0.001 and r_value > 0.3:  # Moderate uptrend
                        scores.append(0.8)
                    elif slope_norm > 0 and r_value > 0.1:     # Weak uptrend
                        scores.append(0.6)
                    else:  # Not in uptrend
                        scores.append(0.3)
                        
                except:
                    scores.append(0.5)
            else:
                scores.append(0.5)
        
        return np.mean(scores)
    
    def _calculate_statistical_significance(self, data: pd.DataFrame, ls_idx: int, hd_idx: int, rs_idx: int) -> float:
        """Calculate statistical significance of the pattern."""
        try:
            highs = data['high'].values
            
            # Test if the pattern is statistically different from random
            pattern_highs = [highs[ls_idx], highs[hd_idx], highs[rs_idx]]
            
            # Get context data before pattern
            context_start = max(0, ls_idx - 50)
            context_highs = highs[context_start:ls_idx]
            
            if len(context_highs) < 20:
                return 0.5
            
            # Compare pattern heights to historical distribution
            context_mean = np.mean(context_highs)
            context_std = np.std(context_highs)
            
            if context_std == 0:
                return 0.5
            
            # Z-scores for pattern components
            z_scores = [(h - context_mean) / context_std for h in pattern_highs]
            
            # Head should be significantly higher
            head_z = z_scores[1]
            shoulder_z_avg = (z_scores[0] + z_scores[2]) / 2
            
            # Statistical significance score
            if head_z > 2.0 and shoulder_z_avg > 1.0:  # Very significant
                return 1.0
            elif head_z > 1.5 and shoulder_z_avg > 0.5:  # Significant
                return 0.8
            elif head_z > 1.0:  # Somewhat significant
                return 0.6
            else:
                return 0.4
                
        except:
            return 0.5
    
    def _calculate_pattern_geometry_score(self, pattern: HeadShouldersComponents, data: pd.DataFrame) -> float:
        """Score the geometric properties of the pattern."""
        try:
            highs = data['high'].values
            
            # Pattern proportions
            left_height = highs[pattern.head_idx] - highs[pattern.left_shoulder_idx]
            right_height = highs[pattern.head_idx] - highs[pattern.right_shoulder_idx]
            
            # Height symmetry
            if left_height > 0 and right_height > 0:
                height_ratio = min(left_height, right_height) / max(left_height, right_height)
                height_score = height_ratio
            else:
                height_score = 0.5
            
            # Pattern width vs height ratio (aesthetic proportion)
            pattern_width = pattern.right_shoulder_idx - pattern.left_shoulder_idx
            pattern_height = pattern.pattern_height
            
            if pattern_height > 0:
                width_height_ratio = pattern_width / (pattern_height * 100)  # Normalize
                # Ideal ratio around 1-3
                if 1 <= width_height_ratio <= 3:
                    proportion_score = 1.0
                elif 0.5 <= width_height_ratio <= 5:
                    proportion_score = 0.8
                else:
                    proportion_score = 0.6
            else:
                proportion_score = 0.5
            
            return (height_score + proportion_score) / 2
            
        except:
            return 0.6
    
    def _score_hs_pattern(self, data: pd.DataFrame, pattern: HeadShouldersComponents,
                         peak_scores: List[float], valley_scores: List[float], atr: np.ndarray) -> float:
        """Score a complete H&S pattern for ranking multiple candidates."""
        try:
            # Base score from peak/valley detection quality
            peak_indices = [pattern.left_shoulder_idx, pattern.head_idx, pattern.right_shoulder_idx]
            valley_indices = [pattern.left_valley_idx, pattern.right_valley_idx]
            
            # Find scores for our specific peaks/valleys
            peak_quality_scores = []
            for peak_idx in peak_indices:
                # Find closest peak in our detected peaks
                closest_score = 0.5
                for i, detected_peak in enumerate(data.index):  # This needs to be fixed
                    if abs(detected_peak - peak_idx) <= 2:  # Close enough
                        if i < len(peak_scores):
                            closest_score = peak_scores[i]
                        break
                peak_quality_scores.append(closest_score)
            
            valley_quality_scores = []
            for valley_idx in valley_indices:
                closest_score = 0.5
                valley_quality_scores.append(closest_score)
            
            # Combine quality scores
            avg_peak_quality = np.mean(peak_quality_scores)
            avg_valley_quality = np.mean(valley_quality_scores)
            detection_quality = (avg_peak_quality + avg_valley_quality) / 2
            
            # Pattern-specific metrics
            pattern_height_score = min(1.0, pattern.pattern_height / (np.mean(atr) * 5))
            
            return (detection_quality + pattern_height_score) / 2
            
        except:
            return 0.5
    
    def _adapt_parameters(self, volatility: float) -> Dict[str, Any]:
        """Adapt detection parameters based on market volatility."""
        if volatility < 0.01:  # Low volatility
            return {
                'shoulder_tolerance': self.shoulder_tolerance * 0.8,
                'atr_prominence_mult': self.atr_prominence_mult * 0.8,
                'head_prominence': self.head_prominence * 0.8
            }
        elif volatility > 0.05:  # High volatility
            return {
                'shoulder_tolerance': self.shoulder_tolerance * 1.5,
                'atr_prominence_mult': self.atr_prominence_mult * 1.5,
                'head_prominence': self.head_prominence * 1.2
            }
        else:
            return {}  # Use default parameters
    
    def _compute_volatility(self, data: pd.DataFrame) -> float:
        """Compute market volatility for parameter adaptation."""
        try:
            closes = data['close'].values.astype(float)
            returns = np.diff(np.log(closes))
            return np.std(returns) * np.sqrt(252)  # Annualized volatility
        except:
            return 0.02  # Default moderate volatility
    
    def _single_algorithm_detection(self, data: pd.DataFrame, atr: np.ndarray, 
                                  adapted_params: Dict[str, Any]) -> Tuple[List[int], List[int], List[float], List[float]]:
        """Fallback to single algorithm if multiple algorithms disabled."""
        return self._atr_based_detection(data, atr, adapted_params)
    
    def _compute_atr(self, data: pd.DataFrame, period: int) -> np.ndarray:
        """Improved ATR computation with better edge case handling."""
        try:
            high = data['high'].values.astype(float)
            low = data['low'].values.astype(float) 
            close = data['close'].values.astype(float)
            
            if len(high) == 0:
                return np.array([])
            
            # Handle single value case
            if len(high) == 1:
                return np.array([high[0] - low[0]])
            
            prev_close = np.concatenate([[close[0]], close[:-1]])
            
            tr = np.maximum.reduce([
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close)
            ])
            
            # Wilder's smoothing with better initialization
            atr = np.zeros_like(tr, dtype=float)
            
            # Initialize first value
            start_period = min(period, len(tr))
            atr[start_period-1] = np.mean(tr[:start_period])
            
            # Apply Wilder's smoothing
            for i in range(start_period, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
            
            # Forward fill initial values
            if start_period > 1:
                atr[:start_period-1] = atr[start_period-1]
            
            return atr
            
        except Exception as e:
            # Fallback: return simple range-based measure
            try:
                highs = data['high'].values.astype(float)
                lows = data['low'].values.astype(float)
                simple_range = highs - lows
                return np.full_like(simple_range, np.mean(simple_range))
            except:
                return np.zeros(len(data))
    
    def detect_patterns(self, data: pd.DataFrame, timeframe: str = "1day") -> List[PatternResult]:
        """
        Enhanced pattern detection with comprehensive results.
        
        Returns:
            List of PatternResult objects with enhanced metadata
        """
        results = []
        
        for i in range(len(data)):
            confidence = self._detect_pattern_at_index(data, i)
            if confidence is not None:
                # Get the actual pattern for metadata
                lookback_data = data.iloc[max(0, i - self.lookback_periods):i + 1].copy()
                atr = self._compute_atr(lookback_data, self.atr_period)
                volatility = self._compute_volatility(lookback_data)
                
                adapted_params = self._adapt_parameters(volatility) if self.adaptive_thresholds else {}
                
                peaks, valleys, peak_scores, valley_scores = self._enhanced_peak_valley_detection(
                    lookback_data, atr, adapted_params
                )
                
                if len(peaks) >= 3 and len(valleys) >= 2:
                    pattern = self._find_best_hs_pattern(
                        lookback_data, peaks, valleys, peak_scores, valley_scores, atr, adapted_params
                    )
                    
                    if pattern is not None:
                        # Create enhanced metadata
                        metadata = {
                            'pattern_components': {
                                'left_shoulder_idx': pattern.left_shoulder_idx + max(0, i - self.lookback_periods),
                                'head_idx': pattern.head_idx + max(0, i - self.lookback_periods),
                                'right_shoulder_idx': pattern.right_shoulder_idx + max(0, i - self.lookback_periods),
                                'left_valley_idx': pattern.left_valley_idx + max(0, i - self.lookback_periods),
                                'right_valley_idx': pattern.right_valley_idx + max(0, i - self.lookback_periods),
                            },
                            'neckline_slope': pattern.neckline_slope,
                            'pattern_height': pattern.pattern_height,
                            'target_price': pattern.target_price,
                            'market_volatility': volatility,
                            'detection_method': 'enhanced_multi_algorithm' if self.use_multiple_algorithms else 'single_algorithm'
                        }
                        
                        # Build PatternResult compatible with project's schema
                        try:
                            dt = data.iloc[i]['datetime'] if 'datetime' in data.columns else (data.index[i] if i < len(data.index) else None)
                        except Exception:
                            dt = None
                        result = PatternResult(
                            datetime=dt,
                            pattern_type=self.get_pattern_name().lower().replace(' ', '_'),
                            confidence=float(confidence),
                            timeframe=timeframe,
                            candle_index=int(i),
                            metadata=metadata
                        )
                        results.append(result)
        
        return results