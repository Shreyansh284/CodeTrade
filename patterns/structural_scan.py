"""
Structural multi-candle pattern scanner for larger timeframes (e.g., 3-12 months).

Detects multi-swing patterns: Head & Shoulders, Double Top, Double Bottom
across segments of ~100-200 candles with confidence scoring and
"confirmed" vs "potential" status.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd


@dataclass
class StructuralSegment:
    start_idx: int
    end_idx: int
    pattern_type: str  # 'head_and_shoulders' | 'double_top' | 'double_bottom'
    confidence: float  # 0..1
    status: str        # 'confirmed' | 'potential'
    start_dt: Optional[pd.Timestamp] = None
    end_dt: Optional[pd.Timestamp] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'pattern_type': self.pattern_type,
            'confidence': float(self.confidence),
            'status': self.status,
            'start_dt': self.start_dt.isoformat() if isinstance(self.start_dt, pd.Timestamp) else str(self.start_dt),
            'end_dt': self.end_dt.isoformat() if isinstance(self.end_dt, pd.Timestamp) else str(self.end_dt),
        }


class StructuralScanner:
    """Scanner operating on larger time windows using swing points with ATR-aware logic."""

    def __init__(
        self,
        swing_min_distance_ratio: float = 0.05,  # min distance as fraction of window (e.g., 5%)
        peak_tolerance: float = 0.02,           # 2% tolerance for equal peaks/troughs
        min_move_ratio: float = 0.10,           # 10% required move for valley/peak between
        max_segments: int = 10,
        # New tunables
        atr_period: int = 14,
        atr_prominence_mult: float = 1.0,       # swing must exceed local extremes by >= ATR*k
        atr_tolerance_mult: float = 1.5,        # allows peaks/troughs equality within k*ATR
        atr_move_mult: float = 1.5,             # required move at least k*ATR
        max_peak_gap_candles: int = 60,         # allow non-adjacent peak/trough pairs up to this gap
        min_separation_candles: int = 8,        # minimal separation between points
        neckline_slope_tol: float = 0.15        # H&S neckline slope tolerance (relative)
    ):
        self.swing_min_distance_ratio = swing_min_distance_ratio
        self.peak_tolerance = peak_tolerance
        self.min_move_ratio = min_move_ratio
        self.max_segments = max_segments
        self.atr_period = atr_period
        self.atr_prominence_mult = atr_prominence_mult
        self.atr_tolerance_mult = atr_tolerance_mult
        self.atr_move_mult = atr_move_mult
        self.max_peak_gap_candles = max_peak_gap_candles
        self.min_separation_candles = min_separation_candles
        self.neckline_slope_tol = neckline_slope_tol

    # ---------------------------- Public API ---------------------------- #
    def scan(
        self,
        data: pd.DataFrame,
        patterns: List[str] = ("head_and_shoulders", "double_top", "double_bottom"),
    ) -> List[Dict[str, Any]]:
        """Scan data for structural patterns and return segment dicts."""
        if data is None or data.empty:
            return []
        required = ['open', 'high', 'low', 'close']
        if not all(c in data.columns for c in required):
            return []

        # Use datetime column if present
        dt_series = data['datetime'] if 'datetime' in data.columns else None

        # Compute ATR for adaptive thresholds
        atr = self._compute_atr(data, period=self.atr_period)

        swings = self._find_swings(data, atr)
        if not swings or len(swings['peaks']) + len(swings['valleys']) < 4:
            return []

        segs: List[StructuralSegment] = []
        if 'double_top' in patterns:
            segs.extend(self._scan_double_top(data, swings, atr))
        if 'double_bottom' in patterns:
            segs.extend(self._scan_double_bottom(data, swings, atr))
        if 'head_and_shoulders' in patterns:
            segs.extend(self._scan_head_shoulders(data, swings, atr))

        # De-duplicate overlapping segments by keeping highest confidence first
        segs = sorted(segs, key=lambda s: s.confidence, reverse=True)
        filtered: List[StructuralSegment] = []
        taken = np.zeros(len(data), dtype=bool)
        for s in segs:
            overlap = taken[s.start_idx:s.end_idx+1].any()
            if not overlap:
                filtered.append(s)
                taken[s.start_idx:s.end_idx+1] = True
            if len(filtered) >= self.max_segments:
                break

        # Attach datetimes
        for s in filtered:
            if dt_series is not None:
                s.start_dt = dt_series.iloc[s.start_idx]
                s.end_dt = dt_series.iloc[s.end_idx]

        return [s.to_dict() for s in filtered]

    # ---------------------------- Indicators --------------------------- #
    def _compute_atr(self, data: pd.DataFrame, period: int = 14) -> np.ndarray:
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
        # Wilder's smoothing
        atr[period-1] = np.nanmean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        # Fill initial NaNs with first valid
        if np.isnan(atr).any():
            first_valid = np.nanmin(np.where(~np.isnan(atr), np.arange(len(atr)), np.inf))
            if np.isfinite(first_valid):
                atr[:int(first_valid)] = atr[int(first_valid)]
            else:
                atr[:] = np.nan_to_num(tr, nan=np.nanmean(tr))
        return atr

    # ---------------------------- Swing points -------------------------- #
    def _find_swings(self, data: pd.DataFrame, atr: np.ndarray) -> Dict[str, List[int]]:
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        n = len(data)
        min_dist = max(5, int(n * self.swing_min_distance_ratio))

        peaks: List[int] = []
        valleys: List[int] = []

        # ATR-aware prominence: ensure the extreme stands out at least atr_prominence_mult * ATR
        for i in range(min_dist, n - min_dist):
            h = highs[i]
            window_left = highs[i - min_dist:i]
            window_right = highs[i+1:i + 1 + min_dist]
            local_max_neighbors = max(window_left.max(), window_right.max())
            if h >= highs[i - min_dist:i + min_dist + 1].max() and (h - local_max_neighbors) >= self.atr_prominence_mult * atr[i]:
                peaks.append(i)

        for i in range(min_dist, n - min_dist):
            l = lows[i]
            window_left = lows[i - min_dist:i]
            window_right = lows[i+1:i + 1 + min_dist]
            local_min_neighbors = min(window_left.min(), window_right.min())
            if l <= lows[i - min_dist:i + min_dist + 1].min() and (local_min_neighbors - l) >= self.atr_prominence_mult * atr[i]:
                valleys.append(i)

        # Fallback: if too few swings, relax conditions and use simple local extrema
        if (len(peaks) + len(valleys)) < 6 or min(len(peaks), len(valleys)) == 0:
            peaks = []
            valleys = []
            loose_dist = max(3, int(n * 0.02))
            for i in range(loose_dist, n - loose_dist):
                h = highs[i]
                if h >= highs[i - loose_dist:i + loose_dist + 1].max():
                    # ensure it's not a plateau of equal highs; prefer first occurrence
                    if not peaks or (i - peaks[-1]) >= loose_dist:
                        peaks.append(i)
                l = lows[i]
                if l <= lows[i - loose_dist:i + loose_dist + 1].min():
                    if not valleys or (i - valleys[-1]) >= loose_dist:
                        valleys.append(i)

        return {'peaks': peaks, 'valleys': valleys}

    # ---------------------------- Patterns ----------------------------- #
    def _peaks_equal(self, p1_val: float, p2_val: float, ref_price: float, atr_val: float) -> bool:
        # Equal within price tolerance or ATR tolerance
        return abs(p1_val - p2_val) <= max(self.peak_tolerance * ref_price, self.atr_tolerance_mult * atr_val)

    def _move_sufficient(self, top: float, bottom: float, ref_price: float, atr_val: float, is_decline: bool) -> bool:
        move = (top - bottom) / ref_price if is_decline else (top - bottom) / bottom
        return (move >= self.min_move_ratio) or ((top - bottom) >= self.atr_move_mult * atr_val)

    def _scan_double_top(self, data: pd.DataFrame, swings: Dict[str, List[int]], atr: np.ndarray) -> List[StructuralSegment]:
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        closes = data['close'].values.astype(float)
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(peaks) < 2 or len(valleys) < 1:
            return segs

        for i in range(len(peaks) - 1):
            for j in range(i + 1, len(peaks)):
                p1, p2 = peaks[i], peaks[j]
                if p2 - p1 < self.min_separation_candles or p2 - p1 > self.max_peak_gap_candles:
                    continue
                peak1, peak2 = highs[p1], highs[p2]
                peak_avg = (peak1 + peak2) / 2.0
                # equality tolerance adaptive to ATR
                if not self._peaks_equal(peak1, peak2, peak_avg, atr[p2]):
                    continue
                # find lowest valley between
                mid_valleys = [v for v in valleys if p1 < v < p2]
                if not mid_valleys:
                    continue
                v = min(mid_valleys, key=lambda idx: lows[idx])
                valley_low = lows[v]
                # sufficient decline from peaks to valley
                if not self._move_sufficient(peak_avg, valley_low, peak_avg, atr[v], is_decline=True):
                    continue
                # Confirmation: close below valley after p2
                confirmed = False
                if p2 + 1 < len(closes):
                    try:
                        if closes[p2+1:].min() < valley_low:
                            confirmed = True
                    except ValueError:
                        # empty slice; ignore
                        pass
                if not confirmed and closes[-1] < valley_low:
                    confirmed = True
                conf = self._score_double_top(abs(peak1 - peak2) / max(peak1, peak2), (peak_avg - valley_low) / peak_avg, p1, p2)
                status = 'confirmed' if confirmed else 'potential'
                segs.append(StructuralSegment(start_idx=p1, end_idx=p2, pattern_type='double_top', confidence=conf, status=status))
        return segs

    def _scan_double_bottom(self, data: pd.DataFrame, swings: Dict[str, List[int]], atr: np.ndarray) -> List[StructuralSegment]:
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        closes = data['close'].values.astype(float)
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(valleys) < 2 or len(peaks) < 1:
            return segs

        for i in range(len(valleys) - 1):
            for j in range(i + 1, len(valleys)):
                t1, t2 = valleys[i], valleys[j]
                if t2 - t1 < self.min_separation_candles or t2 - t1 > self.max_peak_gap_candles:
                    continue
                trough1, trough2 = lows[t1], lows[t2]
                trough_avg = (trough1 + trough2) / 2.0
                if not self._peaks_equal(trough1, trough2, trough_avg, atr[t2]):
                    continue
                mid_peaks = [p for p in peaks if t1 < p < t2]
                if not mid_peaks:
                    continue
                p = max(mid_peaks, key=lambda idx: highs[idx])
                peak_high = highs[p]
                # sufficient rally from troughs to peak
                if not self._move_sufficient(peak_high, trough_avg, trough_avg, atr[p], is_decline=False):
                    continue
                confirmed = False
                if t2 + 1 < len(closes):
                    try:
                        if closes[t2+1:].max() > peak_high:
                            confirmed = True
                    except ValueError:
                        # empty slice; ignore
                        pass
                if not confirmed and closes[-1] > peak_high:
                    confirmed = True
                conf = self._score_double_bottom(abs(trough1 - trough2) / max(trough1, trough2), (peak_high - trough_avg) / trough_avg, t1, t2)
                status = 'confirmed' if confirmed else 'potential'
                segs.append(StructuralSegment(start_idx=t1, end_idx=t2, pattern_type='double_bottom', confidence=conf, status=status))
        return segs

    def _scan_head_shoulders(self, data: pd.DataFrame, swings: Dict[str, List[int]], atr: np.ndarray) -> List[StructuralSegment]:
        highs = data['high'].values.astype(float)
        lows = data['low'].values.astype(float)
        closes = data['close'].values.astype(float)
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(peaks) < 3 or len(valleys) < 2:
            return segs

        # Evaluate triplets of peaks allowing spacing and shoulder similarity
        for i in range(len(peaks) - 2):
            ls, hd, rs = peaks[i], peaks[i+1], peaks[i+2]
            if (hd - ls) < self.min_separation_candles or (rs - hd) < self.min_separation_candles:
                continue
            if (rs - ls) > self.max_peak_gap_candles * 2:
                continue
            ls_h, hd_h, rs_h = highs[ls], highs[hd], highs[rs]
            if not (hd_h > ls_h and hd_h > rs_h):
                continue
            # shoulder similarity (ATR-aware)
            shoulder_avg_h = (ls_h + rs_h) / 2.0
            shoulder_diff_abs = abs(ls_h - rs_h)
            if shoulder_diff_abs > max(0.05 * shoulder_avg_h, self.atr_tolerance_mult * atr[rs]):
                continue
            # head prominence
            head_prom = (hd_h - shoulder_avg_h) / shoulder_avg_h
            if head_prom < max(0.10, self.min_move_ratio):
                continue
            # valleys between
            left_vs = [v for v in valleys if ls < v < hd]
            right_vs = [v for v in valleys if hd < v < rs]
            if not left_vs or not right_vs:
                continue
            lv = min(left_vs, key=lambda idx: lows[idx])
            rv = min(right_vs, key=lambda idx: lows[idx])
            lv_low, rv_low = lows[lv], lows[rv]
            neckline = (lv_low + rv_low) / 2.0
            # neckline slope tolerance
            t_span = max(1, rs - ls)
            slope_rel = abs(rv_low - lv_low) / max(neckline, 1e-9)
            if slope_rel > self.neckline_slope_tol:
                continue
            # confirmation: close below neckline
            confirmed = False
            if rs + 1 < len(closes):
                try:
                    if closes[rs+1:].min() < neckline:
                        confirmed = True
                except ValueError:
                    pass
            if not confirmed and closes[-1] < neckline:
                confirmed = True

            conf = self._score_head_shoulders(
                shoulder_diff=abs(ls_h - rs_h) / max(ls_h, rs_h),
                head_prom=head_prom,
                valley_diff=abs(lv_low - rv_low) / max(lv_low, rv_low),
                ls=ls, hd=hd, rs=rs
            )
            # volume hint: if volume exists and drops on right shoulder, bump confidence slightly
            try:
                if 'volume' in data.columns:
                    vol = data['volume'].values.astype(float)
                    left_vol = np.nanmean(vol[max(0, ls-5):ls+1])
                    right_vol = np.nanmean(vol[rs: min(len(vol), rs+6)])
                    if right_vol < left_vol:
                        conf = min(1.0, conf * 1.05)
            except Exception:
                pass

            status = 'confirmed' if confirmed else 'potential'
            segs.append(StructuralSegment(start_idx=ls, end_idx=rs, pattern_type='head_and_shoulders', confidence=conf, status=status))
        return segs

    # ---------------------------- Scoring ------------------------------- #
    def _score_double_top(self, height_diff: float, decline_ratio: float, p1: int, p2: int) -> float:
        scores: List[float] = []
        # Peak similarity (smaller diff is better)
        if height_diff <= 0.005: scores.append(1.0)
        elif height_diff <= 0.01: scores.append(0.9)
        elif height_diff <= 0.02: scores.append(0.7)
        else: scores.append(0.5)
        # Valley depth
        if decline_ratio >= 0.20: scores.append(1.0)
        elif decline_ratio >= 0.15: scores.append(0.9)
        elif decline_ratio >= 0.10: scores.append(0.7)
        else: scores.append(0.5)
        # Time separation
        sep = max(1, p2 - p1)
        scores.append(1.0 if sep >= 20 else (0.8 if sep >= 10 else 0.5))
        return float(np.mean(scores) * (0.75 + 0.25 * min(scores)))

    def _score_double_bottom(self, depth_diff: float, rally_ratio: float, t1: int, t2: int) -> float:
        scores: List[float] = []
        # Trough similarity
        if depth_diff <= 0.005: scores.append(1.0)
        elif depth_diff <= 0.01: scores.append(0.9)
        elif depth_diff <= 0.02: scores.append(0.7)
        else: scores.append(0.5)
        # Peak rally
        if rally_ratio >= 0.20: scores.append(1.0)
        elif rally_ratio >= 0.15: scores.append(0.9)
        elif rally_ratio >= 0.10: scores.append(0.7)
        else: scores.append(0.5)
        # Time separation
        sep = max(1, t2 - t1)
        scores.append(1.0 if sep >= 20 else (0.8 if sep >= 10 else 0.5))
        return float(np.mean(scores) * (0.75 + 0.25 * min(scores)))

    def _score_head_shoulders(self, shoulder_diff: float, head_prom: float, valley_diff: float, ls: int, hd: int, rs: int) -> float:
        scores: List[float] = []
        # Shoulder symmetry
        if shoulder_diff <= 0.01: scores.append(1.0)
        elif shoulder_diff <= 0.03: scores.append(0.9)
        elif shoulder_diff <= 0.05: scores.append(0.7)
        else: scores.append(0.5)
        # Head prominence
        if head_prom >= 0.20: scores.append(1.0)
        elif head_prom >= 0.15: scores.append(0.9)
        elif head_prom >= 0.10: scores.append(0.7)
        else: scores.append(0.5)
        # Neckline (valley) symmetry
        if valley_diff <= 0.02: scores.append(1.0)
        elif valley_diff <= 0.05: scores.append(0.8)
        elif valley_diff <= 0.10: scores.append(0.6)
        else: scores.append(0.4)
        # Time symmetry
        lt, rt = max(1, hd - ls), max(1, rs - hd)
        td = abs(lt - rt) / max(lt, rt)
        if td <= 0.20: scores.append(1.0)
        elif td <= 0.40: scores.append(0.8)
        elif td <= 0.60: scores.append(0.6)
        else: scores.append(0.4)
        return float(np.mean(scores) * (0.75 + 0.25 * min(scores)))
