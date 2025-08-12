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
    """Scanner operating on larger time windows using swing points."""

    def __init__(
        self,
        swing_min_distance_ratio: float = 0.05,  # min distance as fraction of window (e.g., 5%)
        peak_tolerance: float = 0.02,           # 2% tolerance for equal peaks/troughs
        min_move_ratio: float = 0.10,           # 10% required move for valley/peak between
        max_segments: int = 6
    ):
        self.swing_min_distance_ratio = swing_min_distance_ratio
        self.peak_tolerance = peak_tolerance
        self.min_move_ratio = min_move_ratio
        self.max_segments = max_segments

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

        swings = self._find_swings(data)
        if not swings or len(swings['peaks']) + len(swings['valleys']) < 4:
            return []

        segs: List[StructuralSegment] = []
        if 'double_top' in patterns:
            segs.extend(self._scan_double_top(data, swings))
        if 'double_bottom' in patterns:
            segs.extend(self._scan_double_bottom(data, swings))
        if 'head_and_shoulders' in patterns:
            segs.extend(self._scan_head_shoulders(data, swings))

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

    # ---------------------------- Swing points -------------------------- #
    def _find_swings(self, data: pd.DataFrame) -> Dict[str, List[int]]:
        highs = data['high'].values
        lows = data['low'].values
        n = len(data)
        min_dist = max(5, int(n * self.swing_min_distance_ratio))

        peaks: List[int] = []
        valleys: List[int] = []

        for i in range(min_dist, n - min_dist):
            h = highs[i]
            if h >= highs[i - min_dist:i + min_dist + 1].max():
                # unique peak
                if i not in peaks:
                    peaks.append(i)

        for i in range(min_dist, n - min_dist):
            l = lows[i]
            if l <= lows[i - min_dist:i + min_dist + 1].min():
                if i not in valleys:
                    valleys.append(i)

        return {'peaks': peaks, 'valleys': valleys}

    # ---------------------------- Patterns ----------------------------- #
    def _scan_double_top(self, data: pd.DataFrame, swings: Dict[str, List[int]]) -> List[StructuralSegment]:
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(peaks) < 2 or len(valleys) < 1:
            return segs

        # Consider combinations of adjacent peaks chronologically
        for i in range(len(peaks) - 1):
            p1, p2 = peaks[i], peaks[i+1]
            peak1, peak2 = highs[p1], highs[p2]
            height_diff = abs(peak1 - peak2) / max(peak1, peak2)
            if height_diff > self.peak_tolerance:
                continue
            mid_valleys = [v for v in valleys if p1 < v < p2]
            if not mid_valleys:
                continue
            v = min(mid_valleys, key=lambda idx: lows[idx])
            valley_low = lows[v]
            peak_avg = (peak1 + peak2) / 2.0
            decline_ratio = (peak_avg - valley_low) / peak_avg
            if decline_ratio < self.min_move_ratio:
                continue
            # Confirmation: price below valley after p2
            confirmed = False
            if p2 + 1 < len(lows) and lows[p2+1:].min(initial=valley_low+1) < valley_low:
                confirmed = True
            elif closes[-1] < valley_low:
                confirmed = True

            conf = self._score_double_top(height_diff, decline_ratio, p1, p2)
            status = 'confirmed' if confirmed else 'potential'
            segs.append(StructuralSegment(start_idx=p1, end_idx=p2, pattern_type='double_top', confidence=conf, status=status))
        return segs

    def _scan_double_bottom(self, data: pd.DataFrame, swings: Dict[str, List[int]]) -> List[StructuralSegment]:
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(valleys) < 2 or len(peaks) < 1:
            return segs

        for i in range(len(valleys) - 1):
            t1, t2 = valleys[i], valleys[i+1]
            trough1, trough2 = lows[t1], lows[t2]
            depth_diff = abs(trough1 - trough2) / max(trough1, trough2)
            if depth_diff > self.peak_tolerance:
                continue
            mid_peaks = [p for p in peaks if t1 < p < t2]
            if not mid_peaks:
                continue
            p = max(mid_peaks, key=lambda idx: highs[idx])
            peak_high = highs[p]
            trough_avg = (trough1 + trough2) / 2.0
            rally_ratio = (peak_high - trough_avg) / trough_avg
            if rally_ratio < self.min_move_ratio:
                continue
            confirmed = False
            if t2 + 1 < len(highs) and highs[t2+1:].max(initial=peak_high-1) > peak_high:
                confirmed = True
            elif closes[-1] > peak_high:
                confirmed = True

            conf = self._score_double_bottom(depth_diff, rally_ratio, t1, t2)
            status = 'confirmed' if confirmed else 'potential'
            segs.append(StructuralSegment(start_idx=t1, end_idx=t2, pattern_type='double_bottom', confidence=conf, status=status))
        return segs

    def _scan_head_shoulders(self, data: pd.DataFrame, swings: Dict[str, List[int]]) -> List[StructuralSegment]:
        highs = data['high'].values
        lows = data['low'].values
        closes = data['close'].values
        peaks = swings['peaks']
        valleys = swings['valleys']
        segs: List[StructuralSegment] = []
        if len(peaks) < 3 or len(valleys) < 2:
            return segs

        # Evaluate triplets of consecutive peaks
        for i in range(len(peaks) - 2):
            ls, hd, rs = peaks[i], peaks[i+1], peaks[i+2]
            ls_h, hd_h, rs_h = highs[ls], highs[hd], highs[rs]
            if not (hd_h > ls_h and hd_h > rs_h):
                continue
            # shoulder similarity
            shoulder_diff = abs(ls_h - rs_h) / max(ls_h, rs_h)
            if shoulder_diff > max(0.05, self.peak_tolerance):
                continue
            # head prominence
            shoulder_avg = (ls_h + rs_h) / 2.0
            head_prom = (hd_h - shoulder_avg) / shoulder_avg
            if head_prom < max(0.10, self.min_move_ratio):
                continue
            # valleys between
            left_vs = [v for v in valleys if ls < v < hd]
            right_vs = [v for v in valleys if hd < v < rs]
            if not left_vs or not right_vs:
                continue
            lv = min(left_vs, key=lambda idx: lows[idx])
            rv = min(right_vs, key=lambda idx: lows[idx])
            neckline = (lows[lv] + lows[rv]) / 2.0
            # confirmation: break below neckline
            confirmed = False
            if rs + 1 < len(lows) and lows[rs+1:].min(initial=neckline+1) < neckline:
                confirmed = True
            elif closes[-1] < neckline:
                confirmed = True

            conf = self._score_head_shoulders(shoulder_diff, head_prom, abs(lows[lv]-lows[rv])/max(lows[lv], lows[rv]), ls, hd, rs)
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
        return float(np.mean(scores) * (0.7 + 0.3 * min(scores)))

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
        return float(np.mean(scores) * (0.7 + 0.3 * min(scores)))

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
        return float(np.mean(scores) * (0.7 + 0.3 * min(scores)))
