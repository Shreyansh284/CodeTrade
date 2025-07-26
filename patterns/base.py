"""
Base pattern detector class and data structures for candlestick pattern detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from utils.logging_config import get_logger
from utils.error_handler import PatternDetectionError, safe_execute
from utils.cache_manager import cache_manager

logger = get_logger(__name__)


@dataclass
class PatternResult:
    """Data structure for pattern detection results."""
    datetime: datetime
    pattern_type: str
    confidence: float
    timeframe: str
    candle_index: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern result to dictionary format."""
        return {
            'datetime': self.datetime,
            'pattern_type': self.pattern_type,
            'confidence': self.confidence,
            'timeframe': self.timeframe,
            'candle_index': self.candle_index
        }
    
    def to_csv_row(self) -> Dict[str, Any]:
        """Convert pattern result to CSV export format."""
        # Handle datetime formatting safely
        if hasattr(self.datetime, 'strftime'):
            datetime_str = self.datetime.strftime('%Y-%m-%d %H:%M:%S')
        else:
            datetime_str = str(self.datetime)
            
        return {
            'datetime': datetime_str,
            'pattern_type': self.pattern_type.replace('_', ' ').title(),
            'confidence': f"{self.confidence:.3f}",
            'timeframe': self.timeframe,
            'candle_index': self.candle_index
        }


class BasePatternDetector(ABC):
    """Abstract base class for all candlestick pattern detectors."""
    
    def __init__(self, min_confidence: float = 0.5):
        """
        Initialize the pattern detector.
        
        Args:
            min_confidence: Minimum confidence threshold for pattern detection
        """
        self.min_confidence = min_confidence
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def get_pattern_name(self) -> str:
        """Return the name of the pattern this detector identifies."""
        pass
    
    @abstractmethod
    def _detect_pattern_at_index(self, data: pd.DataFrame, index: int) -> Optional[float]:
        """
        Detect pattern at specific candle index.
        
        Args:
            data: OHLCV DataFrame with datetime index
            index: Index position to check for pattern
            
        Returns:
            Confidence score if pattern detected, None otherwise
        """
        pass
    
    def detect(self, data: pd.DataFrame, timeframe: str = "1min") -> List[PatternResult]:
        """
        Detect patterns in the provided OHLCV data with comprehensive error handling and caching.
        
        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            timeframe: Time period of the data (e.g., "1min", "5min", "1hour")
            
        Returns:
            List of detected patterns with confidence scores
        """
        pattern_name = self.get_pattern_name()
        
        try:
            # Generate cache key for pattern detection
            try:
                data_hash = pd.util.hash_pandas_object(data).sum()
                cache_key = f"pattern_{pattern_name}_{data_hash}_{timeframe}_{self.min_confidence}"
                
                # Try to get from cache first
                cached_patterns = cache_manager.get(cache_key)
                if cached_patterns is not None:
                    self.logger.info(f"Loaded {len(cached_patterns)} {pattern_name} patterns from cache")
                    return cached_patterns
                    
            except Exception as e:
                self.logger.warning(f"Error generating cache key for {pattern_name}: {e}")
                cache_key = None
            
            # Validate input data
            if not self._validate_data(data):
                raise PatternDetectionError(
                    f"Invalid data provided to {pattern_name} detector",
                    error_code="INVALID_PATTERN_DATA",
                    context={
                        'pattern_name': pattern_name,
                        'data_shape': data.shape if data is not None else None,
                        'timeframe': timeframe
                    }
                )
            
            # Use vectorized detection if available, otherwise fall back to iterative
            if hasattr(self, '_detect_vectorized') and len(data) > 100:
                try:
                    patterns = self._detect_vectorized(data, timeframe)
                    self.logger.info(f"Used vectorized detection for {pattern_name}")
                except Exception as e:
                    self.logger.warning(f"Vectorized detection failed for {pattern_name}, falling back to iterative: {e}")
                    patterns = self._detect_iterative(data, timeframe)
            else:
                patterns = self._detect_iterative(data, timeframe)
            
            # Cache the results
            if cache_key and patterns is not None:
                cache_manager.set(cache_key, patterns, ttl=600)  # 10 minutes cache
            
            self.logger.info(
                f"Detected {len(patterns)} {pattern_name} patterns "
                f"(scanned {len(data)} candles)"
            )
            
            return patterns
            
        except PatternDetectionError:
            # Re-raise pattern detection errors
            raise
        except Exception as e:
            # Wrap unexpected errors
            raise PatternDetectionError(
                f"Unexpected error in {pattern_name} pattern detection: {str(e)}",
                error_code="PATTERN_DETECTION_UNEXPECTED_ERROR",
                context={
                    'pattern_name': pattern_name,
                    'timeframe': timeframe,
                    'data_length': len(data) if data is not None else 0
                }
            ) from e
    
    def _detect_iterative(self, data: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        """
        Iterative pattern detection (fallback method).
        
        Args:
            data: OHLCV DataFrame
            timeframe: Time period of the data
            
        Returns:
            List of detected patterns
        """
        pattern_name = self.get_pattern_name()
        patterns = []
        detection_errors = []
        
        # Iterate through data to detect patterns
        for i in range(len(data)):
            try:
                confidence = safe_execute(
                    self._detect_pattern_at_index,
                    data, i,
                    context=f"{pattern_name}_detection_at_{i}",
                    fallback_result=None,
                    show_errors=False
                )
                
                if confidence is not None and confidence >= self.min_confidence:
                    try:
                        pattern_result = PatternResult(
                            datetime=data.iloc[i]['datetime'] if 'datetime' in data.columns else data.index[i],
                            pattern_type=pattern_name.lower().replace(' ', '_'),
                            confidence=confidence,
                            timeframe=timeframe,
                            candle_index=i
                        )
                        patterns.append(pattern_result)
                        
                    except Exception as e:
                        detection_errors.append(f"Index {i}: Failed to create pattern result - {str(e)}")
                        
            except Exception as e:
                detection_errors.append(f"Index {i}: Detection failed - {str(e)}")
                continue
        
        # Log detection errors if any
        if detection_errors:
            self.logger.warning(
                f"{pattern_name} detection had {len(detection_errors)} errors: "
                f"{'; '.join(detection_errors[:3])}..."  # Show first 3 errors
            )
        
        return patterns
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains required OHLCV columns with comprehensive checks.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        try:
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            
            # Check if data exists
            if data is None:
                self.logger.error("Data is None")
                return False
                
            if data.empty:
                self.logger.error("Data is empty")
                return False
            
            # Check for required columns
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            # Check for sufficient data points
            min_periods = self._get_min_periods()
            if len(data) < min_periods:
                self.logger.warning(
                    f"Insufficient data points for {self.get_pattern_name()}: "
                    f"got {len(data)}, need at least {min_periods}"
                )
                return False
            
            # Check for data quality issues
            quality_issues = []
            
            # Check for null values in critical columns
            for col in required_columns:
                null_count = data[col].isnull().sum()
                if null_count > 0:
                    quality_issues.append(f"{col} has {null_count} null values")
            
            # Check OHLC relationships
            if all(col in data.columns for col in ['open', 'high', 'low', 'close']):
                invalid_ohlc = ~(
                    (data['high'] >= data['open']) &
                    (data['high'] >= data['close']) &
                    (data['high'] >= data['low']) &
                    (data['low'] <= data['open']) &
                    (data['low'] <= data['close'])
                )
                invalid_count = invalid_ohlc.sum()
                if invalid_count > 0:
                    quality_issues.append(f"{invalid_count} candles have invalid OHLC relationships")
            
            # Check for non-positive prices
            price_columns = ['open', 'high', 'low', 'close']
            for col in price_columns:
                non_positive = (data[col] <= 0).sum()
                if non_positive > 0:
                    quality_issues.append(f"{col} has {non_positive} non-positive values")
            
            # Log quality issues but don't fail validation unless severe
            if quality_issues:
                total_issues = sum(int(issue.split()[2]) for issue in quality_issues if issue.split()[2].isdigit())
                issue_percentage = (total_issues / len(data)) * 100 if len(data) > 0 else 0
                
                if issue_percentage > 50:
                    self.logger.error(
                        f"Severe data quality issues ({issue_percentage:.1f}% of data): "
                        f"{'; '.join(quality_issues[:3])}"
                    )
                    return False
                elif issue_percentage > 10:
                    self.logger.warning(
                        f"Data quality issues ({issue_percentage:.1f}% of data): "
                        f"{'; '.join(quality_issues[:3])}"
                    )
                else:
                    self.logger.info(f"Minor data quality issues: {'; '.join(quality_issues[:3])}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating data for {self.get_pattern_name()}: {e}")
            return False
    
    def _get_min_periods(self) -> int:
        """
        Get minimum number of periods required for pattern detection.
        Override in subclasses if different minimum is needed.
        
        Returns:
            Minimum number of periods required
        """
        return 1
    
    def _get_pattern_description(self) -> str:
        """
        Get description of the pattern.
        Override in subclasses to provide specific descriptions.
        
        Returns:
            Pattern description string
        """
        return f"{self.get_pattern_name()} candlestick pattern"
    
    def calculate_body_size(self, candle: pd.Series) -> float:
        """
        Calculate the body size of a candle.
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            Absolute body size
        """
        return abs(candle['close'] - candle['open'])
    
    def calculate_upper_shadow(self, candle: pd.Series) -> float:
        """
        Calculate the upper shadow length of a candle.
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            Upper shadow length
        """
        return candle['high'] - max(candle['open'], candle['close'])
    
    def calculate_lower_shadow(self, candle: pd.Series) -> float:
        """
        Calculate the lower shadow length of a candle.
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            Lower shadow length
        """
        return min(candle['open'], candle['close']) - candle['low']
    
    def calculate_total_range(self, candle: pd.Series) -> float:
        """
        Calculate the total range (high - low) of a candle.
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            Total range of the candle
        """
        return candle['high'] - candle['low']
    
    def is_bullish_candle(self, candle: pd.Series) -> bool:
        """
        Check if a candle is bullish (close > open).
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            True if bullish, False otherwise
        """
        return candle['close'] > candle['open']
    
    def is_bearish_candle(self, candle: pd.Series) -> bool:
        """
        Check if a candle is bearish (close < open).
        
        Args:
            candle: Series with OHLC data
            
        Returns:
            True if bearish, False otherwise
        """
        return candle['close'] < candle['open']
    
    def calculate_confidence_score(self, criteria_scores: List[float]) -> float:
        """
        Calculate overall confidence score from individual criteria scores.
        
        Args:
            criteria_scores: List of individual criteria scores (0.0 to 1.0)
            
        Returns:
            Overall confidence score (0.0 to 1.0)
        """
        if not criteria_scores:
            return 0.0
            
        # Use weighted average with emphasis on meeting all criteria
        avg_score = np.mean(criteria_scores)
        min_score = np.min(criteria_scores)
        
        # Penalize if any criteria score is very low
        confidence = avg_score * (0.7 + 0.3 * min_score)
        
        return min(max(confidence, 0.0), 1.0)
    
    def _calculate_vectorized_components(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        Calculate vectorized candle components for all candles at once.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            Dictionary with vectorized components
        """
        try:
            # Calculate all components using vectorized operations
            body_size = np.abs(data['close'] - data['open'])
            upper_shadow = data['high'] - np.maximum(data['open'], data['close'])
            lower_shadow = np.minimum(data['open'], data['close']) - data['low']
            total_range = data['high'] - data['low']
            
            # Avoid division by zero
            total_range = np.where(total_range == 0, np.finfo(float).eps, total_range)
            body_size_safe = np.where(body_size == 0, np.finfo(float).eps, body_size)
            
            return {
                'body_size': body_size,
                'upper_shadow': upper_shadow,
                'lower_shadow': lower_shadow,
                'total_range': total_range,
                'body_size_safe': body_size_safe,
                'is_bullish': data['close'] > data['open'],
                'is_bearish': data['close'] < data['open']
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating vectorized components: {e}")
            return {}
    
    def _apply_vectorized_filters(self, mask: np.ndarray, data: pd.DataFrame, timeframe: str) -> List[PatternResult]:
        """
        Apply vectorized filters and create pattern results.
        
        Args:
            mask: Boolean mask indicating where patterns are detected
            data: Original OHLCV data
            timeframe: Time period
            
        Returns:
            List of pattern results
        """
        try:
            pattern_name = self.get_pattern_name()
            patterns = []
            
            # Get indices where patterns are detected
            pattern_indices = np.where(mask)[0]
            
            for idx in pattern_indices:
                try:
                    # Calculate confidence for this specific pattern
                    confidence = self._detect_pattern_at_index(data, idx)
                    
                    if confidence is not None and confidence >= self.min_confidence:
                        pattern_result = PatternResult(
                            datetime=data.iloc[idx]['datetime'] if 'datetime' in data.columns else data.index[idx],
                            pattern_type=pattern_name.lower().replace(' ', '_'),
                            confidence=confidence,
                            timeframe=timeframe,
                            candle_index=idx
                        )
                        patterns.append(pattern_result)
                        
                except Exception as e:
                    self.logger.warning(f"Error creating pattern result at index {idx}: {e}")
                    continue
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Error applying vectorized filters: {e}")
            return []