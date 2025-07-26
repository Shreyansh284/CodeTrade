"""
Base pattern detector class and data structures for candlestick pattern detection.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class PatternResult:
    """Data structure for pattern detection results."""
    datetime: datetime
    pattern_type: str
    confidence: float
    timeframe: str
    candle_index: int
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pattern result to dictionary format."""
        return {
            'datetime': self.datetime,
            'pattern_type': self.pattern_type,
            'confidence': self.confidence,
            'timeframe': self.timeframe,
            'candle_index': self.candle_index,
            'description': self.description
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
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
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
        Detect patterns in the provided OHLCV data.
        
        Args:
            data: OHLCV DataFrame with columns [open, high, low, close, volume]
            timeframe: Time period of the data (e.g., "1min", "5min", "1hour")
            
        Returns:
            List of detected patterns with confidence scores
        """
        if not self._validate_data(data):
            self.logger.warning(f"Invalid data provided to {self.get_pattern_name()} detector")
            return []
        
        patterns = []
        pattern_name = self.get_pattern_name()
        
        try:
            # Iterate through data to detect patterns
            for i in range(len(data)):
                confidence = self._detect_pattern_at_index(data, i)
                
                if confidence is not None and confidence >= self.min_confidence:
                    pattern_result = PatternResult(
                        datetime=data.index[i],
                        pattern_type=pattern_name,
                        confidence=confidence,
                        timeframe=timeframe,
                        candle_index=i,
                        description=self._get_pattern_description()
                    )
                    patterns.append(pattern_result)
                    
        except Exception as e:
            self.logger.error(f"Error detecting {pattern_name} patterns: {str(e)}")
            
        self.logger.info(f"Detected {len(patterns)} {pattern_name} patterns")
        return patterns
    
    def _validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that the data contains required OHLCV columns.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data is None or data.empty:
            return False
            
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
            
        # Check for sufficient data points
        if len(data) < self._get_min_periods():
            self.logger.warning(f"Insufficient data points. Need at least {self._get_min_periods()}")
            return False
            
        return True
    
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