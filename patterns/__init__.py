"""
Pattern detection module for candlestick patterns.
"""

from .base import BasePatternDetector, PatternResult
from .dragonfly_doji import DragonflyDojiDetector
from .hammer import HammerDetector
from .rising_window import RisingWindowDetector
from .evening_star import EveningStarDetector
from .three_white_soldiers import ThreeWhiteSoldiersDetector

__all__ = [
    'BasePatternDetector',
    'PatternResult',
    'DragonflyDojiDetector',
    'HammerDetector',
    'RisingWindowDetector',
    'EveningStarDetector',
    'ThreeWhiteSoldiersDetector'
]