"""
Pattern detection module for candlestick patterns.
"""

from .base import BasePatternDetector, PatternResult
from .dragonfly_doji import DragonflyDojiDetector
from .hammer import HammerDetector
from .rising_window import RisingWindowDetector
from .evening_star import EveningStarDetector
from .three_white_soldiers import ThreeWhiteSoldiersDetector
from .double_top import DoubleTopDetector
from .double_bottom import DoubleBottomDetector
from .head_and_shoulders import HeadAndShouldersDetector
from .inverse_head_and_shoulders import InverseHeadAndShouldersDetector

__all__ = [
    'BasePatternDetector',
    'PatternResult',
    'DragonflyDojiDetector',
    'HammerDetector',
    'RisingWindowDetector',
    'EveningStarDetector',
    'ThreeWhiteSoldiersDetector',
    'DoubleTopDetector',
    'DoubleBottomDetector',
    'HeadAndShouldersDetector',
    'InverseHeadAndShouldersDetector'
]