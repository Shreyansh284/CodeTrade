"""
Pattern detection module for structural candlestick patterns.

Focused on major structural patterns:
- Head & Shoulders
- Inverse Head & Shoulders
- Double Top  
- Double Bottom
"""

from .base import BasePatternDetector, PatternResult
from .double_top import DoubleTopDetector
from .double_bottom import DoubleBottomDetector
from .head_and_shoulders import ImprovedHeadAndShouldersDetector
from .inverse_head_and_shoulders import InverseHeadAndShouldersDetector

__all__ = [
    'BasePatternDetector',
    'PatternResult',
    'DoubleTopDetector',
    'DoubleBottomDetector',
    'ImprovedHeadAndShouldersDetector',
    'InverseHeadAndShouldersDetector'
]