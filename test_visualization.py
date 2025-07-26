#!/usr/bin/env python3
"""
Simple test script to verify visualization components functionality.
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Add project root to path
sys.path.append('.')

from visualization import ChartRenderer, PatternHighlighter
from patterns.base import PatternResult

def create_sample_data(num_points: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    base_time = datetime.now() - timedelta(hours=num_points)
    
    data = []
    price = 100.0
    
    for i in range(num_points):
        # Simulate price movement
        change = np.random.normal(0, 0.5)
        price += change
        
        # Create OHLCV data
        open_price = price
        high_price = price + abs(np.random.normal(0, 0.3))
        low_price = price - abs(np.random.normal(0, 0.3))
        close_price = price + np.random.normal(0, 0.2)
        volume = int(np.random.uniform(1000, 10000))
        
        data.append({
            'datetime': base_time + timedelta(hours=i),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        price = close_price
    
    return pd.DataFrame(data)

def create_sample_patterns(data: pd.DataFrame) -> list:
    """Create sample pattern results for testing."""
    patterns = []
    
    # Create a few sample patterns
    pattern_types = ['hammer', 'dragonfly_doji', 'rising_window']
    
    for i, pattern_type in enumerate(pattern_types):
        if i * 20 < len(data):
            idx = i * 20
            pattern = PatternResult(
                datetime=data.iloc[idx]['datetime'],
                pattern_type=pattern_type,
                confidence=0.7 + (i * 0.1),
                timeframe='1hour',
                candle_index=idx,
                description=f"Sample {pattern_type} pattern"
            )
            patterns.append(pattern)
    
    return patterns

def test_chart_renderer():
    """Test the ChartRenderer functionality."""
    print("Testing ChartRenderer...")
    
    try:
        # Create sample data
        data = create_sample_data(50)
        patterns = create_sample_patterns(data)
        
        # Initialize renderer
        renderer = ChartRenderer()
        
        # Test basic chart creation
        fig = renderer.create_candlestick_chart(data, title="Test Chart")
        print(f"✓ Created candlestick chart with {len(data)} data points")
        
        # Test pattern highlighting
        fig = renderer.highlight_patterns(fig, patterns, data)
        print(f"✓ Added {len(patterns)} pattern highlights")
        
        # Test interactivity
        fig = renderer.add_interactivity(fig)
        print("✓ Added interactive features")
        
        # Test pattern summary
        summary_fig = renderer.create_pattern_summary_chart(patterns)
        print("✓ Created pattern summary chart")
        
        # Test navigation interface
        nav_data = renderer.create_pattern_navigation_interface(patterns)
        print(f"✓ Created navigation interface with {len(nav_data['patterns'])} patterns")
        
        print("ChartRenderer tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ ChartRenderer test failed: {e}")
        return False

def test_pattern_highlighter():
    """Test the PatternHighlighter functionality."""
    print("\nTesting PatternHighlighter...")
    
    try:
        # Create sample data
        data = create_sample_data(30)
        patterns = create_sample_patterns(data)
        
        # Initialize highlighter
        highlighter = PatternHighlighter()
        
        # Create base chart
        renderer = ChartRenderer()
        fig = renderer.create_candlestick_chart(data, title="Test Highlighter Chart")
        
        # Test pattern overlays
        fig = highlighter.add_pattern_overlays(fig, patterns, data)
        print(f"✓ Added pattern overlays for {len(patterns)} patterns")
        
        # Test confidence indicators
        fig = highlighter.add_confidence_indicators(fig, patterns, data)
        print("✓ Added confidence indicators")
        
        # Test timeline creation
        timeline_fig = highlighter.create_pattern_timeline(patterns)
        print("✓ Created pattern timeline")
        
        # Test detailed overlay
        if patterns:
            fig = highlighter.create_pattern_detail_overlay(fig, patterns[0], data)
            print("✓ Created pattern detail overlay")
        
        print("PatternHighlighter tests passed!")
        return True
        
    except Exception as e:
        print(f"✗ PatternHighlighter test failed: {e}")
        return False

def main():
    """Run all visualization tests."""
    print("Running visualization component tests...\n")
    
    chart_test = test_chart_renderer()
    highlighter_test = test_pattern_highlighter()
    
    if chart_test and highlighter_test:
        print("\n✓ All visualization tests passed!")
        return 0
    else:
        print("\n✗ Some visualization tests failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())