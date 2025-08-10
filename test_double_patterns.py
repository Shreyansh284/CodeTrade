#!/usr/bin/env python3
"""
Test script for Double Top and Double Bottom pattern detectors.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patterns.double_top import DoubleTopDetector
from patterns.double_bottom import DoubleBottomDetector
from data.loader import CSVLoader


def create_sample_double_top_data():
    """Create sample data with a double top pattern."""
    # Create 100 data points
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
    
    # Base trend with upward movement, then double top, then decline
    base_price = 100
    prices = []
    
    for i in range(100):
        if i < 20:
            # Initial uptrend
            price = base_price + (i * 2) + np.random.normal(0, 1)
        elif i < 30:
            # First peak around 140
            peak1_center = 25
            distance_from_peak = abs(i - peak1_center)
            price = base_price + 40 - (distance_from_peak * 2) + np.random.normal(0, 1)
        elif i < 50:
            # Decline to valley around 120
            valley_center = 40
            distance_from_valley = abs(i - valley_center)
            price = base_price + 20 + (distance_from_valley * 1) + np.random.normal(0, 1)
        elif i < 60:
            # Second peak around 140 (similar to first)
            peak2_center = 55
            distance_from_peak = abs(i - peak2_center)
            price = base_price + 41 - (distance_from_peak * 2) + np.random.normal(0, 1)
        else:
            # Decline after second peak
            decline_factor = (i - 60) * 1.5
            price = base_price + 35 - decline_factor + np.random.normal(0, 1)
        
        prices.append(max(price, 50))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = close * 0.02  # 2% volatility
        open_price = close + np.random.normal(0, volatility * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, volatility))
        low = min(open_price, close) - abs(np.random.normal(0, volatility))
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def create_sample_double_bottom_data():
    """Create sample data with a double bottom pattern."""
    # Create 100 data points
    dates = pd.date_range(start='2025-01-01', periods=100, freq='1H')
    
    # Base trend with downward movement, then double bottom, then rally
    base_price = 140
    prices = []
    
    for i in range(100):
        if i < 20:
            # Initial downtrend
            price = base_price - (i * 2) + np.random.normal(0, 1)
        elif i < 30:
            # First trough around 100
            trough1_center = 25
            distance_from_trough = abs(i - trough1_center)
            price = base_price - 40 + (distance_from_trough * 2) + np.random.normal(0, 1)
        elif i < 50:
            # Rally to peak around 120
            peak_center = 40
            distance_from_peak = abs(i - peak_center)
            price = base_price - 20 - (distance_from_peak * 1) + np.random.normal(0, 1)
        elif i < 60:
            # Second trough around 100 (similar to first)
            trough2_center = 55
            distance_from_trough = abs(i - trough2_center)
            price = base_price - 41 + (distance_from_trough * 2) + np.random.normal(0, 1)
        else:
            # Rally after second trough
            rally_factor = (i - 60) * 1.5
            price = base_price - 35 + rally_factor + np.random.normal(0, 1)
        
        prices.append(max(price, 50))  # Ensure positive prices
    
    # Create OHLCV data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic OHLC from close price
        volatility = close * 0.02  # 2% volatility
        open_price = close + np.random.normal(0, volatility * 0.5)
        high = max(open_price, close) + abs(np.random.normal(0, volatility))
        low = min(open_price, close) - abs(np.random.normal(0, volatility))
        volume = np.random.randint(10000, 100000)
        
        data.append({
            'datetime': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_double_top_detector():
    """Test the Double Top pattern detector."""
    print("Testing Double Top Detector...")
    
    # Create sample data
    data = create_sample_double_top_data()
    
    # Initialize detector
    detector = DoubleTopDetector(min_confidence=0.5)
    
    # Detect patterns
    patterns = detector.detect(data, timeframe="1hour")
    
    print(f"Sample data shape: {data.shape}")
    print(f"Close price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"Detected {len(patterns)} double top patterns")
    
    for pattern in patterns:
        print(f"  Pattern at {pattern.datetime}: confidence={pattern.confidence:.3f}")
    
    return len(patterns) > 0


def test_double_bottom_detector():
    """Test the Double Bottom pattern detector."""
    print("\nTesting Double Bottom Detector...")
    
    # Create sample data
    data = create_sample_double_bottom_data()
    
    # Initialize detector
    detector = DoubleBottomDetector(min_confidence=0.5)
    
    # Detect patterns
    patterns = detector.detect(data, timeframe="1hour")
    
    print(f"Sample data shape: {data.shape}")
    print(f"Close price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    print(f"Detected {len(patterns)} double bottom patterns")
    
    for pattern in patterns:
        print(f"  Pattern at {pattern.datetime}: confidence={pattern.confidence:.3f}")
    
    return len(patterns) > 0


def test_with_real_data():
    """Test with real data from StockData (or fallback 5Scripts)."""
    print("\nTesting with real data...")
    
    try:
        # Attempt new flat directory first, fallback to legacy
        loader = CSVLoader("StockData")
        instruments = loader.get_available_instruments()
        if not instruments:
            loader = CSVLoader("5Scripts")
            instruments = loader.get_available_instruments()
        
        if instruments:
            # Use first available instrument
            instrument = instruments[0]
            print(f"Testing with {instrument}...")
            
            data = loader.load_instrument_data(instrument)
            
            if data is not None and len(data) > 50:
                # Test both detectors
                dt_detector = DoubleTopDetector(min_confidence=0.4)  # Lower threshold for real data
                db_detector = DoubleBottomDetector(min_confidence=0.4)
                
                dt_patterns = dt_detector.detect(data, timeframe="1min")
                db_patterns = db_detector.detect(data, timeframe="1min")
                
                print(f"Real data shape: {data.shape}")
                print(f"Double Top patterns: {len(dt_patterns)}")
                print(f"Double Bottom patterns: {len(db_patterns)}")
                
                # Show a few examples
                for pattern in dt_patterns[:3]:
                    print(f"  DT: {pattern.datetime} (confidence: {pattern.confidence:.3f})")
                
                for pattern in db_patterns[:3]:
                    print(f"  DB: {pattern.datetime} (confidence: {pattern.confidence:.3f})")
                
                return True
            else:
                print("Insufficient real data for testing")
                return False
        else:
            print("No instruments available for testing")
            return False
            
    except Exception as e:
        print(f"Error testing with real data: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Double Top and Double Bottom Pattern Detector Tests")
    print("=" * 60)
    
    # Test with synthetic data
    dt_test_passed = test_double_top_detector()
    db_test_passed = test_double_bottom_detector()
    
    # Test with real data
    real_data_test_passed = test_with_real_data()
    
    print("\n" + "=" * 60)
    print("Test Results:")
    print(f"Double Top (synthetic): {'PASS' if dt_test_passed else 'FAIL'}")
    print(f"Double Bottom (synthetic): {'PASS' if db_test_passed else 'FAIL'}")
    print(f"Real data test: {'PASS' if real_data_test_passed else 'FAIL'}")
    print("=" * 60)
    
    if dt_test_passed and db_test_passed:
        print("✅ All pattern detectors are working correctly!")
    else:
        print("❌ Some tests failed. Please check the implementation.")


if __name__ == "__main__":
    main()
