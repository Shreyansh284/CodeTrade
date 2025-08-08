# Double Top and Double Bottom Pattern Detectors

## Overview

The CodeTrade project now includes advanced pattern detectors for Double Top and Double Bottom reversal patterns. These are powerful chart patterns that can signal potential trend reversals.

## Pattern Descriptions

### Double Top Pattern
A **Double Top** is a bearish reversal pattern that typically appears at the end of an uptrend. It consists of:

- **Two peaks** at approximately the same price level
- A **valley** between the peaks (10-20% decline from peaks)
- **Confirmation** when price breaks below the valley low
- Indicates potential trend reversal from bullish to bearish

**Trading Signal**: Sell/Short when price breaks below the valley support level.

### Double Bottom Pattern  
A **Double Bottom** is a bullish reversal pattern that typically appears at the end of a downtrend. It consists of:

- **Two troughs** at approximately the same price level  
- A **peak** between the troughs (10-20% rally from troughs)
- **Confirmation** when price breaks above the peak high
- Indicates potential trend reversal from bearish to bullish

**Trading Signal**: Buy/Long when price breaks above the peak resistance level.

## Usage

### Command Line Interface

```bash
# Export double top patterns for a specific instrument
python export/export_cli.py --instrument RELIANCE --patterns double_top --timeframe 1hour

# Export double bottom patterns  
python export/export_cli.py --instrument TCS --patterns double_bottom --timeframe 1day

# Export both patterns
python export/export_cli.py --instrument ICICIBANK --patterns double_top,double_bottom --timeframe 5min
```

### Streamlit Web Interface

1. Run the web application:
```bash
streamlit run ui/streamlit_app.py
```

2. Select "Double Top" or "Double Bottom" from the pattern dropdown
3. Choose your instrument and timeframe
4. Click "Detect Patterns" to analyze

### Programmatic Usage

```python
from patterns.double_top import DoubleTopDetector
from patterns.double_bottom import DoubleBottomDetector
from data.loader import CSVLoader

# Load data
loader = CSVLoader("5Scripts")
data = loader.load_instrument_data("RELIANCE")

# Initialize detectors with custom parameters
dt_detector = DoubleTopDetector(
    min_confidence=0.6,        # Minimum confidence threshold
    peak_tolerance=0.02,       # 2% tolerance between peaks
    min_valley_decline=0.15,   # 15% minimum valley decline
    lookback_periods=100       # Look back 100 periods
)

db_detector = DoubleBottomDetector(
    min_confidence=0.6,        # Minimum confidence threshold  
    trough_tolerance=0.02,     # 2% tolerance between troughs
    min_peak_rally=0.15,       # 15% minimum peak rally
    lookback_periods=100       # Look back 100 periods
)

# Detect patterns
dt_patterns = dt_detector.detect(data, timeframe="1hour")
db_patterns = db_detector.detect(data, timeframe="1hour")

# Process results
for pattern in dt_patterns:
    print(f"Double Top at {pattern.datetime}: confidence={pattern.confidence:.3f}")

for pattern in db_patterns:
    print(f"Double Bottom at {pattern.datetime}: confidence={pattern.confidence:.3f}")
```

## Configuration Parameters

### DoubleTopDetector Parameters
- `min_confidence` (0.0-1.0): Minimum confidence threshold for pattern detection
- `peak_tolerance` (0.0-1.0): Maximum percentage difference between peaks (default: 2%)
- `min_valley_decline` (0.0-1.0): Minimum decline from peak to valley (default: 10%)  
- `lookback_periods` (int): Number of periods to analyze for pattern formation (default: 50)

### DoubleBottomDetector Parameters
- `min_confidence` (0.0-1.0): Minimum confidence threshold for pattern detection
- `trough_tolerance` (0.0-1.0): Maximum percentage difference between troughs (default: 2%)
- `min_peak_rally` (0.0-1.0): Minimum rally from trough to peak (default: 10%)
- `lookback_periods` (int): Number of periods to analyze for pattern formation (default: 50)

## Confidence Scoring

The confidence score (0.0 to 1.0) is calculated based on multiple criteria:

1. **Peak/Trough Similarity**: How closely the two peaks/troughs match in price
2. **Valley/Peak Depth**: Significance of the intermediate decline/rally
3. **Time Separation**: Adequate time between the two peaks/troughs
4. **Volume Analysis**: Volume patterns during pattern formation (if available)
5. **Pattern Confirmation**: Whether the pattern has been confirmed by price action
6. **Trend Context**: Whether the pattern appears in the appropriate trend context

## Best Practices

1. **Timeframe Selection**: Higher timeframes (1hour, 1day) typically produce more reliable patterns
2. **Confirmation**: Wait for pattern confirmation (breakout) before trading
3. **Volume**: Look for increased volume on the confirmation breakout
4. **Risk Management**: Always use stop-losses when trading these patterns
5. **Multiple Timeframes**: Confirm patterns on multiple timeframes for higher probability trades

## Testing

Run the test script to verify pattern detection:

```bash
python test_double_patterns.py
```

This will test both patterns with synthetic data and real market data to ensure proper functionality.

## Notes

- These patterns work best in trending markets where clear reversals are more likely
- False breakouts can occur, so wait for confirmation with volume
- The patterns are more reliable when they appear after significant trends
- Consider using these patterns in conjunction with other technical indicators for better accuracy
