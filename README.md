# Stock Pattern Detector

A comprehensive professional-grade application for detecting and visualizing candlestick patterns in stock market data, featuring both an interactive web UI and powerful command-line interface with multi-timeframe analysis capabilities.

## 🌟 Key Features

### 📊 Advanced Data Analysis

- **Multi-Source Data Loading**: Load from local CSV files with automatic validation
- **5 Major Instruments**: BAJAJ-AUTO, BHARTIARTL, ICICIBANK, RELIANCE, TCS
- **7 Timeframe Support**: 1min, 5min, 15min, 1hour, 2hour, 5hour, 1day
- **Date Range Selection**: Choose specific date ranges or use all available data
- **All-Timeframes Mode**: Comprehensive analysis across all timeframes simultaneously
- **Intelligent Caching**: Fast data processing with smart caching system
- **Real-time Validation**: Automatic data quality checks and error handling

### 🔍 Professional Pattern Detection

- **Dragonfly Doji**: Reversal signal with long lower shadow and minimal body
- **Hammer**: Bullish reversal with small body and long lower shadow
- **Rising Window**: Gap-up continuation pattern indicating bullish momentum
- **Evening Star**: Three-candle bearish reversal formation
- **Three White Soldiers**: Strong bullish pattern with consecutive rising candles
- **Vectorized Detection**: High-performance pattern recognition algorithms
- **Confidence Scoring**: Statistical confidence levels for each detected pattern

### 📈 Enhanced Interactive Visualization

- **Dynamic Zoom**: Pattern-focused views with context-aware windowing
- **Date Range Controls**: Flexible date selection for focused time period analysis
- **Multiple Chart Modes**:
  - **Detailed**: Full analysis with all patterns and volume
  - **Compact**: Grid layout for individual pattern comparison
  - **Overlay**: Color-coded confidence-based pattern display
- **Professional Styling**: Modern chart themes with confidence-based markers
- **Context Analysis**: Surrounding market conditions for each pattern
- **Volume Integration**: Relative volume analysis and context
- **Interactive Navigation**: Click-to-zoom and pattern exploration

### 🖥️ Powerful Command Line Interface

- **Dual Mode Operation**: Interactive guided mode or direct command execution
- **Comprehensive Batch Processing**: All instruments × all timeframes
- **Flexible Filtering**: Specific instruments, patterns, or timeframes
- **All-Timeframes Processing**: `--timeframe all` or `--all-timeframes`
- **Automation Ready**: Quiet mode for scripting and automation
- **Progress Tracking**: Real-time progress indicators and summaries
- **Professional Export**: Timestamped CSV files with metadata

### 📋 Advanced Results Management

- **Multi-Tab Analysis**: Basic info, technical analysis, and market context
- **Pattern Context Viewer**: Mini-charts showing pattern in market context
- **Volume Analysis**: Relative volume compared to recent averages
- **Technical Metrics**: Price action, shadows, and candle characteristics
- **Sortable Tables**: Interactive pattern tables with filtering
- **Export Options**: Professional CSV export with download capability

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (Recommended: Python 3.9 or higher)
- **Git** (for cloning the repository)
- **Terminal/Command Prompt** access

### 📦 Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Shreyansh284/CodeTrade.git
   cd CodeTrade
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation:**

   ```bash
   python3 test_comprehensive.py
   ```

4. **Prepare your data:** Place CSV files in the `5Scripts` directory:
   ```
   5Scripts/
   ├── BAJAJ-AUTO/
   │   ├── 01-01-2025.csv
   │   ├── 02-01-2025.csv
   │   └── ... (more CSV files)
   ├── BHARTIARTL/
   ├── ICICIBANK/
   ├── RELIANCE/
   └── TCS/
   ```

### 🎯 Usage Modes

The application supports two main interfaces - choose the one that fits your workflow:

#### 🌐 Web Interface (Recommended for Analysis)

**Launch the interactive web application:**

```bash
python3 main.py --gui
```

_Opens at: http://localhost:8501_

**Features:**

- **Clean, Simple Interface**: Streamlined UI focused on pattern analysis
- **Quick Preview**: Sample chart on welcome screen
- **Real-time Charts**: Interactive candlestick charts with clear pattern markers
- **Pattern Details**: Expandable sections with focused pattern views
- **Configurable Display**: Adjustable chart height and candle count

#### 💻 Command Line Interface (Recommended for Automation)

**Interactive guided mode:**

```bash
python3 main.py -export
```

**Quick single analysis:**

```bash
python3 main.py -export --instrument BAJAJ-AUTO --timeframe 5min
```

**Comprehensive batch processing:**

```bash
# All instruments, all timeframes
python3 main.py -export --batch --all-timeframes

# All instruments, specific timeframe
python3 main.py -export --batch --timeframe 5min

# Specific patterns only
python3 main.py -export --batch --patterns hammer,doji --quiet
```

**Advanced options:**

```bash
# Custom output directory
python3 main.py -export --batch --output-dir ./my_exports

# Custom data source
python3 main.py -export --data-dir ./my_data --batch
```

## 📋 Data Format Requirements

Your CSV files should contain the following columns:

- **date**: DD-MM-YYYY format (e.g., "26-07-2025")
- **time**: HH:MM:SS format (e.g., "09:30:00")
- **open**: Opening price
- **high**: Highest price
- **low**: Lowest price
- **close**: Closing price
- **volume**: Trading volume

**Example CSV structure:**

```csv
date,time,open,high,low,close,volume
26-07-2025,09:30:00,100.50,101.25,100.00,101.00,50000
26-07-2025,09:31:00,101.00,101.50,100.75,101.25,45000
```

## 🎯 Features Guide

### 🌐 Web Interface Features

#### Multi-Timeframe Analysis

1. Select "All Timeframes" from the dropdown
2. View tabbed results for each timeframe
3. Compare pattern frequency across timeframes
4. Export consolidated results

#### Enhanced Pattern Analysis

- **🔍 Zoom Feature**: Click "Zoom" on any pattern for focused view
- **📋 Details Feature**: Click "Details" for comprehensive technical analysis
- **📊 Individual Charts**: Toggle different chart styles (Detailed/Compact/Overlay)
- **🎯 Pattern Navigation**: Click patterns in the table to navigate charts

#### Export & Download

- **CSV Export**: Download pattern results with metadata
- **Chart Export**: Save high-resolution chart images
- **Batch Export**: Export multiple timeframes simultaneously

### 💻 CLI Features

#### Interactive Mode

```bash
python3 main.py -export
```

- Guided instrument selection
- All timeframes option available
- Real-time progress tracking
- Export summaries with file details

#### Automation Examples

```bash
# Daily automation script
python3 main.py -export --batch --all-timeframes --quiet > daily_analysis.log

# Focus on specific patterns
python3 main.py -export --batch --patterns hammer,doji,evening_star

# Quick single instrument check
python3 main.py -export --instrument RELIANCE --timeframe 5min
```

## 🔧 Technical Implementation

### Architecture Overview

```
Stock Pattern Detector/
├── main.py                 # Main entry point with unified CLI
├── ui/streamlit_app.py     # Interactive web interface
├── export/                 # Command-line tools and CSV export
├── data/                   # Data loading and aggregation
├── patterns/               # Pattern detection algorithms
├── visualization/          # Chart rendering and highlighting
├── utils/                  # Utilities (logging, caching, error handling)
└── 5Scripts/              # Your CSV data files
```

### Performance Features

- **Intelligent Caching**: Automatic caching of aggregated data and detected patterns
- **Vectorized Detection**: High-performance NumPy-based pattern recognition
- **Memory Optimization**: Efficient data processing for large datasets
- **Progress Tracking**: Real-time feedback for long-running operations

### Data Processing Pipeline

1. **CSV Loading**: Multi-threaded file loading with validation
2. **Data Aggregation**: OHLCV aggregation to different timeframes
3. **Pattern Detection**: Vectorized algorithms scan all candles
4. **Confidence Scoring**: Statistical analysis of pattern reliability
5. **Export Generation**: Professional CSV output with metadata

## 📊 Pattern Detection Details

### Algorithm Specifications

#### Dragonfly Doji

- **Criteria**: Long lower shadow (>2x body), minimal upper shadow (<10% lower), small body (<5% range)
- **Signal**: Potential reversal, especially at support levels
- **Confidence**: Based on shadow ratio and market context

#### Hammer

- **Criteria**: Lower shadow >2x body, upper shadow <50% body, body in upper 1/3
- **Signal**: Bullish reversal after downtrend
- **Confidence**: Strength increases with volume confirmation

#### Rising Window (Gap Up)

- **Criteria**: Current low > previous high, minimum gap 0.5% of previous close
- **Signal**: Bullish continuation, momentum confirmation
- **Confidence**: Based on gap size and volume

#### Evening Star

- **Criteria**: Three-candle pattern (long bullish → small body → long bearish)
- **Signal**: Bearish reversal at resistance
- **Confidence**: Third candle closes below first candle midpoint

#### Three White Soldiers

- **Criteria**: Three consecutive bullish candles, each opens within previous body
- **Signal**: Strong bullish momentum
- **Confidence**: Each candle closes near session high

## 🧪 Testing & Validation

### Comprehensive Testing

```bash
# Run full test suite
python3 test_comprehensive.py

# Quick functionality check
python3 main.py -export --instrument BAJAJ-AUTO --timeframe 5min --quiet
```

### Test Coverage

- ✅ **Import Validation**: All modules and dependencies
- ✅ **Data Loading**: CSV parsing and validation
- ✅ **Pattern Detection**: All 5 pattern algorithms
- ✅ **Export Functionality**: CSV generation and metadata
- ✅ **CLI Interface**: Command parsing and execution
- ✅ **Visualization**: Chart rendering and styling

## 💡 Usage Tips & Best Practices

### For Traders & Analysts

1. **Start with All Timeframes**: Use the comprehensive view to identify patterns across timeframes
2. **Focus on High Confidence**: Filter patterns with >70% confidence for reliability
3. **Volume Confirmation**: Check volume context for pattern validation
4. **Market Context**: Use the context charts to understand market conditions

### For Developers & Automation

1. **Batch Processing**: Use `--batch --all-timeframes` for comprehensive daily analysis
2. **Quiet Mode**: Add `--quiet` flag for scripting and automation
3. **Custom Directories**: Organize data and exports with `--data-dir` and `--output-dir`
4. **Pattern Filtering**: Focus on specific patterns with `--patterns hammer,doji`

### Performance Optimization

- **Caching**: First run takes longer, subsequent runs use cached data
- **Memory Usage**: Large datasets are automatically optimized for rendering
- **Progress Tracking**: Use interactive mode for real-time feedback

## � Troubleshooting

### Common Issues

**"No instruments found"**

- Ensure CSV files are in `5Scripts/[INSTRUMENT_NAME]/` directories
- Check CSV file format matches requirements
- Verify file permissions

**"Import errors"**

- Run `pip install -r requirements.txt`
- Check Python version (3.8+ required)
- Verify all dependencies are installed

**"Streamlit not found"**

```bash
pip install streamlit>=1.28.0
```

**"No patterns detected"**

- Check data quality and completeness
- Verify timeframe has sufficient data points
- Try different confidence thresholds

### Getting Help

1. Run the comprehensive test: `python3 test_comprehensive.py`
2. Check logs in `logs/` directory
3. Use `--help` flag for command options
4. Review data format requirements

## � Output Examples

### CLI Output Sample

```
🔄 Starting analysis for BAJAJ-AUTO (5min)...
✅ Loaded 52,125 data points
✅ Aggregated to 10,425 data points
🔍 Detecting patterns...
  - Detecting Hammer... ✅ Found 96 patterns
  - Detecting Doji... ✅ Found 226 patterns
📊 Total patterns detected: 322
✅ Results exported to: exports/patterns_BAJAJ_AUTO_5min_20250726_142408.csv
```

### CSV Export Sample

```csv
instrument,timeframe,pattern_type,datetime,confidence,candle_index
BAJAJ-AUTO,5min,hammer,2025-07-26 09:30:00,0.85,1250
BAJAJ-AUTO,5min,dragonfly_doji,2025-07-26 10:15:00,0.92,1259
```

## 🎯 Project Roadmap

### Current Version (v1.0)

- ✅ 5 Pattern types with confidence scoring
- ✅ Multi-timeframe analysis (7 timeframes)
- ✅ Interactive web UI with advanced features
- ✅ Comprehensive CLI with batch processing
- ✅ Professional visualization and export

### Future Enhancements

- 🔄 Additional pattern types (Shooting Star, Morning Star, etc.)
- 🔄 Real-time data integration
- 🔄 Advanced filtering and custom thresholds
- 🔄 Pattern performance analytics
- 🔄 API integration for live market data

## 📄 License & Contributing

This project is designed for educational and professional analysis purposes. The codebase is modular and extensible:

- **Add New Patterns**: Extend `BasePatternDetector` class
- **Enhance Visualization**: Modify chart themes and layouts
- **Data Sources**: Adapt `CSVLoader` for different data formats
- **Export Formats**: Extend `CSVExporter` for additional output types

## 🙏 Acknowledgments

Built with modern Python tools:

- **Streamlit**: Interactive web interface
- **Plotly**: Professional charting library
- **Pandas**: Data manipulation and analysis
- **NumPy**: High-performance numerical computing
  stock-pattern-detector/
  ├── main.py # Main entry point with CLI routing
  ├── ui/
  │ ├── **init**.py
  │ └── streamlit_app.py # Streamlit web interface
  ├── export/
  │ ├── **init**.py
  │ ├── cli_interface.py # Command-line interface
  │ ├── csv_exporter.py # CSV export functionality
  │ ├── export_cli.py # Export CLI script
  │ └── export_patterns.py # Legacy export functions
  ├── data/
  │ ├── loader.py # CSV data loading
  │ └── aggregator.py # Time period aggregation
  ├── patterns/
  │ ├── base.py # Base pattern detector
  │ ├── dragonfly_doji.py # Dragonfly doji detector
  │ ├── hammer.py # Hammer detector
  │ ├── rising_window.py # Rising window detector
  │ ├── evening_star.py # Evening star detector
  │ └── three_white_soldiers.py # Three white soldiers detector
  ├── visualization/
  │ ├── charts.py # Chart rendering
  │ └── pattern_highlighter.py # Pattern highlighting
  ├── utils/
  │ └── **init**.py
  ├── requirements.txt # Dependencies
  └── README.md # This file

````

### Code Quality

- Follows PEP 8 style guidelines
- Type hints for function parameters
- Comprehensive error handling
- Logging for debugging and monitoring
- Modular, maintainable architecture

## 📈 Performance

## 🏆 Summary

The **Stock Pattern Detector** is a professional-grade financial analysis tool that combines advanced pattern recognition algorithms with modern visualization techniques. Whether you're a trader looking for technical signals or a developer building automated trading systems, this application provides the tools and flexibility you need.

### Key Strengths
- **Professional Grade**: Vectorized algorithms with confidence scoring
- **Multi-Modal Interface**: Both interactive GUI and automation-ready CLI
- **Comprehensive Analysis**: All major timeframes and pattern types
- **Production Ready**: Caching, error handling, and performance optimization
- **Extensible Architecture**: Easy to add new patterns and features

### Ready to Start?
```bash
git clone https://github.com/Shreyansh284/CodeTrade.git
cd CodeTrade
pip install -r requirements.txt
python3 test_comprehensive.py
python3 main.py --gui
````

---

_Built with ❤️ for the trading and development community_
