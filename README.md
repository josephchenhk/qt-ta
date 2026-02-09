# QT-TA

Technical analysis library for quantitative trading with pattern recognition and signal optimization capabilities.

## Features

- **Technical Analysis**: Comprehensive technical indicators via `pandas-ta`
- **Pattern Recognition**: Find similar chart patterns using image hashing
- **Signal Factory**: Backtest and optimize technical indicator signals
- **Visualization**: Interactive charts using Plotly
- **Token Authentication**: Secure access control via `qt-token`

## Installation

### Using Conda Environment (Recommended)

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml

# Activate the environment
conda activate qt_ta_env

# Install qt-ta in development mode
pip install -e .[dev]
```

### Manual Installation

```bash
# Create a new conda environment
conda create -n qt_ta_env python=3.12
conda activate qt_ta_env

# Install build dependencies
conda install -c conda-forge cython setuptools wheel

# Install core dependencies
conda install pandas numpy pytz
pip install pandas-ta

# Install optional dependencies (as needed)
pip install quantstats scikit-optimize plotly Pillow imagehash scipy tqdm

# Install qt-token and qt-data separately (not available on public PyPI)
# Install from local wheels or private repositories:
pip install path/to/qt-token-*.whl
pip install path/to/qt-data-*.whl

# Install qt-ta
pip install -e .[dev]  # Development mode with test dependencies
# or
pip install .  # Standard installation
```

## Usage

### Technical Analysis

```python
from qt_ta.ta import ta
import pandas as pd

# Load your OHLCV data
df = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
})

# Use pandas-ta accessor
sma = df.ta.sma(length=20)
rsi = df.ta.rsi(length=14)
macd = df.ta.macd()
```

### Pattern Recognition

```python
from qt_ta.ptr import PatternRecognition, ChartGenerator
import pandas as pd

# Create a chart generator
chart = ChartGenerator(
    ohlcv_df=your_ohlcv_data,
    start='2024-01-01',
    end='2024-12-31',
    ta_indicators=[('sma', {'length': 20})],
    charts=[('Candlestick', {'open': 'open', 'high': 'high', 'low': 'low', 'close': 'close'}, {})]
)

# Initialize pattern recognition (requires token)
ptr = PatternRecognition(token=your_token, chart=chart)

# Find similar patterns
similarity_df = ptr.compute_similarities(ohlcv_df=historical_data)
top_similar = ptr.find_similar_charts(number_of_charts=9)

# Visualize similar patterns
fig = ptr.plot_similar_charts(rows=3, cols=3)
fig.show()
```

### Signal Factory

```python
from qt_ta.factory import TASignal, TAFactory
from qt_data import QTData

# Create QTData client (for data retrieval)
data_client = QTData(token=your_token)

# Register a signal
ta_signal = TASignal(token=your_token)

def my_signal_func(df):
    # Your signal logic here
    return df['RSI_14'] > 70  # Example: RSI overbought signal

ta_signal.register_signal('rsi_overbought', 'rsi', {'length': 14}, my_signal_func)

# Create factory for backtesting
factory = TAFactory(
    tickers=['AAPL US Equity'],
    start='2020-01-01',
    end='2024-01-01',
    ta_signal=ta_signal,
    freq='daily',
    source='bql',
    token=your_token
)

# Create features
features = factory.create_features(forward_period=5, backward_period=2)

# Run backtest
results = factory.run_backtest(
    tickers=['AAPL US Equity'],
    start='2020-01-01',
    end='2024-01-01',
    signals=['rsi_overbought'],
    ensemble_signals=lambda row: 1 if row['rsi_overbought'] > 0 else 0
)
```

## Authentication

QT-TA uses token-based authentication for certain features (Pattern Recognition and Signal Factory). **Token authentication is mandatory** for these modules.

### Obtaining a Token

Contact the administrator to obtain a valid authentication token.

### Token Validation

The library uses `qt-token` with automatic validation:
- **Default**: Tries `licensing` validation first, falls back to `original` if it fails
- **Explicit**: You can specify `validation_method='licensing'` or `validation_method='original'`

## Environment Setup

### Using environment.yml

```bash
# Create environment
conda env create -f environment.yml

# Activate environment
conda activate qt_ta_env

# Install package
pip install -e .[dev]
```

### Manual Setup

```bash
# Create conda environment
conda create -n qt_ta_env python=3.11
conda activate qt_ta_env

# Install build tools
conda install -c conda-forge cython setuptools wheel

# Install core packages
conda install pandas numpy pytz
pip install pandas-ta

# Install optional packages (as needed)
pip install quantstats scikit-optimize plotly Pillow imagehash scipy tqdm

# Install dependencies
pip install qt-token qt-data

# Install qt-ta
pip install -e .[dev]
```

## Packaging

```bash
# Build wheel
python setup.py bdist_wheel

# Or use run_packaging.py for custom options
python run_packaging.py

# Remove source files from wheel (for distribution)
python run_packaging.py --remove-source

# Specify custom module name, version, or distribution directory
python run_packaging.py --module="qt-ta" --version="1.0.0" --dist-dir="dist"
```

## Testing

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_ta.py
```

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `numpy`: Numerical computing
- `pandas-ta`: Technical analysis indicators
- `pytz`: Timezone handling

### Optional Dependencies
- `bql`: Bloomberg Query Language (for Bloomberg Terminal users)
- `qt-data`: Data retrieval library
- `qt-token`: Token authentication
- `plotly`: Interactive visualization
- `Pillow`: Image processing
- `imagehash`: Image hashing for pattern recognition
- `scipy`: Scientific computing (optimization, interpolation)
- `scikit-optimize`: Bayesian optimization
- `quantstats`: Portfolio analytics and backtesting
- `tqdm`: Progress bars
- `IPython`: Jupyter notebook support

Install optional dependencies as needed:
```bash
pip install qt-ta[bql]          # Bloomberg support
pip install qt-ta[plotting]     # Visualization
pip install qt-ta[optimization] # Optimization tools
pip install qt-ta[all]          # All optional dependencies
```

## Project Structure

```
qt-ta/
├── qt_ta/
│   ├── ta/              # Technical analysis module
│   ├── ptr/              # Pattern recognition module
│   └── factory/          # Signal factory module
├── tests/                # Test suite
├── pyproject.toml        # Project configuration
├── setup.py              # Build configuration
├── environment.yml       # Conda environment
└── README.md            # This file
```

## License

Proprietary - All Rights Reserved

For any business inquiry, please write to: josephchenhk@gmail.com
