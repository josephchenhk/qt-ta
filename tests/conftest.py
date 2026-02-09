# -*- coding: utf-8 -*-
# @Project : qt-ta
# @Time    : 2025/01/XX
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: conftest.py

"""
Copyright (C) 2025 Joseph Chen - All Rights Reserved
For any business inquiry, please write to: josephchenhk@gmail.com
"""

import os
import pytest
import pandas as pd
import numpy as np

# Set DEV_MODE before importing qt_token if needed
os.environ["DEV_MODE"] = "1"


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame for testing."""
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    data = {
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 105,
        'low': np.random.randn(100).cumsum() + 95,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 100)
    }
    df = pd.DataFrame(data, index=dates)
    return df


@pytest.fixture
def sample_ohlcv_data():
    """Create sample OHLCV data for testing."""
    dates = pd.date_range('2024-01-01', periods=50, freq='D')
    return pd.DataFrame({
        'open': np.random.randn(50).cumsum() + 100,
        'high': np.random.randn(50).cumsum() + 105,
        'low': np.random.randn(50).cumsum() + 95,
        'close': np.random.randn(50).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 50)
    }, index=dates)
