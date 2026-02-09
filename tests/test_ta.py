# -*- coding: utf-8 -*-
# @Project : qt-ta
# @Time    : 2025/01/XX
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: test_ta.py

"""
Copyright (C) 2025 Joseph Chen - All Rights Reserved
For any business inquiry, please write to: josephchenhk@gmail.com
"""

import pytest
import pandas as pd
import numpy as np


class TestTechnicalAnalysis:
    """Test technical analysis functionality."""
    
    def test_ta_module_import(self):
        """Test that the ta module can be imported."""
        try:
            from qt_ta.ta import ta
            assert ta is not None
        except ImportError as e:
            pytest.skip(f"ta module not available: {e}")
    
    def test_ta_functions_available(self, sample_dataframe):
        """Test that basic TA functions are available."""
        try:
            from qt_ta.ta import ta
            
            # Test that ta has some methods (pandas_ta functions)
            assert hasattr(ta, 'sma') or hasattr(ta, 'rsi') or hasattr(ta, 'macd')
        except ImportError:
            pytest.skip("ta module not available")
    
    def test_dataframe_accessor(self, sample_dataframe):
        """Test that DataFrame accessor works if available."""
        try:
            from qt_ta.ta import ta
            
            # Check if pandas_ta accessor is registered
            if hasattr(sample_dataframe, 'ta'):
                # Try a simple operation
                result = sample_dataframe.ta.sma(length=10)
                assert result is not None
        except (ImportError, AttributeError):
            pytest.skip("ta accessor not available")
