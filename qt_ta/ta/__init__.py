"""
TA subpackage

This module re‑exports the key symbols from pandas_ta_patch in a clean way.
If pandas-ta is not installed, a single, concise ImportError will be raised
from the underlying implementation when TA features are actually used.
"""

from .pandas_ta_patch import ta, Error, redirect_error, PANDAS_TA_AVAILABLE, PANDAS_TA_ERROR_MESSAGE

__all__ = ['ta', 'Error', 'redirect_error', 'PANDAS_TA_AVAILABLE', 'PANDAS_TA_ERROR_MESSAGE']