import pandas as pd

# Centralized error message for missing pandas-ta
PANDAS_TA_ERROR_MESSAGE = (
    "pandas-ta is required for technical analysis features.\n"
    "Install it with: pip install pandas-ta\n"
    "Note: pandas-ta 0.4.0+ requires Python 3.12+.\n"
    "For Python 3.9–3.11, you may need to install from source or pin to an older version."
)

try:
    import pandas_ta as ta
    from pandas_ta.custom import create_dir, import_dir
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False

    class DummyTA:
        """
        Minimal stand‑in when pandas-ta is not installed.

        Any attempt to use TA features will raise a concise ImportError
        explaining how to install pandas-ta.
        """

        class AnalysisIndicators:
            def __init__(self, *args, **kwargs):
                raise ImportError(PANDAS_TA_ERROR_MESSAGE)

        @staticmethod
        def get_time(*args, **kwargs):
            return None

        @staticmethod
        def indicators(*args, **kwargs):
            return []

    ta = DummyTA()

import io
from contextlib import redirect_stdout
from typing import Callable
from functools import wraps
import warnings
warnings.filterwarnings("ignore", message="registration of accessor")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered")

class Error:
    """
    Error code and error message. 1 means error; 0 means no error.
    """
    def __init__(self, code: int, msg: str):
        self.code = code
        self.msg = msg
        
    def __str__(self):
        if len(self.msg) > 0:
            return repr(f'[{self.code}] {self.msg}')
        else:
            return repr(f'[{self.code}]')
    
    def __repr__(self): 
        if len(self.msg) > 0:
            return repr(f'[{self.code}] {self.msg}')
        else:
            return repr(f'[{self.code}]')

def redirect_error(func: Callable):
    """
    Redirect error message from stdout to `msg` attribute in Error class.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        f = io.StringIO()
        exception_msg = ''
        with redirect_stdout(f):
            try:
                result = func(*args, **kwargs) 
            except Exception as e:
                exception_msg = str(e)
                result = None
            except Warning as w:
                exception_msg = str(w)
                result = None
        out = f.getvalue()
        if len(out) > 0:
            return Error(1, out), result
        elif result is None:
            return Error(1, exception_msg), result
        else:
            return Error(0, out), result
        return out, result
    return wrapper

if PANDAS_TA_AVAILABLE:
    __df = pd.DataFrame()
    indicators = __df.ta.indicators(as_list=True)
    for attr in indicators:
        setattr(ta, attr, redirect_error(getattr(ta, attr)))
    
    @pd.api.extensions.register_dataframe_accessor("ta")
    class QTraderAnalysisIndicators(ta.AnalysisIndicators):
        """
        QTrader Pandas-TA extension.
        """
        
        def __init__(self, pandas_obj):
            self._validate(pandas_obj)
            self._df = pandas_obj
            self._last_run = ta.get_time(self._exchange, to_string=True)
            
            for attr in indicators:
                setattr(self, attr, redirect_error(getattr(self, attr)))
else:
    # Dummy accessor when pandas-ta is not available
    @pd.api.extensions.register_dataframe_accessor("ta")
    class QTraderAnalysisIndicators:
        """
        Dummy technical analysis accessor when pandas-ta is not installed.
        Any usage will raise a concise ImportError pointing to pandas-ta.
        """

        def __init__(self, pandas_obj):
            raise ImportError(PANDAS_TA_ERROR_MESSAGE)
