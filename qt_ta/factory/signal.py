import re
import warnings
from typing import Callable, Dict, List, Tuple, Union, Any, Optional
from functools import partial
from itertools import product
import pandas as pd
import numpy as np
import quantstats as qs
from skopt import gp_minimize

from qt_data import get_historical_data
from qt_ta.ta import ta
from qt_token import TokenCheckMeta

TOKEN_KEY = "qt_ta/factory"

class TASignal(metaclass=TokenCheckMeta(token_key=TOKEN_KEY)):
    """Convert TA into signal. This is indicator-dependent, therefore each TA indicator
    will need to implement a `register_signal` method.
    """
    
    def __init__(self, token: str=''):
        self.token = token
        
        self._signals = {}
        
    def register_signal(self, signal_name: str, ta_name: str, ta_params: Optional[Dict[str, Any]], ta_signal_func: Callable):
        """Register the signal calculation method for a given indicator.
        
        Args:
            signal_name   : The TA signal name.
            ta_name       : The TA indicator name.
            ta_params     : The TA indicator parameters.
            ta_signal_func: The signal calculation method, i.e., the function.
        """
        self._signals[signal_name] = {
            'ta_name': ta_name, 
            'ta_params': ta_params,
            'ta_signal_func': ta_signal_func
        }
        
    def get_signal_names(self) -> List[str]:
        """Get the registered signal names."""
        return list(self._signals.keys())
    
    def calc_signal(self, ohlcv_df: pd.DataFrame, signal_name: str) -> pd.Series:
        """Get the signal with ohlcv data. The input `signal_name` should be registered in `register_signal` first.
        
        Args:
            ohlcv_df   : The dataframe with OHLCV.
            signal_name: The TA signal name.
            
        Return:
            The signals series.
        """
        ta_name = self._signals[signal_name]['ta_name']
        ta_params = self._signals[signal_name]['ta_params']
        ta_signal_func = self._signals[signal_name]['ta_signal_func']
        if ta_params is None:
            ta_params = {}
        if ta_name.startswith('cdl_'):
            name = ta_name.replace('cdl_', '')
            err, ta_df = ohlcv_df.ta.cdl_pattern(name=name, **ta_params)
        else:
            err, ta_df = getattr(ohlcv_df.ta, ta_name)(**ta_params)
        if err.code:
            raise ValueError(f'`{ta_name}` calculation error\n: {err.msg}.')
        df = ohlcv_df.join(ta_df)
        return ta_signal_func(df)
    

    
def evaluate_expression(df: pd.DataFrame, expression: str, col: str):
    """
    df is a pandas dataframe with index the different data fields: return, sharpe, maximum drawdown, etc. 
    This function is used to evaluate the result of a string expression with some mathematical operators. 
    Examples:
        - for input "{return }- 0.5*{maximum drawdown}", the expression will be evaluated by 
          > `df.loc['return', 'strategy'] - 0.5 * df.loc['maximum drawdown', 'strategy']`; 
        - for input "2*{sharpe}", the expression will be evaluated by 
          > `2*df.loc['sharpe', 'strategy']`.
          
    Args:
        df        : The dataframe with the data fields in its index.
        expression: The string expression where data fields are quoted by brackets `{}`.
        col       : The column name to be used in the dataframe df.
        
    Returns:
        The evaluated numeric results of the string expression.
    """
    pattern = r'\{(.+?)\}' 
    def replacer(match):
        metric = match.group(1).strip()
        try:
            value = df.loc[metric, col]
        except KeyError:
            raise ValueError(f"Metric '{metric}' not found in DataFrame index.")
        return str(value)   
    substituted_expr = re.sub(pattern, replacer, expression)   
    try:
        result = eval(substituted_expr)
    except Exception as e:
        warnings.warn(f"Error evaluating expression: {e}")
        result = None
    if not isinstance(result, float):
        result = None
    return result


        
class TAFactory(metaclass=TokenCheckMeta(token_key=TOKEN_KEY)):
    """TAFactory is used to explore the effectiveness of different technical indicators
    and different combinations of them.
    """
    def __init__(self, 
                 tickers: List[str], 
                 start: str, 
                 end: str, 
                 ta_signal: TASignal, 
                 freq: str='daily', 
                 source: str='bql',
                 token: str='',
    ):
        """
        Instantiate a TAFactory for backtesting and optimization.

        Args:
            tickers          : The list of tickers.
            start            : The start date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            end              : The end date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            ta_signal        : The TASignal instance.
            freq             : The data frequency, either `daily` or `intraday`.
            source           : The data source.
            token            : The valid token to enable the library.
            
        Return:
            None
        """
        self.token = token
        
        self.tickers = tickers
        self.ta_signal = ta_signal
        self.start = start
        self.end = end
        self.freq = freq
        if freq == 'daily':
            self.ohlcv = get_historical_data(
                tickers=tickers,
                flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],
                start=start,
                end=end,
                source=source
            )
        self._features = None
        self._feature_stats = None
        
    def create_features(self, forward_period: int = 5, backward_period: int = 0) -> pd.DataFrame:
        """
        Create features based on backward window and forward window.

        Args:
            forward_period   : The forecasting window.
            backward_period  : The lookback window.
            
        Return:
            A features dataframe with X (the technical signals) and y (the forward-looking returns).
        """
        tickers = self.tickers 
        start = self.start 
        end = self.end 
        ohlcv = self.ohlcv 
        ta_signal = self.ta_signal
        signal_return_df = None
        for ticker in tickers:         
            ohlcv_df = pd.concat([
                ohlcv.loc['PX_OPEN', ticker],
                ohlcv.loc['PX_HIGH', ticker],
                ohlcv.loc['PX_LOW', ticker],
                ohlcv.loc['PX_LAST', ticker],
                ohlcv.loc['PX_VOLUME', ticker],
            ], axis=1)
            ohlcv_df.columns = ['open', 'high', 'low', 'close', 'volume']

            # generate signals
            for signal_name in ta_signal.get_signal_names():
                ohlcv_df[signal_name] = ta_signal.calc_signal(ohlcv_df, signal_name)
                # lag the signals
                for lag in range(1, backward_period+1):
                    ohlcv_df[f'{signal_name}^{lag}'] = ohlcv_df[signal_name].shift(lag)

            # calculate forward returns
            targets = [f'ret_{i}' for i in range(1, forward_period+1)]
            for target in targets:
                i = int(target.replace('ret_' ,''))
                ohlcv_df[target] = ohlcv_df['close'].shift(-i) / ohlcv_df['close'] - 1

            # tag the ticker
            ohlcv_df['ticker'] = ticker

            # filter and keep only rows with signals
            signal_cols = []
            for signal_name in ta_signal.get_signal_names():
                signal_cols.append(signal_name)
                for lag in range(1, backward_period+1):
                    signal_cols.append(f'{signal_name}^{lag}')
            df = ohlcv_df[['ticker'] + signal_cols + targets].copy()

            if signal_return_df is None:
                signal_return_df = df
            else:
                signal_return_df = pd.concat([signal_return_df, df])
        self._features = signal_return_df
        return signal_return_df
    
    def create_feature_stats(self) -> pd.DataFrame:
        """
        Calculate the performance statistics of the features that have been created from `create_feature`.
            
        Return:
            A dataframe of the features performance.
        """
        ta_signal = self.ta_signal
        if self._features is None:
            warnings.warn('Features are not available. Run default `create_features` first.')
            self.create_features()
        _features = self._features
        direction = ['long', 'short']
        stats = ['mean', 'std']
        # metrics = ['ret_1', 'ret_2', 'ret_3', 'ret_4', 'ret_5']
        metrics = [col for col in _features.columns if 'ret_' in col]
        indices_tuple = list(product(*[direction, stats, metrics]))
        pd_indices = pd.MultiIndex.from_tuples(indices_tuple, names=('direction', 'stats', 'metrics'))
        pd_columns = ta_signal.get_signal_names()
        _data = pd.DataFrame(index=pd_indices, columns=pd_columns)
        for side in direction:
            if side == 'long':
                side_int = 1
            elif side == 'short':
                side_int = -1
            for stat in stats:
                for signal_name in ta_signal.get_signal_names():
                    _data.loc[(side, stat, metrics),  [signal_name] ] = (
                        getattr(_features[_features[signal_name]==side_int].dropna()[metrics], stat)()
                    ).values
        self._feature_stats = _data
        return _data
    
    def run_backtest(
            self,
            tickers: List[str]=['700 HK Equity', '9988 HK Equity', '1810 HK Equity'],
            start: str='2010-01-01',
            end:str='2024-01-01',
            freq: str=None,
            signals:List[str]=['cdl_harami', 'cdl_harami_1', 'aberration', 'aberration_1', 'rsi_mean_reversion', 'rsi_mean_reversion_1', 'rsi_trend', 'rsi_trend_1'],
            ensemble_signals: Callable=None,
            source: str='bql'
    ) -> pd.DataFrame:
        """Backtest the signals.
        Use `ensemble_signals` to aggregate the list of signals to a single signal; then the signal calculated today 
        would indicate the position tomorrow. 
        
        Args:
            tickers          : The list of tickers.
            return_windows   : The holding period for return calculation.
            start            : The start date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            end              : The end date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            freq             : The data frequency, either `daily` or `intraday`.
            signals          : The signals to be tested.
            ensemble_signals : The callable function to convert aggregate signals to single signal.
            
        Return:
            The backtest results.
        """
        ta_signal = self.ta_signal
        if freq is None:
            freq = self.freq
        if freq == 'daily':
            ohlcv = get_historical_data(
                tickers=tickers,
                flds=['PX_OPEN', 'PX_HIGH', 'PX_LOW', 'PX_LAST', 'PX_VOLUME'],
                start=start,
                end=end,
                source=source
            )
        
        res_dict = {}
        for ticker in tickers:
            ohlcv_df = pd.concat([
                ohlcv.loc['PX_OPEN', ticker],
                ohlcv.loc['PX_HIGH', ticker],
                ohlcv.loc['PX_LOW', ticker],
                ohlcv.loc['PX_LAST', ticker],
                ohlcv.loc['PX_VOLUME', ticker],
            ], axis=1).dropna(how='all').ffill()
            ohlcv_df.columns = ['open', 'high', 'low', 'close', 'volume']
            
            signal_names = [sig for sig in signals if re.search(r'^\d+$', sig) is None] # only count those without lag
            signal_ts = {}
            for signal_name in signal_names:
                # TODO: make this `_signals` public
                ta_name = ta_signal._signals[signal_name]['ta_name']
                signal_res = ta_signal.calc_signal(ohlcv_df, signal_name)        
                signal_ts[signal_name] = signal_res

            # prepare signals
            ta_df = ohlcv_df.copy()
            for signal_name_with_lag in signals:
                if re.search(r'^\d+$', signal_name_with_lag) is None:
                    signal_name = signal_name_with_lag
                    ta_df[signal_name_with_lag] = signal_ts[signal_name]
                else:
                    signal_name = '_'.join(signal_name_with_lag.split('_')[:-1])
                    lag_str = re.search(r'^\d+$', signal_name_with_lag).group(0).replace('_', '')
                    lag = int(lag_str)
                    ta_df[signal_name_with_lag] = signal_ts[signal_name].shift(lag)

            # aggregate signals to a single signal
            ta_df['signal'] = ta_df[signals].fillna(0).apply(ensemble_signals, axis=1)
            # signal calculated today would indicate the position tomorrow
            ta_df['position'] = ta_df['signal'].shift(1).fillna(0)
            
            # calculate spot returns: open-to-close(o2c), close-to-open(c2o), and close-to-open(c2o)
            ta_df['return_o2c'] = (ta_df['close'] - ta_df['open']) / ta_df['open']
            ta_df['return_c2o'] = (ta_df['open'] - ta_df['close'].shift(1)) / ta_df['close'].shift(1)
            ta_df['return_c2c'] = ta_df['close'].pct_change()
            # handle NaNs
            ta_df['return_o2c'] = ta_df['return_o2c'].fillna(0)
            ta_df['return_c2o'] = ta_df['return_c2o'].fillna(0)
            ta_df['return_c2c'] = ta_df['return_c2c'].fillna(0)
            # Note: this is not a very accurate pnl calculation (c2o and o2c return should be cumprod)
            ta_df['return'] = ta_df['return_o2c'] * ta_df['position']  + ta_df['return_c2o'] * ta_df['position'].shift(1)
            ta_df['return'] = ta_df['return'].fillna(0)
            
            res_dict[ticker] = ta_df[['return_c2c', 'return']]
        return res_dict
    
    def calc_goal(
            self,
            wts: List[float],
            ticker: str='GC1 Comdty',
            start: str='2010-01-01',
            end: str='2024-02-21',
            freq: str='daily',
            signals: List[str]=[],
            ensemble_signals_with_params: Callable=None,
            metric: str='{Sharpe}',
            goal: str='maximize'
    ) -> float:    
        """Calculation of the optimization goal.
        
        Args:
            wts                          : The parameters that feed into `ensemble_signals_with_params`.
            tickers                      : The list of tickers.
            start                        : The start date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            end                          : The end date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            freq                         : The data frequency, either `daily` or `intraday`.
            signals                      : The signals to be tested.
            ensemble_signals_with_params : The callable function (with parmas) to convert aggregate signals to single signal.
            metric                       : The metric from `run_backtest`, can also be an expression of the multiple metrics.
            goal                         : Either 'maximize' or 'minimize'.
            
        Return:
            The optimized value.
        """
        result = self.run_backtest(
            tickers=[ticker],
            start=start,
            end=end,
            freq=freq,
            signals=signals,
            ensemble_signals=partial(ensemble_signals_with_params, wts=wts)
        )

        perf_df = pd.concat([
            qs.reports.metrics(result[ticker]['return_c2c'].fillna(0), display=False),
            qs.reports.metrics(result[ticker]['return'].fillna(0), display=False),
        ], axis=1)
        perf_df.columns = [f'{ticker}', f'Strategy({ticker})']

        result = evaluate_expression(perf_df, metric, f'Strategy({ticker})')
        if goal == 'maximize':
            if result is None:
                return -10
            else:
                return -result
        elif goal == 'minimize':
            if result is None:
                return 10
            else:
                return result
    
    def optimize_ensembled_signal_params(
            self,
            ticker: str,
            start: str,
            end:str,
            freq: str,
            signals:List[str],
            ensemble_signals_with_params: Callable,
            metric: str,
            goal: str,
            param_bounds: List[Tuple],
            n_calls: int=50,
            source: str='bql'
    ) -> Dict[str, Any]:
        """Optimize the parameters in `ensemble_signals`.
        
        Args:
            tickers                      : The list of tickers.
            start                        : The start date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            end                          : The end date (yyyy-mm-dd) or datetime (yyyy-mm-dd HH:MM:SS).
            freq                         : The data frequency, either `daily` or `intraday`.
            signals                      : The signals to be tested.
            ensemble_signals_with_params : The callable function (with parmas) to convert aggregate signals to single signal.
            metric                       : The metric from `run_backtest`, can also be an expression of the multiple metrics.
            goal                         : Either 'maximize' or 'minimize'.
            param_bounds                 : The range of the params.
            n_calls                      : The maximum number of iterations in optimization.
            source                       : The data source.
            
        Return:
            The optimized results with `Optimized Params` and `Optimized Value`.
        """
        my_bounds = param_bounds # Example: 2 weights [(-10, 10) for _ in range(2)]  
        my_goal = partial(
            self.calc_goal,
            ticker=ticker,
            start=start,
            end=end,
            freq=freq,
            signals=signals,
            ensemble_signals_with_params=ensemble_signals_with_params,
            metric=metric,
            goal=goal
        )

        result = gp_minimize(my_goal, my_bounds, n_calls=n_calls)
        
        return {
            "Optimized Params": result.x,
            "Optimized Value": result.fun
        }
    


    

"""
# ------- Example of registering signals -------

def sig_cdl_harami(ta_df: pd.DataFrame) -> pd.Series:
    '''Create signal for cdl_harami.'''
    def check_value(x: Union[int,float]) -> Union[int,float]:
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0
    return ta_df['CDL_HARAMI'].apply(check_value)

def sig_aberration(ta_df: pd.DataFrame) -> pd.Series:
    '''Create signal for aberration.'''
    def generate_signal(row1: pd.Series, row2: pd.Series) -> int:
        if row1['close'] < row1['ABER_XG_5_15'] and row2['close'] > row2['ABER_XG_5_15']:
            return 1
        elif row1['close'] > row1['ABER_SG_5_15'] and row2['close'] < row2['ABER_SG_5_15']:
            return -1
        else:
            return 0
    sig = ta_df.apply(lambda x: generate_signal(ta_df.iloc[ta_df.index.get_loc(x.name) - 1], ta_df.loc[x.name]) 
        if x.name > ta_df.index[0] else None, axis=1)
    sig.name = 'ABERRATION'
    return sig
    

ta_signal = TASignal()
ta_signal.register_signal('cdl_harami', 'cdl_harami', None, sig_cdl_harami)
ta_signal.register_signal('aberration', 'aberration', None, sig_aberration)    
"""