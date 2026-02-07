from typing import List
from typing import Optional
import pandas as pd
from datetime import datetime
from datetime import timedelta

from .utils_intraday import get_intraday_bars
import bql
bq = bql.Service()

def get_bql_ohlcv(
        ticker: str, 
        start: str, 
        end: str, 
):
    open_ = bq.data.px_open()
    high = bq.data.px_high()
    low = bq.data.px_low()
    close_ = bq.data.px_last()
    volume = bq.data.px_volume()
    req = bql.Request(
        ticker,
        {
            'open': open_,
            'high': high,
            'low': low,
            'close': close_,
            'volume': volume
        },
        with_params={'dates': bq.func.range(start, end)}
    )
    resp = bq.execute(req)
    ohlcv_df = pd.concat([r.df().set_index('DATE')[[r.name]] for r in resp], axis=1).dropna(how='all')
    return ohlcv_df

def get_ohlcv(
        ticker: str, 
        start: str, 
        end: str, 
        source: str='bql', 
        tz: str='Asia/Hong_Kong', 
        downsample_freq: Optional[str]=None,
        packages: List[str]=['equity','macro','fixed-income']
) -> pd.DataFrame:
    assert source in ('bql', 'bqia'), 'Param `source` is not valid, only "bql" or "bqia" is allowed.'
    if source == 'bql':
        query_start = start
        batch_size = 6000
        ohlcv_lst = []
        while query_start < end:
            query_end = (datetime.strptime(query_start, '%Y-%m-%d') + timedelta(days=batch_size)).strftime('%Y-%m-%d')
            if query_end > end:
                query_end = end
            ohlcv_df_ = get_bql_ohlcv(ticker, query_start, query_end)
            ohlcv_lst.append(ohlcv_df_)
            # print(query_start, query_end)
            query_start = (datetime.strptime(query_end, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d') 
        ohlcv_df = pd.concat(ohlcv_lst)
    elif source == 'bqia':
        df_bars = get_intraday_bars(
            dict_sec_pcs={ticker:(ticker,"BGN")},
            dict_fut=None,
            start=start,
            end=end,
            security_id_type='custom',
            downsample_freq=downsample_freq,
            tz=tz,
            packages=packages,
            add_description=False
        )
        ohlcv_df = df_bars.set_index('datetime')[['open', 'high', 'low', 'close', 'volume']]
    return ohlcv_df