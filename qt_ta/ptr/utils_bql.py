#####
import pandas as pd 
import datetime as dt
import pytz
import re
from typing import Union

import bql 
bq = bql.Service()

#####
def _split_date_range(bq, start, end):
    """function to split the input date range into sub-ranges
    """
    test_ticker = 'JPY Curncy'
    test_data = bq.data.px_last(dates=bq.func.range(start, end))
    test_req = bql.Request(test_ticker, test_data)
    test_res = bq.execute(test_req)
    test_df = test_res[0].df()
    test_df.sort_values(by='DATE', ascending=True, inplace=True)
    list_dates = list(test_df['DATE'])
    
    start_date = list_dates[0]
    end_date = list_dates[-1]
    start_year = start_date.year
    end_year = end_date.year
    
    if end_year > start_year:
        return [y for y in range(start_year, end_year+1)]
    else:
        return [end_year]
    
    
def _on_request_error(*args):
    return None


def _invert_sign(sign):
    if sign=='-':
        return '+'
    if sign=='+':
        return '-'
    
    
def convert_tz(custom_timezone):
    """convert the eco calendar output timezone (i.e. GMT+08:00)
    to the pytz timezone obj (i.e. Etc/GMT-8)
    """
    sign = _invert_sign(re.findall(r'\-|\+', custom_timezone)[0])
    time = int(re.split(r'\-|\+|\:', custom_timezone)[1])
    return pytz.timezone(f'Etc/GMT{sign}{time}')


def round_event_datetime(datetime_obj, round_down=True):
    """function to round event_datetime to minutes
    """
    sec = datetime_obj.second
    if sec>0:
        if round_down:
            return datetime_obj + dt.timedelta(seconds=-sec)
        else:
            return datetime_obj + dt.timedelta(seconds=-sec, minutes=1)
    else:
        return datetime_obj
    

def get_eco_calendar_data(bq,
                          universe, 
                          start, 
                          end, 
                          tickers=None,
                          subtype='all',
                          relevancy='medium', 
                          view='extended',
                          event_timezone=None,
                          verbose=False):
    """funciton to request eco calendar for a long range
    
    Params:
    ----------------
    bq:
        bql service
        
    universe:
        bql universe 
        
    start, end: str
        start date, end date in yyyy-mm-dd format
        
    tickers: list
        list of eco tickers to match. default to None.
        
    subtype, relevancy, view: str
        BQL eco calendar parameters
        
    event_timezone: str
        pytz timeonze. Convert the RELEASE_DATE_TIME to specified timezone by adding additional columns including: event_datetime, event_date, event_time, event_timezone. default to None. 
        
    """
    
    if verbose:
        print("Splitting the date range ...")
    list_years = _split_date_range(bq, start, end)
    
    # make bql ECO request
    requests = []
    for y in list_years:
        if verbose:
            print(f"getting data for year {y}")
        start_date = f"{y}-01-01"
        end_date = f"{y}-12-31"
        
        
        calendar = bq.data.calendar(dates=bq.func.range(start_date, end_date), 
                                    subtype=subtype,
                                    relevancy=relevancy,
                                    view=view)
        if tickers is not None:
            # calendar_eco = calendar.matches(calendar['ticker']==ticker)
            calendar_eco = calendar.matches(calendar['ticker'].in_(tickers))
        else:
            calendar_eco = calendar
        request = bql.Request(universe, 
                              {'Calendar': calendar_eco}, 
                              preferences={'addcols': ['surprise', 'subtype']})
        requests.append(request)
    
    responses = list(bq.execute_many(requests, on_request_error=_on_request_error))
    df_calendar = pd.concat([response[0].df() for response in responses], axis=0)
    df_calendar = df_calendar.dropna(subset=['ACTUAL'])
    
    if event_timezone is not None:
        if 'TIMEZONE' in df_calendar.columns:
            default_timezone = convert_tz(df_calendar['TIMEZONE'].unique()[0])
            standard_timezone = pytz.timezone(event_timezone)
            # add event_datetime as the datetime in standardized timezone
            try:
                # convert tz-naive to tz-aware
                df_calendar['event_datetime'] = df_calendar['RELEASE_DATE_TIME'].apply(
                    lambda x: pd.to_datetime(x).tz_localize(default_timezone).tz_convert(standard_timezone))
            except:
                # convert timezone
                df_calendar['event_datetime'] = df_calendar['RELEASE_DATE_TIME'].apply(
                    lambda x: pd.to_datetime(x).tz_convert(default_timezone).tz_convert(standard_timezone))

            # round event_datetime to minutes
            df_calendar['event_datetime'] = df_calendar['event_datetime'].apply(lambda x: round_event_datetime(x))
            df_calendar['event_date'] = df_calendar['event_datetime'].apply(lambda x: x.date)
            df_calendar['event_time'] = df_calendar['event_datetime'].apply(lambda x: x.time)
            df_calendar['event_timezone'] = df_calendar['event_datetime'].apply(lambda x: x.tz)
        else:
            raise Exception("'TIMEZONE' column not found. Please set 'view' parameter to 'extended'.")
    
    return df_calendar


def get_id_to_figi_mapping(universe, 
                           figi_col_name='figi',
                           id_col_name='parsekey',
                           tickers_to_add_pcs=None):
    """function to map input id to figi
    """
    request = bql.Request(universe, {figi_col_name: bq.data.ID_BB_GLOBAL()})
    response = bq.execute(request)
    data = response.get(figi_col_name).df()
    figi_mapping = data.reset_index()

    if tickers_to_add_pcs is None:
        figi_mapping[figi_col_name] = figi_mapping.apply(
            lambda row: f'{row[figi_col_name]}.BGN' 
            if row['ID'].split(' ')[-1].lower() in ['curncy','corp']
            else row[figi_col_name], 
            axis=1)
    else:
        figi_mapping[figi_col_name] = figi_mapping.apply(
            lambda row: f'{row[figi_col_name]}.BGN' 
            if row['ID'] in tickers_to_add_pcs
            else row[figi_col_name], 
            axis=1)
    return figi_mapping.rename(columns={'ID':id_col_name})


def get_figi_to_id_mapping(universe, 
                           figi_col_name='figi',
                           id_col_name='short_name',
                           id_data_item='name'):
    """function to map figi to parsekey
    """
    request = bql.Request(universe, 
                          {id_col_name: getattr(bq.data, id_data_item)()})
    response = bq.execute(request)
    data = response.get(id_col_name).df()
    figi_mapping = data.reset_index()

    return figi_mapping.rename(columns={'ID':figi_col_name})