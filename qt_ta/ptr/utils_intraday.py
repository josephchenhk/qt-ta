#####
import warnings
import pandas as pd 
import datetime as dt
import pytz
import re
from pandas.tseries.offsets import BusinessDay, DateOffset
from datetime import timedelta

try:
    import bloomberg.bquant.intraday as bqia
except:
    warnings.warn("bloomberg.bquant.intraday has not been installed.")
import bql 
bq = bql.Service()

# import utils_bql as ut_bql
from .utils_bql import get_figi_to_id_mapping, round_event_datetime

#####

PACKAGES = ['macro', 'equity', 'fixed-income']


def get_id_mapping(dict_sec_pcs):
    """Function to get id mapping
    
    Params:
    ------------
    dict_sec_pcs:
        Dictionary of the input securities (non-futures).
        Key is the custom name (user-defined) of the security.
        Value is a tuple of (parsekey, pcs which is optional)
        
    Returns:
        DataFrame contains: custom, figi, parsekey
    """
    lst_custom = [x for x in dict_sec_pcs.keys()]
    lst_pk = [x[0] for x in dict_sec_pcs.values()]
    lst_pcs = [x[1] for x in dict_sec_pcs.values()]
    df_sec_pcs = pd.DataFrame(data={'custom': lst_custom,
                                    'parsekey': lst_pk,
                                     'pcs': lst_pcs})

    # request figi via bql
    univ = bq.univ.list(lst_pk)
    flds = {'figi': bq.data.id_bb_global().droppe()}
    req = bql.Request(univ, flds)
    res = bq.execute(req)
    df = res[0].df().reset_index()
    df.rename(columns={'ID': 'parsekey'}, inplace=True)

    # combine dataframe
    id_mapping = pd.merge(left=df_sec_pcs, left_on='parsekey',
                          right=df, right_on='parsekey')

    # add pcs
    id_mapping['figi'] = id_mapping.apply(
        lambda row: row['figi']+'.'+row['pcs'] if row['pcs']!='' else row['figi'],
        axis=1)
    id_mapping.drop(columns=['pcs'], inplace=True)
    
    return id_mapping


def get_roll_config(dict_fut):
    """Function to get the FuturesRollConfig
    
    Params:
    ------------
    dict_fut:
        Dictionary of the input futures. 
        Key is the custom name (user-defined) of the future roll.
        Value is a tuple of (future_root, yellow_key, roll_offset)
        
    Returns:
        FuturesRollFactory
    """
    roll_configs = {}
    
    for key, val in dict_fut.items():
        future_root, yellow_key, roll_offset = val
        roll_configs[key] = bqia.FuturesRollConfig(future_root, 
                                                   yellow_key,
                                                   DateOffset(roll_offset))
    
    return bqia.FuturesRollFactory(roll_configs)


def set_intraday_config(id_mapping=None,
                        roll_config=None,
                        packages=['equity','macro','fixed-income']):
    """function to set bqintraday config
    """
    bqia_configs = {}
    if id_mapping is not None:
        bqia_configs['security_id_mapping'] = id_mapping
    if roll_config is not None:
        bqia_configs['futures_roll_factory'] = roll_config
    bqia_configs['enabled_packages'] = packages
    bqia.set_config(bqia_configs)
            
        
def get_intraday_bars(dict_sec_pcs={"spx":("spx index","")},
                      dict_fut=None,
                      start='2024-06-01',
                      end='2024-07-01',
                      security_id_type='custom',
                      downsample_freq=None,
                      tz=None,
                      packages=['equity','macro','fixed-income'],
                      add_description=True):
    list_sec = []
    
    # get id_mapping
    if dict_sec_pcs is not None:
        id_mapping = get_id_mapping(dict_sec_pcs)
        list_sec += list(id_mapping[security_id_type])
    else:
        id_mapping = None
        
    # get roll config
    if dict_fut is not None:
        roll_config = get_roll_config(dict_fut)
        list_sec += list(dict_fut.keys())
    else:
        roll_config = None
        
    # set intraday config
    set_intraday_config(id_mapping, roll_config, packages)
    
    # request bars
    df_bars = bqia.get_bars(securities=list_sec,
                            security_id_type=security_id_type,
                            start=start,
                            end=end,
                            tz=tz,
                            downsample_freq=downsample_freq,
                           )
    
    if add_description:
        
        # add short_name and parskey
        figis = list(df_bars['security'].unique())
        
        # convert figi --> short_name
        figi_to_name = get_figi_to_id_mapping(figis, 
                                                     id_col_name='short_name', 
                                                     id_data_item='name')
        figi_to_name_mapping = figi_to_name.set_index('figi')['short_name'].to_dict()
        
        # convert figi --> parsekey
        figi_to_pk = get_figi_to_id_mapping(figis, 
                                                   id_col_name='parsekey', 
                                                   id_data_item='parsekyable_des')
        figi_to_pk_mapping = figi_to_pk.set_index('figi')['parsekey'].to_dict()
        
        # add short_name and parsekey columns to the bars
        df_bars['short_name'] = df_bars['security'].apply(
            lambda x: x if figi_to_name_mapping[x] is None\
                        else figi_to_name_mapping[x])
        df_bars['parsekey'] = df_bars['security'].apply(
            lambda x: x if figi_to_pk_mapping[x] is None\
                        else figi_to_pk_mapping[x])
        
    return df_bars
    
    
def get_event_metadata(df_calendar,
                       pre_event_window=-5,
                       post_event_window=10):
    # Get datetime of bars around the event datetime
    temp_data = []
    for event in df_calendar['EVENT_NAME'].unique():

        # filter calendar for each event
        df_calendar_subset = df_calendar[df_calendar['EVENT_NAME']==event]

        # get unique event time, rounded to minutes
        list_event_datetime = list(df_calendar_subset['event_datetime'].unique())
        list_event_datetime = [round_event_datetime(x) for x in list_event_datetime]

        for event_datetime in list_event_datetime:
            for m in range(pre_event_window, post_event_window+1):
                bar_datetime = event_datetime + timedelta(minutes=m)
                temp_data.append((event, event_datetime, bar_datetime, m))
                
    list_col_names = ['EVENT_NAME','event_datetime','datetime', 'time_diff']
    df_event_metadata = pd.DataFrame(temp_data, columns=list_col_names)
    
    return df_event_metadata


def filter_bars_around_events(df_bars,
                              df_calendar,
                              pre_event_window=-5,
                              post_event_window=10):
    # Get event metadata
    df_event_metadata = get_event_metadata(df_calendar, 
                                           pre_event_window, 
                                           post_event_window)
    
    # Filter bars around the event release time
    temp = []
    left = df_event_metadata.set_index('datetime')
    right = df_bars.set_index('datetime')
    df_event_bars = left.join(right, how='left').reset_index()
    df_event_bars.dropna(subset=['key'], inplace=True)
    
    # select useful columns
    df_event_bars = df_event_bars[list(left.columns)+['datetime']+list(right.columns)]
    df_event_bars.sort_values(by=['EVENT_NAME','event_datetime','key','datetime'], inplace=True)
    df_event_bars.reset_index(drop=True, inplace=True)
    
    # combine event bars data with Eco calendar data
    left = df_event_bars.set_index(['EVENT_NAME','event_datetime'])
    right = df_calendar.reset_index().set_index(['EVENT_NAME','event_datetime'])
    df_data = left.join(right, how='left').reset_index()
    
    return df_data


    
    
    
    
        
    


