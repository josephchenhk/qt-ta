import pandas as pd
import plotly.graph_objects as go
from qt_ta.ta import ta

def generate_candle_chart(ohlcv_df: pd.DataFrame, convert_xaxis: bool=False):
    if convert_xaxis:
        x = list(range(len(ohlcv_df.index)))
    else:
        x = ohlcv_df.index
    # Create candlestick figure
    fig = go.Figure(
        data=[go.Candlestick(
            x=x,
            open=ohlcv_df['open'],
            high=ohlcv_df['high'],
            low=ohlcv_df['low'],
            close=ohlcv_df['close'])
        ]
    )

    # Customize figure layout
    fig.update_layout(
        title='OHLCV Candle Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(rangeslider=dict(visible=False)),  # Hide the slider
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    ) 

    # Show the chart
    # fig.show()
    return fig


def generate_line_chart(price_srs: pd.Series, convert_xaxis: bool=False):
    if convert_xaxis:
        x = list(range(len(price_srs.index)))
    else:
        x = price_srs.index
    # Create candlestick figure
    fig = go.Figure(
        data=[go.Scatter(
            x=x,
            y=price_srs.values,
            mode='lines'
        )
        ]
    )

    # Customize figure layout
    fig.update_layout(
        title='Line Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis=dict(rangeslider=dict(visible=False)),  # Hide the slider
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    ) 

    # Show the chart
    # fig.show()
    return fig

def generate_volume_chart(ohlcv_df: pd.DataFrame, convert_xaxis: bool=False, convert_yaxis: bool=False):
    if convert_xaxis:
        x = list(range(len(ohlcv_df.index)))
    else:
        x = ohlcv_df.index
    
    if convert_yaxis:
        y = ohlcv_df['volume'] / ohlcv_df['volume'].iloc[-1]
    else:
        y = ohlcv_df['volume']
    
    # Create candlestick figure
    fig = go.Figure(
        data=[go.Bar(
            x=x,
            y=y,
            name='volume'
        )
        ]
    )

    # Customize figure layout
    fig.update_layout(
        title='Volume Chart',
        xaxis_title='Date',
        yaxis_title='Volume',
        xaxis=dict(rangeslider=dict(visible=False)),  # Hide the slider
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    ) 

    # Show the chart
    # fig.show()
    return fig


def generate_macd_chart(ohlcv_df: pd.DataFrame, start: str=None, end: str=None):
    err, macd_df = ohlcv_df.ta.macd()
    if err.code != 0:
        raise ValueError('Failed to calculate MACD on the time series.')
    # Trim the period   
    if start is not None:
        macd_df = macd_df[macd_df.index >= start]
    if end is not None:
        macd_df = macd_df[macd_df.index <= end]
    # Create TA chart based on candlesticks
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=macd_df.index,
            y=macd_df['MACDh_12_26_9'],
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macd_df.index,
            y=macd_df['MACD_12_26_9'],
            line_color='dimgray',
            opacity=0.8
        )
    )

    fig.add_trace(
        go.Scatter(
            x=macd_df.index,
            y=macd_df['MACDs_12_26_9'],
            line_color='deepskyblue',
            opacity=0.8
        )
    )

    # fig.show()
    return fig