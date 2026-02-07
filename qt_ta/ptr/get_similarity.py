from typing import List
from typing import Tuple
from typing import Dict
from typing import Optional
from typing import Any
from typing import Union
from functools import partial
import io
import multiprocessing

from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import imagehash

from qt_ta.ta import ta
from qt_token import TokenCheckMeta

TOKEN_KEY = "qt_ta/ptr"

from typing import Callable
import pandas as pd

def _check_overlap(
    range_a: List[Union[float,int]], 
    range_b: List[Union[float, int]], 
    overlap_threshold: float=0.5
):
    """
    Checks if two ranges have a specified overlap percentage.

    Args:
        range_a: A list [a1, a2] representing the first range.
        range_b: A list [b1, b2] representing the second range.
        overlap_threshold: The minimum overlap percentage required (default 0.3).

    Returns:
        True if the overlap is at or above the threshold, False otherwise.
    """

    a1, a2 = range_a
    b1, b2 = range_b

    # Calculate the intersection of the ranges
    intersection_start = max(a1, b1)
    intersection_end = min(a2, b2)

    intersection_length = max(0, intersection_end - intersection_start)

    # Calculate the lengths of the original ranges
    range_a_length = a2 - a1
    range_b_length = b2 - b1

    # Calculate the overlap percentage
    if range_a_length == 0 or range_b_length ==0:
        return False

    overlap_percentage_a = intersection_length / range_a_length
    overlap_percentage_b = intersection_length / range_b_length

    # Check if either overlap percentage meets the threshold
    return overlap_percentage_a >= overlap_threshold or overlap_percentage_b >= overlap_threshold

def _generate_chart(
    ohlcv_df: pd.DataFrame, 
    start: str,
    end: str,
    normalise_xaxis: bool=False, 
    normalise_yaxis: List[str]=[],
    ta_indicators: Optional[List[Tuple[str, Dict[str, Any]]]]=[],
    charts: List[Tuple[str, Dict[str, str], Dict[str, Any]]]=[],
    title: str='',
    xaxis_title: str='',
    yaxis_title: str='',
    output: str='go.Figure'
) -> Union[go.Figure, PngImageFile]:
    """
    Generates a Plotly figure from OHLCV data, incorporating technical indicators and custom charts.

    Args:
        ohlcv_df       : DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
                         Index should represent the time series.
        start          : Start date/time for the chart's display window (inclusive).
        end            : End date/time for the chart's display window (inclusive).
        normalise_xaxis: If True, normalizes the x-axis to a range of integers. Defaults to False.
        normalise_yaxis: List of column names to normalize the y-axis by dividing by the last value. Defaults to [].
        ta_indicators  : List of tuples, where each tuple specifies a technical indicator
                         and its parameters. Defaults to []. Example: [("sma", {"length": 20})].
        charts         : List of tuples, where each tuple defines a chart to plot.
                         Each tuple contains:
                         - Chart type (e.g., "Candlestick", "Scatter").
                         - Data parameters (dict mapping Plotly argument names to DataFrame column names).
                         - Plot parameters (dict with additional Plotly arguments).
                         Defaults to [].
        title          : Title of the chart. Defaults to ''.
        xaxis_title    : Title of the x-axis. Defaults to ''.
        yaxis_title    : Title of the y-axis. Defaults to ''.
        output         : The return format. Defaults to 'go.Figure'. Alternative option is 'PngImageFile'.

    Kwargs:
        Any additional keyword arguments that can be passed to Plotly's `update_layout` method.

    Returns:
        go.Figure: A Plotly Figure object representing the generated chart. Or PngImageFile

    Raises:
        ValueError: If an error occurs during the calculation of a technical indicator.
    """
    # TA indicators
    for ta_indicator, ta_parameters in ta_indicators:
        ta_err, ta_df = getattr(ohlcv_df.ta, ta_indicator)(**ta_parameters)
        if ta_err.code != 0:
            raise ValueError(f'Error in {ta_indicator}: {ta_err.msg}.')
        ohlcv_df = pd.concat([ohlcv_df, ta_df], axis=1)

    # truncate the display window
    chart_ohlcv_df = ohlcv_df[
        (ohlcv_df.index >= start) & (ohlcv_df.index <= end)
    ].copy()
 
    # Normalise x
    if normalise_xaxis:
        x = list(range(len(chart_ohlcv_df.index)))
    else:
        x = chart_ohlcv_df.index

    # Normalise y
    for field in normalise_yaxis:
        chart_ohlcv_df[field] = chart_ohlcv_df[field] / chart_ohlcv_df[field].iloc[-1]

    # plot the charts
    fig = go.Figure()
    for chart_type, data_params, plot_params in charts:
        chart = getattr(go, chart_type)
        data_dict = {k: list(chart_ohlcv_df[v].values) for k,v in data_params.items()}
        plot_params.update(data_dict)
        fig.add_trace(
            chart(
                x=x,
                **plot_params
            )
        )

    # Customize figure layout
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        xaxis=dict(rangeslider=dict(visible=False)),  # Hide the slider
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
    ) 

    # convert format if necessary
    if output == 'PngImageFile':
        # Export the figure to PNG bytes
        img_bytes = fig.to_image(format="png")
        # Load the PNG bytes into a PIL Image object
        img_pil = Image.open(io.BytesIO(img_bytes))
        return img_pil
    return fig


def _combine_plotly_figures(figures: go.Figure, rows: int, cols: int, title: str='') -> go.Figure:
    """
    Combines multiple Plotly Figure objects into an M x N subplot grid, keeping only the first subplot's legend.

    Args:
        figures: A list of Plotly Figure objects.
        rows   : The number of rows in the subplot grid.
        cols   : The number of columns in the subplot grid.

    Returns:
        A combined Plotly Figure object, or None if there's an error.
    """

    if not isinstance(figures, list) or not all(isinstance(fig, go.Figure) for fig in figures):
        print("Error: 'figures' must be a list of Plotly Figure objects.")
        return None

    if rows * cols < len(figures):
        print(f"Error: The grid size (rows * cols={rows}*{cols}={rows*cols}) is smaller " +
            f"than the number of figures ({len(figures)}).")
        return None

    try:
        fig = make_subplots(
            rows=rows, 
            cols=cols, 
            subplot_titles=[f"Target"] + [f"Figure ({i})" for i in range(len(figures)-1)]
        )

        for i, figure in enumerate(figures):
            row = i // cols + 1  # Calculate row index (1-based)
            col = i % cols + 1   # Calculate column index (1-based)

            for j, trace in enumerate(figure.data):
                if i == 0:
                    fig.add_trace(trace, row=row, col=col)
                else:
                    trace.showlegend = False
                    fig.add_trace(trace, row=row, col=col)

            # Update layout (optional, handle titles, axes, etc.)
            if figure.layout.xaxis:
              fig.update_xaxes(title_text=figure.layout.xaxis.title.text, row=row, col=col, rangeslider_visible=False)
            if figure.layout.yaxis:
              fig.update_yaxes(title_text=figure.layout.yaxis.title.text, row=row, col=col)

        fig.update_layout(
            height=rows * 400, 
            width=cols * 600, 
            title_text=title,
            plot_bgcolor='rgba(0, 0, 0, 0)',
            paper_bgcolor='rgba(0, 0, 0, 0)',
        ) 

        return fig

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def rotate_curve(x, y, angle_degrees, center_x=0, center_y=0):
    """Rotates a curve defined by x and y coordinates by a given angle around a specified center.

    Args:
        x: A NumPy array or list of x-coordinates.
        y: A NumPy array or list of y-coordinates.
        angle_degrees: The angle of rotation in degrees.
        center_x: The x-coordinate of the rotation center (default: 0).
        center_y: The y-coordinate of the rotation center (default: 0).

    Returns:
        A tuple containing two NumPy arrays: the rotated x and y coordinates.
    """

    angle_radians = np.deg2rad(angle_degrees)
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Translate to the origin
    x_translated = x - center_x
    y_translated = y - center_y

    # Apply rotation
    coordinates = np.array([x_translated, y_translated])
    rotated_coordinates = rotation_matrix @ coordinates

    # Translate back
    rotated_x = rotated_coordinates[0, :] + center_x
    rotated_y = rotated_coordinates[1, :] + center_y

    return rotated_x, rotated_y

def errors_after_rotation(angle_degrees, target_y, x, y, center_x=0, center_y=0):
    rx, ry = rotate_curve(
        x=x,
        y=y,
        angle_degrees=angle_degrees,
        center_x=center_x,
        center_y=center_y
    )
    func_interp = interp1d(rx, ry, kind='linear') 
    
    sum_errors = 0
    sum_errors += (ry[0] - target_y[0])**2
    sum_errors += (ry[-1] - target_y[-1])**2
    for i in range(1, len(ry)-1):
        interp_y = func_interp(x[i])
        sum_errors += (interp_y - target_y[i])**2
    return sum_errors

def get_rotated_stats(target_nav: pd.Series, nav_stats: pd.DataFrame, fwd_window: int=50) -> pd.DataFrame:

    center_x = nav_stats.iloc[:-fwd_window].index[-1]
    center_y = 1
    find_rotation_angles = partial(
        errors_after_rotation,
        target_y=target_nav,     
        x=nav_stats.index[:len(target_nav)],
        y=nav_stats['wavg'][:len(target_nav)],
        center_x=center_x,
        center_y=center_y
    )
    result = minimize_scalar(find_rotation_angles)  
    if not result.success:
        print('Failed to find the gauge invariant rotation angle.')

        
    min_angles = result.x
    min_error = result.fun  # Minimum value of function find_rotation_angles(x)

    # use min_angle to rotate all three curves
    rx_min, ry_min = rotate_curve(
        x=nav_stats.index,
        y=nav_stats['min'],
        angle_degrees=min_angles,
        center_x=center_x,
        center_y=center_y
    )
    
    rx_max, ry_max = rotate_curve(
        x=nav_stats.index,
        y=nav_stats['max'],
        angle_degrees=min_angles,
        center_x=center_x,
        center_y=center_y
    )
    
    rx_wavg, ry_wavg = rotate_curve(
        x=nav_stats.index,
        y=nav_stats['wavg'],
        angle_degrees=min_angles,
        center_x=center_x,
        center_y=center_y
    )

    rnav_stats = pd.DataFrame({
        'rx_min': rx_min, 
        'ry_min': ry_min, 
        'rx_max': rx_max, 
        'ry_max': ry_max, 
        'rx_wavg': rx_wavg, 
        'ry_wavg': ry_wavg
    })

    return rnav_stats

def _combine_nav_plots(target_nav: pd.Series, nav: pd.DataFrame, nav_stats: pd.DataFrame, rnav_stats: pd.DataFrame) -> go.Figure:
    """
    Combines three plotly plots into a 3x1 subplot layout.

    Args:
        
        target_nav  : Series for the target.
        nav         : DataFrame of time series of the similar paths.
        nav_stats   : DataFrame of time series of the statistics (min, max, wavg) of similar paths.
        rnav_stats  : DataFrame of time series of the rotated statistics (rx_min, rx_max, rx_wavg, ...) of similar paths.

    Returns:
        A combined plotly Figure object.
    """
    target_start = target_nav.index[0].strftime('%Y-%m-%d')
    target_end = target_nav.index[-1].strftime('%Y-%m-%d')
    rx_min = rnav_stats['rx_min'].values
    ry_min = rnav_stats['ry_min'].values
    rx_max = rnav_stats['rx_max'].values
    ry_max = rnav_stats['ry_max'].values
    rx_wavg = rnav_stats['rx_wavg'].values
    ry_wavg = rnav_stats['ry_wavg'].values

    fig = make_subplots(rows=3, cols=1, subplot_titles=(f'NAV', f'NAV Stats', f'Rescaled'))

    # First Subplot (NAV)
    fig.add_trace(
        go.Scatter(
            x=list(nav.index[:len(target_nav)]),
            y=list(target_nav.values),
            mode='lines',
            line=dict(width=4, color='grey'),
            name='cur:' + str((target_start, target_end))
        ),
        row=1, col=1
    )

    for col in nav.columns:
        fig.add_trace(
            go.Scatter(
                x=list(nav.index),
                y=list(nav[col].values),
                mode='lines',
                name=str(col)
            ),
            row=1, col=1
        )

    # Second Subplot (NAV Stats)
    fig.add_trace(
        go.Scatter(
            x=list(nav_stats.index[:len(target_nav)]),
            y=list(target_nav),
            mode='lines',
            line=dict(width=4, color='grey'),
            name='cur:' + str((target_start, target_end))
        ),
        row=2, col=1
    )

    for col in nav_stats.columns:
        if col == 'min' or col == 'max':
            data = go.Scatter(
                x=list(nav_stats.index),
                y=list(nav_stats[col]),
                mode='lines',
                line=dict(dash='dash'),
                name=str(col)
            )
        else:
            data = go.Scatter(
                x=list(nav_stats.index),
                y=list(nav_stats[col]),
                mode='lines',
                name=str(col)
            )
        fig.add_trace(data, row=2, col=1)

    # Third Subplot (Rescaled)
    fig.add_trace(
        go.Scatter(
            x=list(nav.index[:len(target_nav)]),
            y=list(target_nav.values),
            mode='lines',
            line=dict(width=4, color='grey'),
            name='cur:' + str((target_start, target_end))
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(rx_min),
            y=list(ry_min),
            mode='lines',
            line=dict(dash='dash'),
            name='rescaled min'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(rx_max),
            y=list(ry_max),
            mode='lines',
            line=dict(dash='dash'),
            name='rescaled max'
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=list(rx_wavg),
            y=list(ry_wavg),
            mode='lines',
            name='rescaled wavg'
        ),
        row=3, col=1
    )

    # Update Layout
    fig.update_layout(
        xaxis1=dict(rangeslider=dict(visible=False)),
        xaxis2=dict(rangeslider=dict(visible=False)),
        xaxis3=dict(rangeslider=dict(visible=False)),
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=1600, #Adjust height as needed
        width=1800, #Adjust width as needed
    )

    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Price', row=2, col=1)
    fig.update_yaxes(title_text='Price', row=3, col=1)

    return fig
    
class LazyChartGenerator:
    """Lazy Chart generator to plot figure from OHLCV data, incorporating technical indicators and custom charts.
    """
    
    def __init__(
        self,  
        ohlcv_df: pd.DataFrame, 
        start: str,
        end: str,
        generate_chart: Optional[Callable]=None,
        normalise_xaxis: bool=False, 
        normalise_yaxis: List[str]=[],
        ta_indicators: Optional[List[Tuple[str, Dict[str, Any]]]]=[],
        charts: List[Tuple[str, Dict[str, str], Dict[str, Any]]]=[],
        title: str='',
        xaxis_title: str='',
        yaxis_title: str='',
        output: str='go.Figure',
    ):
        """
        Generates a Plotly figure from OHLCV data, incorporating technical indicators and custom charts.

        Args:
            ohlcv_df       : DataFrame containing OHLCV (Open, High, Low, Close, Volume) data.
                             Index should represent the time series.
            start          : Start date/time for the chart's display window (inclusive).
            end            : End date/time for the chart's display window (inclusive).

        Kwargs:
            generate_chart : The chart generating function.
            normalise_xaxis: If True, normalizes the x-axis to a range of integers. Defaults to False.
            normalise_yaxis: List of column names to normalize the y-axis by dividing by the last value. Defaults to [].
            ta_indicators  : List of tuples, where each tuple specifies a technical indicator
                             and its parameters. Defaults to []. Example: [("sma", {"length": 20})].
            charts         : List of tuples, where each tuple defines a chart to plot.
                             Each tuple contains:
                             - Chart type (e.g., "Candlestick", "Scatter").
                             - Data parameters (dict mapping Plotly argument names to DataFrame column names).
                             - Plot parameters (dict with additional Plotly arguments).
                             Defaults to [].
            title          : Title of the chart. Defaults to ''.
            xaxis_title    : Title of the x-axis. Defaults to ''.
            yaxis_title    : Title of the y-axis. Defaults to ''.
            output         : The return format. Defaults to 'go.Figure'. Alternative option is 'PngImageFile'.

        Kwargs:
            Any additional keyword arguments that can be passed to Plotly's `update_layout` method.

        Returns:
            go.Figure: A Plotly Figure object representing the generated chart. Or PngImageFile

        Raises:
            ValueError: If an error occurs during the calculation of a technical indicator.
        """
        if generate_chart is None:
            self.generate_chart = _generate_chart
        else:
            self.generate_chart = generate_chart
        self.ohlcv_df = ohlcv_df
        self.start = start
        self.end = end
        self.normalise_xaxis = normalise_xaxis
        self.normalise_yaxis = normalise_yaxis
        self.ta_indicators = ta_indicators
        self.charts = charts
        self.title = title
        self.xaxis_title = xaxis_title
        self.yaxis_title = yaxis_title
        self.output = output 

    def get_parameters(self) -> Dict[str, Any]:
        """Get the parameters.
        """
        return {
            "ohlcv_df": self.ohlcv_df,
            "start": self.start,
            "end": self.end,
            "normalise_xaxis": self.normalise_xaxis,
            "normalise_yaxis": self.normalise_yaxis,
            "ta_indicators": self.ta_indicators,
            "charts": self.charts, 
            "title": self.title, 
            "xaxis_title": self.xaxis_title, 
            "yaxis_title": self.yaxis_title, 
            "output": self.output,
        }

    def execute(self, **kwargs):
        """Generate the chart with the given parameters.
        """
        if len(kwargs) == 0:
            params = self.get_parameters()
            # print(f'debug: 1 {len(params)}')
            result = self.generate_chart(**params)
        elif kwargs.keys() == self.get_parameters().keys():
            # print(f'debug: 2 {len(kwargs)}')
            result = self.generate_chart(**kwargs)
        else:    
            params = self.get_parameters()
            for k, v in params.items():
                if k not in kwargs:
                    kwargs[k] = v
            # print(f'debug: 3 {len(kwargs)}')
            result = self.generate_chart(**kwargs)
        return result

    def __call__(self, **kwargs): 
        """Generate the chart with the given parameters (allow the object to be called directly).

        Any input in kwargs would override the original parameters.
        """
        return self.execute(**kwargs)


class PatternRecognition(metaclass=TokenCheckMeta(token_key=TOKEN_KEY)):
    """Pattern Recognition module is used to identify similar patterns.
    """
    def __init__(self, token: str='', chart: Optional[LazyChartGenerator]=None):
        """Initializer of Pattern Recognition Algo.

        Kwargs:
            token: The valid token to use this module.
            chart: The target chart generator defined by the user.
        """
        self.token = token
        if chart is None:
            raise ValueError('The required user input `chart` is missing. User must define the chart first.')
        if not isinstance(chart, LazyChartGenerator):
            raise TypeError('The user input `chart` must be of type `LazyChartGenerator`.')
        self.chart = chart
        self._input_ohlcv_df = {}
        self._output_similarity_df = {}
        self._output_top_similarity_df = {}

    
    def plot_chart(self, **kwargs):
        """Generate the chart with the given parameters.
        """
        ori_params = self.chart.get_parameters()
        override_params = {k: v for k, v in ori_params.items()}
        for k, v in kwargs.items():
            override_params[k] = v
        return self.chart(**override_params)
    

    @staticmethod
    def get_similarity_score(image1: PngImageFile, image2: PngImageFile, hash_size: int=8) -> float:
        """
        Compares two images and returns a similarity score between 0 and 1.

        Args:
            image1: The first image.
            image2: The second image.

        Returns:
            A float value between 0 and 1 representing the similarity score.
        """

        # Load images
        image1 = image1.convert('L')  # Convert to grayscale
        image2 = image2.convert('L')

        # Calculate average hash
        hash1 = imagehash.average_hash(image1, hash_size=hash_size)
        hash2 = imagehash.average_hash(image2, hash_size=hash_size)
        # Calculate Hamming distance between hashes
        distance_avg = hash1 - hash2
        max_distance_avg = len(hash1.hash) * hash_size
        similarity_avg = 1 - distance_avg/max_distance_avg

        # Calculate Perceptual hash
        hash1 = imagehash.phash(image1, hash_size=hash_size)
        hash2 = imagehash.phash(image2, hash_size=hash_size)
        # Calculate Hamming distance between hashes
        distance_p = hash1 - hash2
        max_distance_p = len(hash1.hash) * hash_size
        similarity_p = 1 - distance_p/max_distance_p

        # Calculate Difference hash
        hash1 = imagehash.dhash(image1, hash_size=hash_size)
        hash2 = imagehash.dhash(image2, hash_size=hash_size)
        # Calculate Hamming distance between hashes
        distance_d = hash1 - hash2
        max_distance_d = len(hash1.hash) * hash_size
        similarity_d = 1 - distance_d/max_distance_d

        # Calculate Wavelet hash
        hash1 = imagehash.whash(image1, hash_size=hash_size)
        hash2 = imagehash.whash(image2, hash_size=hash_size)
        # Calculate Hamming distance between hashes
        distance_w = hash1 - hash2
        max_distance_w = len(hash1.hash) * hash_size
        similarity_w = 1 - distance_w/max_distance_w

        # Calculate Difference hash
        hash1 = imagehash.colorhash(image1, binbits=hash_size)
        hash2 = imagehash.colorhash(image2, binbits=hash_size)
        # Calculate Hamming distance between hashes
        distance_color = hash1 - hash2
        max_distance_color = len(hash1.hash) * hash_size
        similarity_color = 1 - distance_color/max_distance_color

        # Normalize distance to a similarity score between 0 and 1
        # similarity = 1 - (distance / (2**hash_size - 1))  # 2^hash_size - 1 is the maximum possible hash value
        similarity = (similarity_avg + similarity_p + similarity_d + similarity_w) * 0.125 + similarity_color * 0.5

        return similarity

    def _compute_sample_similarity(
            self, 
            i: int, 
            window: int, 
            search_ohlcv_df: pd.DataFrame, 
            target_img: PngImageFile
    ) -> Tuple[str, str, int, int, float]:
        """Compute the similarity score of a given sample.

        Args:
            i              : The index in `search_ohlcv_df`.
            window         : The display window (number of timesteps, i.e., x-axis) in the chart(image).
            search_ohlcv_df: The OHLCV dataframe.
            target_img     : The target chart(image) that we want to compare with.

        Returns:
            The five key metrics when computing the similarity score:
            ------------------------------------------------------
            | start	| end |	start_index	| end_index | similarity |
            ------------------------------------------------------
        """
        sample_ohlcv_df = search_ohlcv_df.iloc[i-window:i]
        sample_start, sample_end = sample_ohlcv_df.index[0].strftime('%Y-%m-%d'), sample_ohlcv_df.index[-1].strftime('%Y-%m-%d')
        sample_img = self.plot_chart(start=sample_start, end=sample_end)
        similarity = self.get_similarity_score(image1=target_img, image2=sample_img, hash_size=8)
        return sample_start, sample_end, i-window, i-1, similarity
    
    def compute_similarities(self, **kwargs) -> pd.DataFrame: 
        """Computing the similarities of the given dataset.

        Kwargs:
            dataset: The dataset to be computed (defaults to `default`).
            ohlcv_df: The OHLCV time series data.

        Return:
            The dataframe with the similarity scores. The five columns of the results are:
            ------------------------------------------------------
            | start	| end |	start_index	| end_index | similarity |
            ------------------------------------------------------
            where the `start_index` and `end_index` are corresponding to the indices in `ohlcv_df`.
        """
        target_start = self.chart.get_parameters()['start']
        target_end = self.chart.get_parameters()['end']
        target_ohlcv_df = self.chart.get_parameters()['ohlcv_df']
        target_ohlcv_df = target_ohlcv_df[
            (target_ohlcv_df.index>=target_start)
            & (target_ohlcv_df.index<=target_end)
        ].copy()
        window = target_ohlcv_df.shape[0]

        if 'dataset' in kwargs:   
            dataset_name = kwargs['dataset']
            if dataset_name == 'default':
                print('Warning: You are overriding the `default` dataset!')
        else:
            dataset_name = 'default'

        if 'ohlcv_df' in kwargs:
            ohlcv_df = kwargs['ohlcv_df']
        else:
            ohlcv_df = self.chart.get_parameters()['ohlcv_df']
        search_ohlcv_df = ohlcv_df[(ohlcv_df.index < target_start)]
        target_img = self.plot_chart()
        print(f'Calculate similarity scores before {target_start} ...')
        
        indices = range(window, search_ohlcv_df.shape[0])
        
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = list(
                tqdm(
                    pool.imap(
                        partial(
                            self._compute_sample_similarity, 
                            window=window, 
                            search_ohlcv_df=search_ohlcv_df, 
                            target_img=target_img
                        ), 
                        indices
                    ), 
                    total=len(indices)
                )
            )
        
        result_df = pd.DataFrame(results, columns=['start', 'end', 'start_index', 'end_index', 'similarity'])
        similarity_df = result_df.sort_values(by='similarity', ascending=False).reset_index(drop=True)
        self._input_ohlcv_df[dataset_name] = ohlcv_df
        self._output_similarity_df[dataset_name] = similarity_df
        return similarity_df

    def find_similar_charts(
            self, 
            dataset_name: str='default', 
            number_of_charts: int=9, 
            overlap_threshold: float=0.5
    ):
        """Find the similar charts/patterns in the given dataset.

        Kwargs:
            dataset_name: The chosen dataset.
            number_of_charts: The selected number of charts.
            overlap_threshold: The overlap threshold when choosing the distinct charts (valid values from 0 to 1,
                               smaller value means less overlap; bigger value means more overlap).

        Returns:
            The top-{number_of_charts} similar patterns in the searching dataset.
        """
        if dataset_name not in self._output_similarity_df:
            raise ValueError(f'`dataset_name`=\"{dataset_name}\" is not valid. ' +
                f'You need to run `ptr.compute_similarities(dataset=\"{dataset_name}\")` first.'
            )
        sorted_df_similarity = self._output_similarity_df[dataset_name]
        N = int(number_of_charts)

        similar_charts_indices = []
        selected_ranges = []
        for i, row in sorted_df_similarity.iterrows():
            if len(similar_charts_indices) >= N:
                break
            start_index = row['start_index']
            end_index = row['end_index']
            range_a = [start_index, end_index]
            has_overlap = False
            for range_b in selected_ranges:
                if _check_overlap(range_a, range_b, overlap_threshold=overlap_threshold):
                    has_overlap = True
                    break
            if not has_overlap:
                selected_ranges.append(range_a)
                similar_charts_indices.append(i)
        top_similarity_df = sorted_df_similarity.loc[similar_charts_indices].reset_index(drop=True).copy()
        self._output_top_similarity_df[dataset_name] = top_similarity_df
        return top_similarity_df

    def plot_similar_charts(
            self, 
            rows: int, 
            cols: int, 
            dataset_name: str='default'
    ) -> go.Figure:
        """Plot the similar charts and display in subplots (rows x cols).

        Args:
            rows: The row number of the subplots.
            cols: The column number of the subplots.

        Kwargs:
            dataset_name: The chosen dataset.

        Returns:
            The similar patterns in the searching dataset.
        """
        if dataset_name not in self._output_top_similarity_df:
            raise ValueError(f'`dataset_name`=\"{dataset_name}\" is not valid. ' +
                f'You need to run `ptr.find_similar_charts(dataset=\"{dataset_name}\")` first.'
            )
        chart = self.chart
        top_similarity_df = self._output_top_similarity_df[dataset_name]
        figures = [chart(output='go.Figure', title='Target')]
        for i, row in top_similarity_df.iterrows():
            sample_start = row['start']
            sample_end = row['end']
            figures.append(chart(start=sample_start, end=sample_end, output='go.Figure', title=f'({i})'))
        try:
            return _combine_plotly_figures(figures, rows=rows, cols=cols)
        except Exception as e:
            print(f"The number of subplots are {top_similarity_df.shape[0]+1}, " +
                f"but your input params `rows`={rows}, and `cols`={cols}. You need to adjust the parameters. " +
                f"\nThe error is {e}.")

    def plot_likelihood_charts(
            self, 
            dataset_name: str='default', 
            fwd_window: int=50, 
            indices_to_skip: List[int]=[]
    ) -> go.Figure:
        """Plot the likelihood analysis.

        Kwargs:
            dataset_name: The chosen dataset.
            fwd_window  : The forward window to forecast.

        Returns:
            The likelihood analysis chart with the searching dataset.
        """
        if dataset_name not in self._output_top_similarity_df:
            raise ValueError(f'`dataset_name`=\"{dataset_name}\" is not valid. ' +
                f'You need to run `ptr.find_similar_charts(dataset=\"{dataset_name}\")` first.'
            )
        top_similar_charts = self._output_top_similarity_df[dataset_name]
        mask = ~top_similar_charts.index.isin(indices_to_skip)
        top_similar_charts = top_similar_charts[mask].copy()
        
        ohlcv_df = self._input_ohlcv_df[dataset_name]

        # get target ohlcv_df
        target_start = self.chart.get_parameters()['start']
        target_end = self.chart.get_parameters()['end']
        target_ohlcv_df = self.chart.get_parameters()['ohlcv_df']
        target_ohlcv_df = target_ohlcv_df[
            (target_ohlcv_df.index>=target_start)
            & (target_ohlcv_df.index<=target_end)
        ].copy()
        window = target_ohlcv_df.shape[0]

        # calculate and normalise similar paths' returns
        sum_scores = top_similar_charts['similarity'].sum()
        fwd_daily_returns = {}
        norm_daily_prices = {}
        weights = {}
        for i, row in top_similar_charts.iterrows():
            sample_start = row['start']
            sample_end = row['end']
            score = row['similarity']
            ohlcv_sample = ohlcv_df.iloc[:ohlcv_df.index.get_loc(sample_end) + 1 + fwd_window]
            ohlcv_sample = ohlcv_sample[ohlcv_sample.index>=sample_start]
        
            # forward returns
            daily_returns = ohlcv_sample[ohlcv_sample.index>=sample_end]['close'].pct_change().fillna(0)
            fwd_daily_returns[(sample_start, sample_end)] = daily_returns.values
            
            # normalized prices
            end_close = ohlcv_sample[ohlcv_sample.index>=sample_end]['close'].values[0]
            norm_daily_prices[(sample_start, sample_end)] = (ohlcv_sample['close']/end_close).values
        
            weights[(sample_start, sample_end)] = score / sum_scores
        nav = pd.DataFrame(norm_daily_prices)

        # calculate and normalise target returns
        target_end_close = target_ohlcv_df[target_ohlcv_df.index<=target_end]['close'].values[-1]
        target_nav = target_ohlcv_df['close']/target_end_close

        # calculate weighted average, upper and lower bounds of the similar paths
        wavg_nav = 0
        for col in nav.columns:
            wavg_nav += nav[col] * weights[col]       
        nav_stats = pd.DataFrame({
            'max': nav.max(axis=1),
            'min': nav.min(axis=1),
            'wavg': wavg_nav
        })

        # rotate the curve 
        rnav_stats = get_rotated_stats(target_nav, nav_stats, fwd_window=50)

        # plot the combined chart
        chart = _combine_nav_plots(target_nav, nav, nav_stats, rnav_stats)
        return chart

    def update_similarity(self, dataset_name: str, similarity_df: pd.DataFrame):
        """Manually assign the similarity dataframe.

        Kwargs:
            dataset_name : The chosen dataset.
            similarity_df: The similarity dataframe with columns ['start', 'end', 
                           'start_index', 'end_index', 'similarity'].
        """
        self._output_similarity_df[dataset_name] = similarity_df.copy()