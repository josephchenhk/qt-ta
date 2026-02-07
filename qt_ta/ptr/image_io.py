import os
import warnings
from typing import List
from pathlib import Path
import io
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
from IPython.display import Image as IPythonImage
try:
    from qt_common.s3.io import list_s3_objects, delete_s3_object, s3_path_exists, read_from_s3, write_to_s3
except:
    warnings.warn("qt_common is not available.")

BUCKET = os.environ['BQUANT_SANDBOX_ENTERPRISE_BUCKET']
PREFIX = 'bquant_common_s3'

def write_image(
        fig: go.Figure, 
        file_path: str, 
        file_scale: int=1, 
        file_width: int=800, 
        file_height: int=600, 
        file_format: str='png', 
        location: str='s3',
        bucket: str=BUCKET, 
        prefix: str=PREFIX
):
    assert location in ('s3', 'local'), 'parameter `location` can only be either `s3` or `local`.'
    if location == 'local':
        folder_path = '/'.join(file_path.split('/')[:-1])
        Path(folder_path).mkdir(parents=True, exist_ok=True)
        fig.write_image(file_path, scale=file_scale, width=file_width, height=file_height, format=file_format)
    elif location == 's3':
        fig_data = io.BytesIO()
        fig.write_image(fig_data, scale=file_scale, width=file_width, height=file_height, format=file_format)
        # rewind the BytesIO object
        fig_data.seek(0)
        write_to_s3(data=fig_data, file_path=file_path, data_format='bytesio', bucket=bucket, prefix=prefix)
    
def read_image(
        file_path: str, 
        location: str='s3',
        bucket: str=BUCKET, 
        prefix: str=PREFIX
) -> PngImageFile:
    assert location in ('s3', 'local'), 'parameter `location` can only be either `s3` or `local`.'
    if location == 'local':
        image_read = Image.open(file_path)
    elif location == 's3':
        fig_data_read = read_from_s3(file_path=file_path, data_format='bytesio', bucket=bucket, prefix=prefix)
        image_read = Image.open(fig_data_read)
    return image_read

def list_images(folder_path: str, location: str='s3', bucket: str=BUCKET, prefix: str=PREFIX) -> List[str]:
    assert location in ('s3', 'local'), 'parameter `location` can only be either `s3` or `local`.'
    if location == 'local':
        images = os.listdir(folder_path)
    elif location == 's3':
        images_full_paths = list_s3_objects(folder_path=folder_path, bucket=bucket, prefix=prefix)
        images = [p.split('/')[-1] for p in images_full_paths]
    return images
                  

def plot_subplots(image_paths: List[str]) -> go.Figure:
    N = len(image_paths)
    imgs = []
    img_arrs = []
    for i in range(N):
        # print(i, f'ptr/1/{sorted_result[i+1][0]}')
        img = read_image(file_path=image_paths[i], location='s3')
        imgs.append(img)
        img_arrs.append(np.array(img))
    
    img_names = [n.split('/')[-1].replace('.png', '') for n in image_paths]

    columns = 2
    rows = N // columns + N % columns
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=img_names)

    for i, img_arr in enumerate(img_arrs):
        row = i // columns + 1
        col = i % columns + 1
        fig.add_trace(go.Image(z=img_arr), row=row, col=col)

    fig.update_layout(
        width=1200,
        height=500 * rows,
        margin=dict(l=10, r=10, b=10, t=40),
        grid_pattern="independent",
    )
    return fig

def display_png(img: PngImageFile) -> IPythonImage:
    buffered = io.BytesIO()
    img.save(buffered, format='PNG')
    png_img = buffered.getvalue()
    return IPythonImage(data=png_img)
    