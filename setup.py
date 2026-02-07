# -*- coding: utf-8 -*-
# @Project : qt-ta
# @Time    : 9/2/2024 4:20 PM
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: setup.py

"""
Copyright (C) 2024 Joseph Chen - All Rights Reserved
For any business inquiry, please write to: josephchenhk@gmail.com
"""

from typing import List
import os
from setuptools import setup
from setuptools import find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

def find_sources(folder: str) -> List[str]:
    sources = []
    for root, dirs, files in os.walk(folder):
        if ".ipynb_checkpoints" in root:
            continue
        for file in files:
            if file == '__init__.py':
                continue
            if file.endswith(".pyx") or file.endswith(".py"):
                sources.append(os.path.join(root, file))
    return sources

extensions = [
    Extension(source.replace('/', '.').replace('.py', ''), [f"{source}"]) 
    for source in find_sources('qt_ta')
]
print(extensions)

setup(
    name="qt-ta",
    version="1.0.0",
    description="Technical analysis for quantitative trading",
    author="Joseph Chen",
    author_email="josephchenhk@gmail.com",
    maintainer="Joseph Chen",
    url="",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
    # install_requires=[
    #     "numpy>=2.0"
    # ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ]
)