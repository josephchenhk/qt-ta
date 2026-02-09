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

def find_sources(folder: str, include_py: bool = False) -> List[str]:
    """Find all Cython source files (.pyx) and optionally .py files to compile."""
    sources = []
    for root, dirs, files in os.walk(folder):
        # Skip hidden directories and build artifacts
        dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
        if ".ipynb_checkpoints" in root:
            continue
        for file in files:
            # Compile .pyx files (Cython source)
            if file.endswith(".pyx"):
                sources.append(os.path.join(root, file))
            # Optionally compile .py files as Cython extensions for better protection
            # Note: __init__.py is excluded - it will be compiled to .pyc instead
            elif include_py and file.endswith(".py") and file != "__init__.py" and file != "setup.py":
                sources.append(os.path.join(root, file))
    return sources

extensions = [
    Extension(
        source.replace(os.sep, '.').replace('.pyx', '').replace('.py', ''), 
        [source]
    ) 
    for source in find_sources('qt_ta', include_py=True)
]
print(f"Found {len(extensions)} extensions to compile")

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
    # Dependencies are now defined in pyproject.toml
    # install_requires is handled by pyproject.toml
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
        "Topic :: Office/Business :: Financial",
    ]
)