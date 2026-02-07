# -*- coding: utf-8 -*-
# @Project : qtrader-core
# @Time    : 2024/9/17
# @Author  : Joseph Chen
# @Email   : josephchenhk@gmail.com
# @FileName: run_packaging

import os
import sys
import glob
import shutil
import argparse
import tempfile
import subprocess
import zipfile
from pathlib import Path

def run_command(cmd, header=""):
    print(f'{header} {cmd}')
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(f"ERROR: {result.stderr}")
    return result

def remove_sources_from_wheel(wheel_path, extensions_to_remove=['.py', '.pyx']):
    """Remove source files from a wheel file."""
    print(f"Removing source files from wheel: {wheel_path}")
    
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Extract the wheel to the temporary directory
        with zipfile.ZipFile(wheel_path, 'r') as wheel_zip:
            wheel_zip.extractall(temp_dir)
        
        # Get list of files to remove
        files_to_remove = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if any(file.endswith(ext) for ext in extensions_to_remove) and not file.endswith('__init__.py'):
                    file_path = os.path.join(root, file)
                    files_to_remove.append(file_path)
                    print(f"  Will remove: {os.path.relpath(file_path, temp_dir)}")
        
        # Remove the files
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Create a new wheel
        os.remove(wheel_path)
        with zipfile.ZipFile(wheel_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_wheel:
            for root, _, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, temp_dir)
                    new_wheel.write(file_path, arc_name)
    
    print(f"Source files removed from wheel: {wheel_path}")
    return wheel_path

def main(
        my_module: str = "qt-ta",
        my_module_ver: str = "1.0.0",
        dist_dir: str = "jquant"
):
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Package the module and optionally remove source files from wheel.")
    parser.add_argument('--remove-source', action='store_true', help='Remove source files (*.py, *.pyx, *.pyd) from the wheel package')
    parser.add_argument('--module', type=str, default=my_module, help=f'Module name (default: {my_module})')
    parser.add_argument('--version', type=str, default=my_module_ver, help=f'Module version (default: {my_module_ver})')
    parser.add_argument('--dist-dir', type=str, default=dist_dir, help=f'Distribution directory (default: {dist_dir})')
    args = parser.parse_args()
    
    # Update parameters from command line arguments
    my_module = args.module
    my_module_ver = args.version
    dist_dir = args.dist_dir

    # Get Python version string
    py_ver = f'{sys.version_info[0]}{sys.version_info[1]}'
    
    # Windows specific commands
    if os.name == 'nt':  # Windows
        # os.chdir(f'{my_module}')
        run_command("python setup.py build --compiler=mingw32", "0.")
        os.chdir('..')

        # Create dist_dir if it doesn't exist
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)

        run_command(f"pip wheel --no-deps --no-cache-dir {my_module}/ -w dist {my_module}", "1.")

        # Find wheel file
        wheel_pattern = f"dist{os.sep}{my_module.replace('-', '_')}-{my_module_ver}-cp{py_ver}-cp{py_ver}-win_amd64.whl"
        wheel_files = glob.glob(wheel_pattern)
        
        if args.remove_source and wheel_files:
            for wheel_file in wheel_files:
                remove_sources_from_wheel(wheel_file)
        
        # Move the wheel file to dist_dir
        for wheel_file in wheel_files:
            target_path = os.path.join(dist_dir, os.path.basename(wheel_file))
            run_command(f"move {wheel_file} {target_path}", "2.")

        # Find and optionally remove C files
        c_files = []
        for root, _, files in os.walk(f'.{os.sep}{my_module}'):
            for file in files:
                if file.endswith('.c'):
                    c_files.append(os.path.join(root, file))
        
        if c_files:
            run_command(f'for /r .\\{my_module} %f in (*.c) do echo "%f"', "3.")
            run_command(f'for /r .\\{my_module} %f in (*.c) do del "%f"', "4.")

        # Clean up build artifacts
        run_command(f'rmdir /s /q dist', "5.")
        run_command(f'rmdir /s /q {my_module}{os.sep}build', "6.")
        run_command(f'rmdir /s /q {my_module}{os.sep}{my_module.replace("-", "_")}.egg-info', "7.")

    else:  # Linux and Mac OS
        # Build the wheel
        # os.chdir(f'{my_module}')
        run_command("python setup.py build", "0.")
        os.chdir('..')

        # Create dist_dir if it doesn't exist
        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)

        run_command(f"pip wheel --no-deps --no-cache-dir {my_module}/ -w dist {my_module}", "1.")

        # Find wheel files
        wheel_pattern = f"dist/{my_module.replace('-', '_')}-{my_module_ver}-cp{py_ver}-*.whl"
        wheel_files = glob.glob(wheel_pattern)
        
        # Remove source files from wheel if requested
        if args.remove_source and wheel_files:
            for wheel_file in wheel_files:
                remove_sources_from_wheel(wheel_file)
        
        # Move wheel files to dist_dir
        for wheel_file in wheel_files:
            target_path = os.path.join(dist_dir, os.path.basename(wheel_file))
            shutil.move(wheel_file, target_path)
            run_command(f"echo Moved {wheel_file} to {target_path}", "2.")

        # Find and remove C files
        c_files = []
        for root, _, files in os.walk(f'./{my_module}'):
            for file in files:
                if file.endswith('.c'):
                    c_file_path = os.path.join(root, file)
                    c_files.append(c_file_path)
                    print(f"3. Found C file: {c_file_path}")
                    if os.path.exists(c_file_path):
                        os.remove(c_file_path)
                        print(f"4. Removed C file: {c_file_path}")

        # Clean up build artifacts using platform-independent methods
        if os.path.exists('dist'):
            shutil.rmtree('dist')
            run_command("echo Removed dist directory", "5.")
        
        build_dir = os.path.join(my_module, 'build')
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir)
            run_command(f"echo Removed {build_dir}", "6.")
        
        egg_info_dir = os.path.join(my_module, f"{my_module.replace('-', '_')}.egg-info")
        if os.path.exists(egg_info_dir):
            shutil.rmtree(egg_info_dir)
            run_command(f"echo Removed {egg_info_dir}", "7.")

    print(f'Finished updating {my_module} {my_module_ver}')

if __name__ == "__main__":
    main()