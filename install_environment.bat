@echo off
REM Installation script for Intel NPU Benchmark Environment (Windows)
REM This script sets up the complete environment for running ONNX Runtime and OpenVINO benchmarks

echo === Intel NPU Benchmark Environment Setup ===
echo This script will install all required packages for benchmarking.
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+ first.
    pause
    exit /b 1
)

echo ✓ Python found

REM Create virtual environment
echo Creating virtual environment...
python -m venv .venv_intel_npu_benchmark
call .venv_intel_npu_benchmark\Scripts\activate.bat

echo ✓ Virtual environment created and activated

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with Intel XPU support (CRITICAL: Must be installed first)
echo Installing PyTorch with Intel XPU support...
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

echo ✓ PyTorch XPU installed

REM Install remaining packages from requirements
echo Installing remaining packages...
pip install -r requirements-remaining.txt
if errorlevel 1 (
    echo Warning: Some packages failed to install. Continuing...
    echo Attempting to install critical packages individually...
    pip install onnxruntime onnxruntime-openvino openvino psutil transformers
    if errorlevel 1 (
        echo Error: Critical packages failed to install.
        pause
        exit /b 1
    )
)

echo ✓ All packages installed successfully

REM Verify installation
echo.
echo === Verifying Installation ===

python -c "
import torch
import torchvision
import torchaudio
import onnxruntime as ort
import openvino as ov
import transformers
print('✓ Core packages imported successfully')

print(f'PyTorch version: {torch.__version__}')
print(f'TorchVision version: {torchvision.__version__}')
print(f'TorchAudio version: {torchaudio.__version__}')
print(f'ONNX Runtime version: {ort.__version__}')
print(f'Available ONNX Runtime providers: {ort.get_available_providers()}')

core = ov.Core()
print(f'OpenVINO available devices: {core.available_devices}')

if 'OpenVINOExecutionProvider' in ort.get_available_providers():
    print('✓ OpenVINO Execution Provider available')
else:
    print('⚠ OpenVINO Execution Provider not available')

if 'NPU' in core.available_devices:
    print('✓ NPU device detected')
else:
    print('⚠ NPU device not detected')
"

echo.
echo === Installation Complete ===
echo To activate the environment in the future, run:
echo .venv_intel_npu_benchmark\Scripts\activate.bat
echo.
echo To run benchmarks:
echo python "Final Comprehensive ORT Benchmark Script.py"
echo python "Final Comprehensive Benchmark Script.py"
pause
