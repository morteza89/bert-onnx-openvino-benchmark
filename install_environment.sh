#!/bin/bash
# Installation script for Intel NPU Benchmark Environment
# This script sets up the complete environment for running ONNX Runtime and OpenVINO benchmarks

set -e  # Exit on any error

echo "=== Intel NPU Benchmark Environment Setup ==="
echo "This script will install all required packages for benchmarking."
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
if [[ $(echo "$python_version >= 3.8" | bc -l) -eq 0 ]]; then
    echo "Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✓ Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv_intel_npu_benchmark
source .venv_intel_npu_benchmark/bin/activate

echo "✓ Virtual environment created and activated"

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with Intel XPU support (CRITICAL: Must be installed first)
echo "Installing PyTorch with Intel XPU support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

echo "✓ PyTorch XPU installed"

# Install remaining packages from requirements
echo "Installing remaining packages..."
pip install -r requirements-remaining.txt

echo "✓ All packages installed successfully"

# Verify installation
echo ""
echo "=== Verifying Installation ==="

python3 -c "
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

echo ""
echo "=== Installation Complete ==="
echo "To activate the environment in the future, run:"
echo "source .venv_intel_npu_benchmark/bin/activate"
echo ""
echo "To run benchmarks:"
echo "python 'Final Comprehensive ORT Benchmark Script.py'"
echo "python 'Final Comprehensive Benchmark Script.py'"
