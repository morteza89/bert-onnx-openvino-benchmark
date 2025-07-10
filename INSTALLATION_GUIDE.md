# Installation Guide for Benchmark Scripts

## Overview
This guide helps you set up the environment needed to run the comprehensive benchmark scripts for ONNX Runtime and OpenVINO on NPU.

## Requirements Files

### 1. `requirements.txt` (Complete Environment)
Contains all packages with exact versions from the working environment, including Intel-specific libraries for NPU support.

### 2. `requirements-minimal.txt` (Essential Only)
Contains only the essential packages needed to run the benchmark scripts.

## Installation Options

### Option 1: Complete Installation (Recommended for NPU)
```bash
# Create virtual environment
python -m venv .venv_benchmark
source .venv_benchmark/bin/activate  # Linux/Mac
# or
.venv_benchmark\Scripts\activate     # Windows

# IMPORTANT: Install PyTorch with XPU support first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install remaining packages
pip install -r requirements-remaining.txt
```

### Option 2: Automated Installation (Easiest)
```bash
# Linux/Mac
chmod +x install_environment.sh
./install_environment.sh

# Windows
install_environment.bat
```

### Option 3: Manual Installation with requirements.txt
```bash
# Create virtual environment
python -m venv .venv_benchmark
source .venv_benchmark/bin/activate  # Linux/Mac
# or
.venv_benchmark\Scripts\activate     # Windows

# Note: This may have PyTorch version conflicts, use Option 1 if issues occur
pip install -r requirements.txt
```

### Option 4: Minimal Installation (CPU/GPU)
```bash
# Create virtual environment
python -m venv .venv_benchmark_minimal
source .venv_benchmark_minimal/bin/activate  # Linux/Mac
# or
.venv_benchmark_minimal\Scripts\activate     # Windows

# Install PyTorch with XPU support first
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/xpu

# Install essential packages only
pip install -r requirements-minimal.txt
```

## Package Descriptions

### Core ML Frameworks
- **torch**: PyTorch with XPU support for Intel NPU
- **transformers**: Hugging Face transformers library
- **onnxruntime**: ONNX Runtime for model inference
- **onnxruntime-openvino**: OpenVINO execution provider for ONNX Runtime
- **openvino**: Intel OpenVINO toolkit
- **optimum**: Hugging Face optimization tools

### Intel NPU Support
- **mkl**: Intel Math Kernel Library
- **intel-openmp**: Intel OpenMP runtime
- **intel-sycl-rt**: Intel SYCL runtime for heterogeneous computing
- **onemkl-sycl-***: Intel oneMKL SYCL libraries for linear algebra operations

### Utilities
- **psutil**: System and process utilities for monitoring
- **numpy**: Numerical computing library
- **tqdm**: Progress bars for long-running operations

## Verification

After installation, verify the setup:

```python
# Test ONNX Runtime with OpenVINO EP
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())

# Test OpenVINO
import openvino as ov
core = ov.Core()
print("Available devices:", core.available_devices)

# Test transformers
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
print("Transformers working!")

# Test Intel extensions (if using NPU)
try:
    import intel_extension_for_pytorch as ipex
    print("Intel Extension for PyTorch available")
except ImportError:
    print("Intel Extension for PyTorch not available (optional)")
```

## Running the Benchmarks

After successful installation:

1. **Direct ONNX Runtime Benchmark:**
   ```bash
   python "Final Comprehensive ORT Benchmark Script.py"
   ```

2. **ORTModel Benchmark:**
   ```bash
   python "Final Comprehensive Benchmark Script.py"
   ```

3. **Original Simple Benchmark:**
   ```bash
   python "Latency Benchmark Script.py"
   ```

## Troubleshooting

### Common Issues

1. **NPU not detected:**
   - Ensure Intel NPU drivers are installed
   - Check with: `python -c "import openvino as ov; print(ov.Core().available_devices)"`

2. **OpenVINO EP not available:**
   - Install: `pip install onnxruntime-openvino`
   - Check: `python -c "import onnxruntime as ort; print('OpenVINOExecutionProvider' in ort.get_available_providers())"`

3. **Memory issues:**
   - Reduce iterations in benchmark scripts
   - Use minimal requirements if full installation fails

4. **Version conflicts:**
   - Use virtual environment to isolate packages
   - Install packages one by one if conflicts occur

## Hardware Requirements

- **CPU**: Intel processor (recommended for NPU support)
- **Memory**: Minimum 8GB RAM (16GB+ recommended for large models)
- **NPU**: Intel NPU (for NPU benchmarking)
- **Storage**: 2GB+ free space for models and dependencies

## Notes

- The Intel-specific packages are required for NPU acceleration
- PyTorch XPU support enables GPU/NPU acceleration on Intel hardware
- OpenVINO execution provider in ONNX Runtime provides additional optimization
- Some packages may require specific Intel drivers or runtime libraries
