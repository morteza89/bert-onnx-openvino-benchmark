# Benchmarking BERT ONNX vs OpenVINO Model on Intel CPU GPU NPU

This folder contains comprehensive benchmarking tools and results for comparing ONNX Runtime and OpenVINO inference performance on Intel hardware (CPU, GPU, NPU).

## 📁 File Structure

### Benchmark Scripts
- **`Comprehensive ORT Benchmark Script.py`** - Comprehensive benchmark using direct ONNX Runtime (`ort.InferenceSession`)
- **`Comprehensive optimum Benchmark Script.py`** - Comprehensive benchmark using Optimum `ORTModelForTokenClassification`
- **`Step-by-step ONNX Export .py`** - Step-by-step guide for exporting ONNX models from PyTorch

### Installation Files
- **`requirements.txt`** - Complete environment with all packages and exact versions
- **`requirements-minimal.txt`** - Essential packages only
- **`requirements-remaining.txt`** - Packages to install after PyTorch XPU
- **`install_environment.bat`** - Automated installation script for Windows
- **`install_environment.sh`** - Automated installation script for Linux/Mac
- **`INSTALLATION_GUIDE.md`** - Comprehensive installation and setup guide

### Results and Documentation
- **`BENCHMARK_COMPARISON_RESULTS.md`** - Detailed analysis and comparison of all benchmark results
- **`benchmark_results_20250710_124343.json`** - Results from ORTModel benchmark
- **`ort_benchmark_results_20250710_133153.json`** - Results from direct ONNX Runtime benchmark
- **`README.md`** - This file
- **`PROJECT_SUMMARY.md`** - High-level project overview and summary

## 🚀 Quick Start

### 1. Setup Directory Structure
The scripts expect the following directory structure:
```
your-project/
├── benchmarking BERT ONNX vs OV model on Intel CPU GPU NPU/
│   ├── Comprehensive ORT Benchmark Script.py
│   ├── Comprehensive optimum Benchmark Script.py
│   ├── Step-by-step ONNX Export .py
│   └── ... (other files)
└── proxy_models_for_intel/
    ├── test_bigger_model/
    │   ├── config.json
    │   ├── model.safetensors
    │   ├── model_openvino_1.xml
    │   ├── model_openvino_1.bin
    │   └── ...
    └── test_smaller_model/
        └── ... (similar structure)
```

### 2. Install Environment
```bash
# Windows
install_environment.bat

# Linux/Mac
chmod +x install_environment.sh
./install_environment.sh
```

### 3. Run Benchmarks
```bash
# Best performing approach (Direct ONNX Runtime)
python "Comprehensive ORT Benchmark Script.py"

# Alternative approach (ORTModel)
python "Comprehensive optimum Benchmark Script.py"

# Export ONNX model from PyTorch
python "Step-by-step ONNX Export .py"
```

## 🏆 Key Findings

### Performance Ranking (NPU):
1. 🥇 **Direct ONNX Runtime**: 7.347ms mean latency, 132.60 inf/sec
2. 🥈 **OpenVINO**: 7.520ms mean latency, 131.64 inf/sec  
3. 🥉 **ORTModelForTokenClassification**: 9.828ms mean latency, 123.28 inf/sec

### Recommendations:
- **Production**: Use Direct ONNX Runtime for best performance
- **Development**: ORTModel is easier but has 25% overhead
- **Hardware**: Intel NPU provides excellent acceleration for BERT inference

## 📊 Benchmark Features

- **High-precision timing** (nanosecond accuracy)
- **Statistical analysis** (mean, median, P95, P99, std dev)
- **Throughput testing** (sustained performance over 60 seconds)
- **Text variety testing** (realistic workload simulation)
- **Cold start analysis** (first inference performance)
- **Comprehensive logging** (detailed statistics and system info)
- **JSON export** (results saved for analysis)

## 🔄 ONNX Export Features

The `Step-by-step ONNX Export .py` script provides:
- **Step-by-step model conversion** from PyTorch to ONNX
- **Validation testing** to ensure model accuracy
- **Export configuration** with proper input/output specifications
- **Troubleshooting guidance** for common export issues

## 🔧 Requirements

- **Hardware**: Intel processor with NPU support (recommended)
- **Software**: Python 3.8+, Intel NPU drivers
- **Memory**: 8GB+ RAM (16GB+ recommended)
- **Storage**: 2GB+ free space

## ⚙️ Configuration

### Model Path Configuration
The scripts use relative paths to locate model files. By default, they look for:
```
../proxy_models_for_intel/test_bigger_model/
```

To use a different model or path, modify the `BASE_DIR` variable in each script:
```python
# For test_smaller_model
BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proxy_models_for_intel", "test_smaller_model")

# For custom path
BASE_DIR = "path/to/your/model/directory"
```

### Device Configuration
Change the target device by modifying the `device` variable:
```python
device = "NPU"  # Options: "CPU", "GPU", "NPU"
```

## 📖 Model Information

- **Model Type**: BERT for Token Classification
- **Framework**: PyTorch → ONNX → OpenVINO
- **Input**: 512 token sequences
- **Batch Size**: 1 (fixed)
- **Precision**: FP32

## 🎯 Use Cases

This benchmark suite is ideal for:
- **Model deployment decisions** (ONNX Runtime vs OpenVINO)
- **Hardware evaluation** (CPU vs GPU vs NPU performance)
- **Optimization validation** (measuring inference improvements)
- **Production planning** (throughput and latency requirements)
- **Research and development** (performance baselines)
- **Model conversion** (PyTorch to ONNX export with step-by-step guidance)

## 🔍 Troubleshooting

See `INSTALLATION_GUIDE.md` for detailed troubleshooting steps and common issues.

## 📝 Notes

- All benchmarks are designed for Intel hardware with NPU support
- Results may vary on different hardware configurations
- PyTorch XPU installation requires special index URL
- Intel-specific libraries are required for optimal NPU performance
- **Scripts use relative paths** for GitHub-friendly portability
- Model files should be placed in the `proxy_models_for_intel` directory relative to the script location
