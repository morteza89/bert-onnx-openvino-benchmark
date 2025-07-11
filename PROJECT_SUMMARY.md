# Project Summary: Native OpenVINO vs ONNX with OpenVINO Execution Provider

## Project Overview

Comprehensive performance comparison of BERT token classification model inference comparing:
1. **Native OpenVINO models** (using OpenVINO API directly)
2. **ONNX models with OpenVINOExecutionProvider** through different libraries (onnxruntime & optimum.onnxruntime)

All tests conducted on Intel NPU hardware to evaluate performance differences between native OpenVINO execution and ONNX models utilizing OpenVINO's execution provider.

## Files Included (15+ total)

### 🚀 Benchmark Scripts (3 files)

1. **Comprehensive ORT Benchmark Script.py** - Benchmarks ONNX models using direct `onnxruntime` with `OpenVINOExecutionProvider`
2. **Comprehensive optimum Benchmark Script.py** - Benchmarks ONNX models using `optimum.onnxruntime` with `OpenVINOExecutionProvider`
3. **Step-by-step ONNX Export .py** - Exports PyTorch models to ONNX format and converts to native OpenVINO format

### Installation & Setup (7 files)

1. **requirements/** - Directory containing all Python package requirements
   - `requirements.txt` - Complete environment (60+ packages)
   - `requirements-minimal.txt` - Essential packages only
   - `requirements-remaining.txt` - Install after PyTorch XPU
   - `requirements-essential.txt` - Core packages
2. **install_environment.bat** - Windows auto-installer
3. **install_environment.sh** - Linux/Mac auto-installer
4. **INSTALLATION_GUIDE.md** - Complete setup guide

### Results & Documentation (5+ files)

1. **README.md** - Project overview and quick start
2. **BENCHMARK_COMPARISON_RESULTS.md** - Detailed performance analysis
3. **PROJECT_SUMMARY.md** - This file
4. **benchmark_results/** - Directory containing all benchmark result JSON files
   - `ort_benchmark_results_20250711_133908.json` - ONNX (onnxruntime) small model results
   - `ort_benchmark_results_20250711_134756.json` - ONNX (onnxruntime) large model results
   - `benchmark_results_20250711_135852.json` - ONNX (optimum) small model results
   - `benchmark_results_20250711_140243.json` - ONNX (optimum) large model results
5. **LICENSE** - MIT License for open-source usage

## Key Performance Results

### Performance Comparison: Native OpenVINO vs ONNX with OpenVINO Execution Provider

**Key Finding**: Native OpenVINO models consistently outperform ONNX models by 5-16%, even when ONNX models use OpenVINOExecutionProvider.

#### Small Model (test_smaller_model)

| Method                              | Latency | Throughput     | Rank |
| ----------------------------------- | ------- | -------------- | ---- |
| Native OpenVINO (via optimum)      | 2.98ms  | 341.9 inf/sec  | 🥇   |
| Native OpenVINO (via direct ORT)   | 3.07ms  | 314.2 inf/sec  | 🥈   |
| ONNX + OpenVINOEP (onnxruntime)     | 3.42ms  | 295.5 inf/sec  | 🥉   |
| ONNX + OpenVINOEP (optimum.onnxruntime) | 3.46ms  | 300.0 inf/sec  | 4th  |

#### Large Model (test_bigger_model)

| Method                              | Latency | Throughput     | Rank |
| ----------------------------------- | ------- | -------------- | ---- |
| Native OpenVINO (via direct ORT)   | 7.79ms  | 128.6 inf/sec  | 🥇   |
| Native OpenVINO (via optimum)      | 7.82ms  | 128.0 inf/sec  | 🥈   |
| ONNX + OpenVINOEP (onnxruntime)     | 8.18ms  | 122.7 inf/sec  | 🥉   |
| ONNX + OpenVINOEP (optimum.onnxruntime) | 8.34ms  | 120.2 inf/sec  | 4th  |

## Quick Start Commands

```bash
# 1. Install environment (Windows)
install_environment.bat

# 2. Run ONNX benchmarks
python "Comprehensive ORT Benchmark Script.py"
python "Comprehensive optimum Benchmark Script.py"

# 3. Export models (PyTorch → ONNX → Native OpenVINO)
python "Step-by-step ONNX Export .py"

# 4. View results
# Check benchmark_results/ directory for JSON files
```

## Hardware Tested

- **Device**: Intel NPU
- **Model**: BERT Token Classification
- **Input**: 512 tokens, batch size 1
- **Test Duration**: 2000 iterations + 60s throughput + text variety testing
- **Models**: test_smaller_model & test_bigger_model

## Production Recommendation

✅ **Use Native OpenVINO models** for production deployments

- **5-16% faster** than ONNX models with OpenVINOExecutionProvider
- **Better consistency** and reliability across runs
- **Optimal hardware utilization** on Intel NPU
- **Nearly identical results** across multiple benchmark runs

### Migration Path:
1. **Development**: Start with ONNX + OpenVINOExecutionProvider for easy prototyping
2. **Production**: Convert to native OpenVINO models for optimal performance
3. **Result**: Nearly identical behavior with 5-16% performance improvement

---

_Updated: July 11, 2025_
_Environment: Intel NPU with PyTorch XPU support_
_License: MIT License - Open Source_
