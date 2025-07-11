# Project Summary: BERT ONNX vs OpenVINO Benchmarking

## Project Overview

Comprehensive performance comparison of BERT token classification model inference using ONNX Runtime and OpenVINO on Intel NPU hardware.

## Files Included (14 total)

### 🚀 Benchmark Scripts (4 files)

1. **Final Comprehensive ORT Benchmark Script.py**
2. **Final Comprehensive optimum Benchmark Script.py** - Alternative using ORTModel
3. **Latency Benchmark Script.py** - Original simple benchmark
4. **Latency Benchmark Script - ORTModel.py** - Simple ORTModel version

### Installation & Setup (6 files)

1. **requirements.txt** - Complete environment (60+ packages)
2. **requirements-minimal.txt** - Essential packages only (15 packages)
3. **requirements-remaining.txt** - Install after PyTorch XPU
4. **install_environment.bat** - Windows auto-installer
5. **install_environment.sh** - Linux/Mac auto-installer
6. **INSTALLATION_GUIDE.md** - Complete setup guide

### Results & Documentation (4 files)

1. **README.md** - Project overview and quick start
2. **BENCHMARK_COMPARISON_RESULTS.md** - Detailed performance analysis
3. **benchmark_results_20250710_124343.json** - ORTModel results
4. **ort_benchmark_results_20250710_133153.json** - Direct ONNX Runtime results

## Key Performance Results

### Comparison Summary

| Method              | Latency | Throughput     | Winner |
| ------------------- | ------- | -------------- | ------ |
| Direct ONNX Runtime | 7.347ms | 132.60 inf/sec | 🥇     |
| OpenVINO IR         | 7.420ms | 131.84 inf/sec | 🥈     |
| ORTModel            | 9.828ms | 123.28 inf/sec | 🥉     |

## Quick Start Commands

```bash
# 1. Install environment (Windows)
install_environment.bat

# 2. Run best benchmark
python "Final Comprehensive ORT Benchmark Script.py"

# 3. View results
# Check generated JSON files and console output
```

## Hardware Tested

- **Device**: Intel NPU
- **Model**: BERT Token Classification
- **Input**: 512 tokens, batch size 1
- **Test Duration**: 2000 iterations + 60s throughput

## Production Recommendation

✅ **Use Direct ONNX Runtime** for production deployments

- 25% faster than ORTModel approach
- Better consistency and reliability
- Lower overhead and resource usage

---

_Generated: July 10, 2025_
_Environment: Intel NPU with PyTorch XPU support_
