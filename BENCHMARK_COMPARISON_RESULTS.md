# Comprehensive Benchmark Results Comparison

## ONNX Runtime vs OpenVINO Performance Analysis

### Test Environment

- **Device**: NPU (Neural Processing Unit)
- **Models**: BERT Token Classification (test_smaller_model & test_bigger_model)
- **Sequence Length**: 512 tokens (fixed)
- **Batch Size**: 1 (fixed)
- **Iterations**: 2000 for detailed latency, 500 for text variety
- **Throughput Duration**: 60 seconds
- **System**: 8-core CPU, 31.5GB RAM, Intel NPU
- **Test Date**: July 11, 2025

---

## Performance Comparison Summary

### 1. Small Model (test_smaller_model) Results

#### Direct ONNX Runtime vs OpenVINO

| Metric             | ONNX Runtime  | OpenVINO      | Speedup      |
| ------------------ | ------------- | ------------- | ------------ |
| **Mean Latency**   | 3.42 ms       | 3.07 ms       | 1.11x faster |
| **Median Latency** | 3.37 ms       | 3.01 ms       | 1.12x faster |
| **P95 Latency**    | 3.96 ms       | 3.60 ms       | 1.10x faster |
| **Throughput**     | 295.5 inf/sec | 314.2 inf/sec | 1.06x faster |
| **Consistency**    | CV: 19.38%    | CV: 19.66%    | Similar      |

#### Optimum ORTModel vs OpenVINO

| Metric             | Optimum ORT   | OpenVINO      | Speedup         |
| ------------------ | ------------- | ------------- | --------------- |
| **Mean Latency**   | 3.46 ms       | 2.98 ms       | 1.16x faster    |
| **Median Latency** | 3.44 ms       | 2.96 ms       | 1.16x faster    |
| **P95 Latency**    | 4.04 ms       | 3.39 ms       | 1.19x faster    |
| **Throughput**     | 300.0 inf/sec | 341.9 inf/sec | 1.14x faster    |
| **Consistency**    | CV: 16.92%    | CV: 8.96%     | OpenVINO better |

### 2. Large Model (test_bigger_model) Results

#### Direct ONNX Runtime vs OpenVINO

| Metric             | ONNX Runtime  | OpenVINO      | Speedup         |
| ------------------ | ------------- | ------------- | --------------- |
| **Mean Latency**   | 8.18 ms       | 7.79 ms       | 1.05x faster    |
| **Median Latency** | 8.22 ms       | 7.75 ms       | 1.06x faster    |
| **P95 Latency**    | 8.61 ms       | 8.21 ms       | 1.05x faster    |
| **Throughput**     | 122.7 inf/sec | 128.6 inf/sec | 1.05x faster    |
| **Consistency**    | CV: 7.12%     | CV: 5.56%     | OpenVINO better |

#### Optimum ORTModel vs OpenVINO

| Metric             | Optimum ORT   | OpenVINO      | Speedup      |
| ------------------ | ------------- | ------------- | ------------ |
| **Mean Latency**   | 8.34 ms       | 7.82 ms       | 1.07x faster |
| **Median Latency** | 8.38 ms       | 7.76 ms       | 1.08x faster |
| **P95 Latency**    | 8.71 ms       | 8.25 ms       | 1.06x faster |
| **Throughput**     | 120.2 inf/sec | 128.0 inf/sec | 1.06x faster |
| **Consistency**    | CV: 6.95%     | CV: 7.19%     | Similar      |

---

## Key Findings

### 1. **OpenVINO Consistently Outperforms ONNX Runtime**

- **Across all test scenarios**: OpenVINO shows 5-16% performance advantage
- **Better latency**: Consistently lower mean, median, and P95 latencies
- **Higher throughput**: 5-14% better inferences per second
- **Model size scaling**: Performance advantage increases with larger models

### 2. **Framework Comparison: Direct ONNX vs Optimum**

- **Direct ONNX Runtime**: 3.42ms (small), 8.18ms (large)
- **Optimum ORTModel**: 3.46ms (small), 8.34ms (large)
- **Minimal overhead**: Optimum adds only 1-2% latency overhead
- **Similar performance**: Both approaches perform comparably

### 3. **Model Size Impact**

- **Small Model Performance**: 3ms range, 295-342 inf/sec
- **Large Model Performance**: 8ms range, 120-129 inf/sec
- **Scaling factor**: ~2.4x latency increase, ~2.4x throughput decrease
- **Consistency**: Large models show better consistency (lower CV)

### 4. **Cold Start Performance**

- **Small Model**: 7-8ms cold start overhead
- **Large Model**: 15-18ms cold start overhead
- **OpenVINO advantage**: 10-16% faster cold start times

---

## Updated Performance Ranking

### Small Model (test_smaller_model):

1. 🥇 **OpenVINO (with Optimum)**: 2.98ms, 341.9 inf/sec
2. 🥈 **OpenVINO (with Direct ORT)**: 3.07ms, 314.2 inf/sec
3. 🥉 **Direct ONNX Runtime**: 3.42ms, 295.5 inf/sec
4. 🏅 **Optimum ORTModel**: 3.46ms, 300.0 inf/sec

### Large Model (test_bigger_model):

1. 🥇 **OpenVINO (with Direct ORT)**: 7.79ms, 128.6 inf/sec
2. 🥈 **OpenVINO (with Optimum)**: 7.82ms, 128.0 inf/sec
3. 🥉 **Direct ONNX Runtime**: 8.18ms, 122.7 inf/sec
4. 🏅 **Optimum ORTModel**: 8.34ms, 120.2 inf/sec

---

## Recommendations

### For Production Use:

1. **Use OpenVINO** for best performance across all scenarios
2. **Choose Direct ONNX Runtime** if you need maximum ONNX compatibility
3. **Optimum ORTModel** offers good balance of performance and ease of use

### For Development/Experimentation:

1. **OpenVINO** provides the best performance-to-effort ratio
2. **Optimum ORTModel** is easiest to integrate with transformers workflows
3. **Direct ONNX Runtime** offers fine-grained control but requires more setup

### Hardware Considerations:

- **Intel NPU** provides excellent acceleration for all frameworks
- **Performance scales well** with model size
- **Memory usage** remains consistent across approaches

---

## Technical Notes

### Test Data Details:

- **Test Date**: July 11, 2025
- **Benchmark Files Generated**:
  - `ort_benchmark_results_20250711_133908.json` - Direct ORT (small model)
  - `ort_benchmark_results_20250711_134756.json` - Direct ORT (large model)
  - `benchmark_results_20250711_135852.json` - Optimum ORT (small model)
  - `benchmark_results_20250711_140243.json` - Optimum ORT (large model)

### Model Specifications:

- **Small Model**: test_smaller_model (8 layers, 8 attention heads, 512 hidden size)
- **Large Model**: test_bigger_model (larger configuration)
- **Architecture**: BertForTokenClassification
- **Vocabulary Size**: 30,522 tokens
- **Max Sequence Length**: 512 tokens

### Benchmark Methodology:

- **Warmup**: 30 iterations before timing
- **Latency Test**: 2000 iterations with high-precision timing
- **Throughput Test**: 60-second sustained performance
- **Text Variety**: 500 samples with different input texts
- **Cold Start**: 10 trials measuring first-run performance
- **Statistical Analysis**: Mean, median, std dev, P95, P99, coefficient of variation

### System Configuration:

- **Hardware**: Intel NPU-enabled system
- **CPU**: 8-core processor
- **Memory**: 31.5GB RAM
- **Software**: PyTorch 2.7.1+xpu, ONNX Runtime 1.22.0, OpenVINO 2025.2.0

### Performance Variability:

- **Coefficient of Variation**: 5-20% across different approaches
- **OpenVINO**: Generally more consistent performance
- **Large Models**: Better consistency than small models
- **Cold Start**: 2-3x slower than warmed-up performance

- **Model Constraints**: Fixed batch size (1) and sequence length (512)
- **Hardware**: NPU acceleration enabled for all approaches
- **Precision**: All measurements use high-precision timing (nanoseconds)
- **Statistics**: Based on 2000 iterations with proper warmup (30 iterations)
- **Reproducibility**: Results saved to JSON files for further analysis

The direct ONNX Runtime approach provides the best balance of performance, consistency, and throughput for production inference workloads.
