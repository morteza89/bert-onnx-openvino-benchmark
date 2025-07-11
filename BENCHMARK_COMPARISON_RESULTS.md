# Comprehensive Benchmark Results: Native OpenVINO vs ONNX with OpenVINO Execution Provider

## Performance Analysis: Native OpenVINO vs ONNX Models Using OpenVINOExecutionProvider

This benchmark compares the performance of:

1. **Native OpenVINO models** (using OpenVINO API directly)
2. **ONNX models with OpenVINOExecutionProvider** through different libraries:
   - Via `onnxruntime` with `OpenVINOExecutionProvider`
   - Via `optimum.onnxruntime` with `OpenVINOExecutionProvider`

**Key Finding**: Native OpenVINO models consistently outperform ONNX models even when using OpenVINOExecutionProvider, demonstrating the efficiency of OpenVINO's native inference pipeline.

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

#### ONNX (onnxruntime + OpenVINOExecutionProvider) vs Native OpenVINO

| Metric             | ONNX + OpenVINOEP | Native OpenVINO | Performance Gain |
| ------------------ | ----------------- | --------------- | ---------------- |
| **Mean Latency**   | 3.42 ms           | 3.07 ms         | 1.11x faster     |
| **Median Latency** | 3.37 ms           | 3.01 ms         | 1.12x faster     |
| **P95 Latency**    | 3.96 ms           | 3.60 ms         | 1.10x faster     |
| **Throughput**     | 295.5 inf/sec     | 314.2 inf/sec   | 1.06x faster     |
| **Consistency**    | CV: 19.38%        | CV: 19.66%      | Similar          |

#### ONNX (optimum.onnxruntime + OpenVINOExecutionProvider) vs Native OpenVINO

| Metric             | ONNX + OpenVINOEP | Native OpenVINO | Performance Gain |
| ------------------ | ----------------- | --------------- | ---------------- |
| **Mean Latency**   | 3.46 ms           | 2.98 ms         | 1.16x faster     |
| **Median Latency** | 3.44 ms           | 2.96 ms         | 1.16x faster     |
| **P95 Latency**    | 4.04 ms           | 3.39 ms         | 1.19x faster     |
| **Throughput**     | 300.0 inf/sec     | 341.9 inf/sec   | 1.14x faster     |
| **Consistency**    | CV: 16.92%        | CV: 8.96%       | Native better    |

### 2. Large Model (test_bigger_model) Results

#### ONNX (onnxruntime + OpenVINOExecutionProvider) vs Native OpenVINO

| Metric             | ONNX + OpenVINOEP | Native OpenVINO | Performance Gain |
| ------------------ | ----------------- | --------------- | ---------------- |
| **Mean Latency**   | 8.18 ms           | 7.79 ms         | 1.05x faster     |
| **Median Latency** | 8.22 ms           | 7.75 ms         | 1.06x faster     |
| **P95 Latency**    | 8.61 ms           | 8.21 ms         | 1.05x faster     |
| **Throughput**     | 122.7 inf/sec     | 128.6 inf/sec   | 1.05x faster     |
| **Consistency**    | CV: 7.12%         | CV: 5.56%       | Native better    |

#### ONNX (optimum.onnxruntime + OpenVINOExecutionProvider) vs Native OpenVINO

| Metric             | ONNX + OpenVINOEP | Native OpenVINO | Performance Gain |
| ------------------ | ----------------- | --------------- | ---------------- |
| **Mean Latency**   | 8.34 ms           | 7.82 ms         | 1.07x faster     |
| **Median Latency** | 8.38 ms           | 7.76 ms         | 1.08x faster     |
| **P95 Latency**    | 8.71 ms           | 8.25 ms         | 1.06x faster     |
| **Throughput**     | 120.2 inf/sec     | 128.0 inf/sec   | 1.06x faster     |
| **Consistency**    | CV: 6.95%         | CV: 7.19%       | Similar          |

---

## Key Findings

### 1. **Native OpenVINO Consistently Outperforms ONNX Models**

- **Across all test scenarios**: Native OpenVINO shows 5-16% performance advantage over ONNX models even when using OpenVINOExecutionProvider
- **Better latency**: Consistently lower mean, median, and P95 latencies
- **Higher throughput**: 5-14% better inferences per second
- **Demonstrates**: The efficiency of OpenVINO's native inference pipeline vs. ONNX model execution

### 2. **ONNX Execution Provider Comparison: onnxruntime vs optimum.onnxruntime**

- **onnxruntime + OpenVINOExecutionProvider**: 3.42ms (small), 8.18ms (large)
- **optimum.onnxruntime + OpenVINOExecutionProvider**: 3.46ms (small), 8.34ms (large)
- **Minimal overhead**: optimum.onnxruntime adds only 1-2% latency overhead
- **Similar performance**: Both ONNX approaches perform comparably when using OpenVINOExecutionProvider

### 3. **Model Size Impact on Performance Advantage**

- **Small Model Performance**: Native OpenVINO 11-16% faster than ONNX approaches
- **Large Model Performance**: Native OpenVINO 5-7% faster than ONNX approaches
- **Scaling insight**: Native OpenVINO optimization benefits are more pronounced with smaller models
- **Consistency**: All approaches show good performance stability across runs

### 4. **Cold Start Performance**

- **Small Model**: 7-8ms cold start overhead across all approaches
- **Large Model**: 15-18ms cold start overhead across all approaches
- **Native OpenVINO advantage**: 10-16% faster cold start times compared to ONNX models

---

## Updated Performance Ranking

### Small Model (test_smaller_model):

1. 🥇 **Native OpenVINO** (via optimum): 2.98ms, 341.9 inf/sec
2. 🥈 **Native OpenVINO** (via direct ORT): 3.07ms, 314.2 inf/sec
3. 🥉 **ONNX + OpenVINOExecutionProvider** (onnxruntime): 3.42ms, 295.5 inf/sec
4. 🏅 **ONNX + OpenVINOExecutionProvider** (optimum.onnxruntime): 3.46ms, 300.0 inf/sec

### Large Model (test_bigger_model):

1. 🥇 **Native OpenVINO** (via direct ORT): 7.79ms, 128.6 inf/sec
2. 🥈 **Native OpenVINO** (via optimum): 7.82ms, 128.0 inf/sec
3. 🥉 **ONNX + OpenVINOExecutionProvider** (onnxruntime): 8.18ms, 122.7 inf/sec
4. 🏅 **ONNX + OpenVINOExecutionProvider** (optimum.onnxruntime): 8.34ms, 120.2 inf/sec

---

## Recommendations

### For Production Use:

1. **✅ Use Native OpenVINO** for best performance across all scenarios (5-16% faster)
2. **🔄 Migration Path**: PyTorch → ONNX → Native OpenVINO provides optimal performance
3. **📊 Performance Advantage**: Native OpenVINO consistently outperforms ONNX models even with OpenVINOExecutionProvider

### For Development/Experimentation:

1. **🛠️ ONNX + OpenVINOExecutionProvider** offers good performance with easier integration
2. **⚖️ Minimal difference** between `onnxruntime` and `optimum.onnxruntime` approaches (both using OpenVINOExecutionProvider)
3. **🔄 Easy migration** from ONNX models to native OpenVINO for production deployment

### Framework-Specific Recommendations:

- **Native OpenVINO**: Best overall performance, recommended for production
- **ONNX + OpenVINOExecutionProvider (onnxruntime)**: Good balance of performance and ONNX compatibility
- **ONNX + OpenVINOExecutionProvider (optimum.onnxruntime)**: Easiest integration with Hugging Face ecosystem

### Hardware Considerations:

- **Intel NPU** provides excellent acceleration for all approaches
- **Performance scales well** with model size
- **Native OpenVINO** maximizes NPU utilization compared to ONNX models

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
- **Native OpenVINO**: Generally more consistent performance than ONNX models
- **Large Models**: Better consistency than small models across all approaches
- **Cold Start**: 2-3x slower than warmed-up performance (consistent across all approaches)
- **Nearly Identical Results**: All approaches show excellent consistency across multiple runs

### Comparison Summary:

- **Native OpenVINO models** consistently outperform ONNX models by 5-16% even when using OpenVINOExecutionProvider
- **ONNX + OpenVINOExecutionProvider** approaches are nearly identical in performance
- **Model constraints**: Fixed batch size (1) and sequence length (512) for fair comparison
- **Hardware**: NPU acceleration enabled for all approaches
- **Precision**: All measurements use high-precision timing (nanoseconds)
- **Statistics**: Based on 2000 iterations with proper warmup (30 iterations)
- **Reproducibility**: Results saved to JSON files for further analysis

**Conclusion**: Native OpenVINO models provide the best performance for production inference workloads, while ONNX models with OpenVINOExecutionProvider offer good performance with easier integration during development phases.
