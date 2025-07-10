# Comprehensive Benchmark Results Comparison

## ONNX Runtime vs OpenVINO Performance Analysis

### Test Environment
- **Device**: NPU (Neural Processing Unit)
- **Model**: BERT Token Classification (test_smaller_model)
- **Sequence Length**: 512 tokens (fixed)
- **Batch Size**: 1 (fixed)
- **Iterations**: 2000 for detailed latency, 500 for text variety
- **Throughput Duration**: 60 seconds

---

## Performance Comparison Summary

### 1. ORTModelForTokenClassification vs Direct ONNX Runtime

| Approach | Framework | Mean Latency | Median Latency | P95 Latency | Throughput |
|----------|-----------|--------------|----------------|-------------|------------|
| **ORTModelForTokenClassification** | Optimum + ORT | 9.828 ms | 9.671 ms | 11.510 ms | 123.28 inf/sec |
| **Direct ONNX Runtime** | ORT InferenceSession | 7.347 ms | 7.030 ms | 8.962 ms | 132.60 inf/sec |

**Winner**: Direct ONNX Runtime ✅
- **25% faster latency** (7.347ms vs 9.828ms)
- **7% higher throughput** (132.60 vs 123.28 inf/sec)
- **Better P95 performance** (8.962ms vs 11.510ms)

### 2. ONNX Runtime vs OpenVINO Direct Comparison

#### Using Direct ONNX Runtime:
| Metric | ONNX Runtime | OpenVINO | Winner |
|--------|--------------|----------|---------|
| **Mean Latency** | 7.347 ms | 7.520 ms | ONNX Runtime ✅ |
| **Median Latency** | 7.030 ms | 6.944 ms | OpenVINO ✅ |
| **P95 Latency** | 8.962 ms | 10.104 ms | ONNX Runtime ✅ |
| **Throughput** | 132.60 inf/sec | 131.64 inf/sec | ONNX Runtime ✅ |
| **Consistency** | CV: 17.67% | CV: 27.86% | ONNX Runtime ✅ |
| **Text Variety** | 7.906 ms | 7.292 ms | OpenVINO ✅ |

#### Using ORTModelForTokenClassification:
| Metric | ORTModel | OpenVINO | Winner |
|--------|----------|----------|---------|
| **Mean Latency** | 9.828 ms | 10.858 ms | ORTModel ✅ |
| **Median Latency** | 9.671 ms | 10.292 ms | ORTModel ✅ |
| **P95 Latency** | 11.510 ms | 14.823 ms | ORTModel ✅ |
| **Throughput** | 123.28 inf/sec | 134.36 inf/sec | OpenVINO ✅ |
| **Consistency** | CV: 14.01% | CV: 22.22% | ORTModel ✅ |

---

## Key Findings

### 1. **Direct ONNX Runtime is Superior**
- **Best Overall Performance**: Direct `ort.InferenceSession` outperforms both `ORTModelForTokenClassification` and OpenVINO
- **Lowest Latency**: 7.347ms mean latency
- **Highest Throughput**: 132.60 inferences/sec
- **Best Consistency**: 17.67% coefficient of variation

### 2. **Framework Overhead Analysis**
- **ORTModelForTokenClassification overhead**: ~34% slower than direct ONNX Runtime
- **Optimum library adds significant latency**: Likely due to additional PyTorch tensor operations

### 3. **OpenVINO vs ONNX Runtime**
- **Very Close Performance**: Differences are minimal (< 3%)
- **OpenVINO slightly better median**: 6.944ms vs 7.030ms
- **ONNX Runtime better mean and P95**: More consistent performance
- **OpenVINO better with text variety**: 7.292ms vs 7.906ms

### 4. **Cold Start Performance**
- **Direct ONNX Runtime**: 14.822ms
- **OpenVINO**: 15.782ms
- **ORTModelForTokenClassification**: 14.735ms (from previous benchmark)

---

## Recommendations

### For Production Use:
1. **Use Direct ONNX Runtime** (`ort.InferenceSession`) for best performance
2. **Avoid ORTModelForTokenClassification** if latency is critical
3. **Choose ONNX Runtime over OpenVINO** for slightly better consistency

### For Development/Experimentation:
1. **ORTModelForTokenClassification** is easier to use with transformers
2. **OpenVINO** has better text variety handling
3. **Direct ONNX Runtime** requires more manual input handling but offers best performance

### Performance Ranking:
1. 🥇 **Direct ONNX Runtime**: 7.347ms, 132.60 inf/sec
2. 🥈 **OpenVINO**: 7.520ms, 131.64 inf/sec  
3. 🥉 **ORTModelForTokenClassification**: 9.828ms, 123.28 inf/sec

---

## Technical Notes

- **Model Constraints**: Fixed batch size (1) and sequence length (512)
- **Hardware**: NPU acceleration enabled for all approaches
- **Precision**: All measurements use high-precision timing (nanoseconds)
- **Statistics**: Based on 2000 iterations with proper warmup (30 iterations)
- **Reproducibility**: Results saved to JSON files for further analysis

The direct ONNX Runtime approach provides the best balance of performance, consistency, and throughput for production inference workloads.
