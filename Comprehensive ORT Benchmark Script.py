import os
import time
import logging
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
import openvino as ov
import statistics
import gc
import psutil
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- File paths (adjust as needed) ---
# Relative path to model directory - adjust based on your setup
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "proxy_models_for_intel", "test_bigger_model"))
# Alternative: Use test_smaller_model for smaller model
# BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "proxy_models_for_intel", "test_smaller_model")

onnx_path = os.path.join(BASE_DIR, "model.onnx")
ov_xml_path = os.path.join(BASE_DIR, "model_openvino_1.xml")
device = "NPU"

# --- Tokenizer setup ---
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def prepare_inputs(text_variety=False):
    """Prepare inputs with text variety"""
    if text_variety:
        # Use different text samples for more realistic benchmarking
        texts = [
            "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing.",
            "Natural language processing is a subfield of artificial intelligence that deals with computer understanding of human language.",
            "Machine learning models require significant computational resources for training and inference on large datasets.",
            "OpenVINO and ONNX Runtime are popular inference frameworks optimized for deployment on various hardware architectures.",
            "Performance benchmarking is crucial for model optimization and selection of the best inference solution.",
            "Deep learning has revolutionized many fields including computer vision, natural language processing, and speech recognition.",
            "Transformer models have become the foundation of modern NLP applications due to their superior performance.",
            "Neural processing units (NPUs) are specialized hardware designed to accelerate machine learning workloads efficiently.",
            "Inference optimization techniques include quantization, pruning, and knowledge distillation to reduce model size.",
            "Edge computing brings AI capabilities closer to data sources, reducing latency and improving user experience."
        ]
        text = texts[np.random.randint(0, len(texts))]
    else:
        text = "This is a comprehensive benchmarking test for model inference latency evaluation using consistent input text."

    inputs = tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=512  # Fixed sequence length
    )

    input_dict = {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'token_type_ids': inputs.get('token_type_ids', np.zeros_like(inputs['input_ids']))
    }

    return input_dict


def warmup_ort_session(ort_session, input_dict, warmup_iterations=30):
    """Warmup the ONNX Runtime session to ensure stable timing"""
    logger.info(
        f"Warming up ONNX Runtime session with {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        _ = ort_session.run(None, input_dict)
    gc.collect()


def warmup_openvino(compiled_model, input_dict, warmup_iterations=30):
    """Warmup OpenVINO model"""
    logger.info(
        f"Warming up OpenVINO model with {warmup_iterations} iterations...")
    for _ in range(warmup_iterations):
        _ = compiled_model(input_dict)
    gc.collect()


def benchmark_ort_with_statistics(ort_session, input_dict, iterations=2000, model_name="ONNX Runtime"):
    """Benchmark ONNX Runtime with detailed statistics using high-precision timing"""
    logger.info(f"Benchmarking {model_name} with {iterations} iterations...")

    # Collect individual inference times
    inference_times = []

    # Force garbage collection before benchmarking
    gc.collect()

    # Use high-precision timing
    for i in range(iterations):
        start = time.perf_counter_ns()
        _ = ort_session.run(None, input_dict)
        end = time.perf_counter_ns()
        inference_times.append((end - start) / 1_000_000)  # Convert to ms

    # Calculate statistics
    mean_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    std_time = statistics.stdev(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)

    return {
        'mean': mean_time,
        'median': median_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'p95': p95_time,
        'p99': p99_time,
        'raw_times': inference_times
    }


def benchmark_openvino_with_statistics(compiled_model, input_dict, iterations=2000, model_name="OpenVINO"):
    """Benchmark OpenVINO with detailed statistics using high-precision timing"""
    logger.info(f"Benchmarking {model_name} with {iterations} iterations...")

    # Collect individual inference times
    inference_times = []

    # Force garbage collection before benchmarking
    gc.collect()

    # Use high-precision timing
    for i in range(iterations):
        start = time.perf_counter_ns()
        _ = compiled_model(input_dict)
        end = time.perf_counter_ns()
        inference_times.append((end - start) / 1_000_000)  # Convert to ms

    # Calculate statistics
    mean_time = statistics.mean(inference_times)
    median_time = statistics.median(inference_times)
    std_time = statistics.stdev(inference_times)
    min_time = min(inference_times)
    max_time = max(inference_times)
    p95_time = np.percentile(inference_times, 95)
    p99_time = np.percentile(inference_times, 99)

    return {
        'mean': mean_time,
        'median': median_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'p95': p95_time,
        'p99': p99_time,
        'raw_times': inference_times
    }


def benchmark_ort_throughput(ort_session, input_dict, duration_seconds=60, model_name="ONNX Runtime"):
    """Benchmark ONNX Runtime throughput over longer duration"""
    logger.info(
        f"Benchmarking {model_name} throughput for {duration_seconds} seconds...")

    start_time = time.perf_counter()
    inference_count = 0
    inference_times = []

    while time.perf_counter() - start_time < duration_seconds:
        iter_start = time.perf_counter()
        _ = ort_session.run(None, input_dict)
        iter_end = time.perf_counter()
        inference_times.append((iter_end - iter_start) * 1000)  # Convert to ms
        inference_count += 1

    total_time = time.perf_counter() - start_time
    throughput = inference_count / total_time

    return throughput, inference_count, total_time, inference_times


def benchmark_openvino_throughput(compiled_model, input_dict, duration_seconds=60, model_name="OpenVINO"):
    """Benchmark OpenVINO throughput over longer duration"""
    logger.info(
        f"Benchmarking {model_name} throughput for {duration_seconds} seconds...")

    start_time = time.perf_counter()
    inference_count = 0
    inference_times = []

    while time.perf_counter() - start_time < duration_seconds:
        iter_start = time.perf_counter()
        _ = compiled_model(input_dict)
        iter_end = time.perf_counter()
        inference_times.append((iter_end - iter_start) * 1000)  # Convert to ms
        inference_count += 1

    total_time = time.perf_counter() - start_time
    throughput = inference_count / total_time

    return throughput, inference_count, total_time, inference_times


def benchmark_text_variety(ort_session, compiled_model, tokenizer, num_samples=500):
    """Benchmark with different text inputs for more realistic scenarios"""
    logger.info(f"=== Text Variety Benchmarking ({num_samples} samples) ===")

    ort_times = []
    ov_times = []

    for i in range(num_samples):
        # Prepare varied inputs
        input_dict = prepare_inputs(text_variety=True)

        # Benchmark ONNX Runtime
        start = time.perf_counter_ns()
        _ = ort_session.run(None, input_dict)
        end = time.perf_counter_ns()
        ort_times.append((end - start) / 1_000_000)

        # Benchmark OpenVINO
        start = time.perf_counter_ns()
        _ = compiled_model(input_dict)
        end = time.perf_counter_ns()
        ov_times.append((end - start) / 1_000_000)

    ort_stats = {
        'mean': statistics.mean(ort_times),
        'median': statistics.median(ort_times),
        'std': statistics.stdev(ort_times),
        'min': min(ort_times),
        'max': max(ort_times),
        'p95': np.percentile(ort_times, 95),
        'p99': np.percentile(ort_times, 99)
    }

    ov_stats = {
        'mean': statistics.mean(ov_times),
        'median': statistics.median(ov_times),
        'std': statistics.stdev(ov_times),
        'min': min(ov_times),
        'max': max(ov_times),
        'p95': np.percentile(ov_times, 95),
        'p99': np.percentile(ov_times, 99)
    }

    return ort_stats, ov_stats


def benchmark_cold_start(ort_session, compiled_model, tokenizer, num_trials=10):
    """Benchmark cold start performance"""
    logger.info(f"=== Cold Start Benchmarking ({num_trials} trials) ===")

    ort_cold_times = []
    ov_cold_times = []

    for i in range(num_trials):
        # Prepare fresh inputs
        input_dict = prepare_inputs()

        # Force garbage collection to simulate cold start
        gc.collect()

        # Benchmark first inference after cold start
        start = time.perf_counter_ns()
        _ = ort_session.run(None, input_dict)
        end = time.perf_counter_ns()
        ort_cold_times.append((end - start) / 1_000_000)

        # Force garbage collection again
        gc.collect()

        # Benchmark OpenVINO cold start
        start = time.perf_counter_ns()
        _ = compiled_model(input_dict)
        end = time.perf_counter_ns()
        ov_cold_times.append((end - start) / 1_000_000)

        # Small delay between trials
        time.sleep(0.1)

    return ort_cold_times, ov_cold_times


def print_detailed_stats(stats, model_name):
    """Print detailed statistics"""
    logger.info(f"=== {model_name} Detailed Statistics ===")
    logger.info(f"Mean: {stats['mean']:.3f} ms")
    logger.info(f"Median: {stats['median']:.3f} ms")
    logger.info(f"Std Dev: {stats['std']:.3f} ms")
    logger.info(f"Min: {stats['min']:.3f} ms")
    logger.info(f"Max: {stats['max']:.3f} ms")
    logger.info(f"P95: {stats['p95']:.3f} ms")
    logger.info(f"P99: {stats['p99']:.3f} ms")
    logger.info(
        f"Coefficient of Variation: {(stats['std']/stats['mean']*100):.2f}%")


def save_results_to_json(results, filename_prefix="ort_benchmark_results"):
    """Save benchmark results to JSON file"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.json"

    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"Results saved to {filename}")


def main():
    logger.info("=== COMPREHENSIVE ONNX RUNTIME LATENCY BENCHMARK ===")
    logger.info(f"Running on device: {device}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # System info
    logger.info("=== System Information ===")
    logger.info(f"CPU Count: {psutil.cpu_count()}")
    logger.info(
        f"Memory Total: {psutil.virtual_memory().total / (1024**3):.2f} GB")
    logger.info(f"ONNX Runtime Version: {ort.__version__}")

    # --- Load models ---
    logger.info("Loading ONNX model with OpenVINO Execution Provider...")
    provider_options = {'device_type': device}
    ort_session = ort.InferenceSession(
        onnx_path,
        providers=['OpenVINOExecutionProvider'],
        provider_options=[provider_options]
    )

    logger.info("Loading OpenVINO model...")
    ie = ov.Core()
    model_ov = ie.read_model(ov_xml_path)
    compiled_model = ie.compile_model(model=model_ov, device_name=device)

    # --- Prepare standard inputs ---
    input_dict = prepare_inputs()

    # --- Cold Start Benchmark ---
    ort_cold_times, ov_cold_times = benchmark_cold_start(
        ort_session, compiled_model, tokenizer, num_trials=10)

    # --- Warmup ---
    warmup_ort_session(ort_session, input_dict, warmup_iterations=30)
    warmup_openvino(compiled_model, input_dict, warmup_iterations=30)

    # --- Detailed Latency Benchmarking ---
    logger.info("=== Detailed Latency Benchmarking (2000 iterations) ===")

    ort_stats = benchmark_ort_with_statistics(
        ort_session, input_dict, iterations=2000, model_name="ONNX Runtime")
    ov_stats = benchmark_openvino_with_statistics(
        compiled_model, input_dict, iterations=2000, model_name="OpenVINO")

    print_detailed_stats(ort_stats, "ONNX Runtime")
    print_detailed_stats(ov_stats, "OpenVINO")

    # --- Throughput Benchmarking ---
    logger.info("=== Throughput Benchmarking (60 seconds) ===")

    ort_throughput, ort_count, ort_time, ort_throughput_times = benchmark_ort_throughput(
        ort_session, input_dict, duration_seconds=60, model_name="ONNX Runtime")
    ov_throughput, ov_count, ov_time, ov_throughput_times = benchmark_openvino_throughput(
        compiled_model, input_dict, duration_seconds=60, model_name="OpenVINO")

    logger.info(
        f"ONNX Runtime Throughput: {ort_throughput:.2f} inferences/sec ({ort_count} inferences in {ort_time:.2f}s)")
    logger.info(
        f"OpenVINO Throughput: {ov_throughput:.2f} inferences/sec ({ov_count} inferences in {ov_time:.2f}s)")
    logger.info(f"Throughput Speedup: {ov_throughput/ort_throughput:.2f}x")

    # --- Text Variety Benchmarking ---
    ort_variety_stats, ov_variety_stats = benchmark_text_variety(
        ort_session, compiled_model, tokenizer, num_samples=500)

    logger.info(
        f"Text Variety - ONNX Runtime: {ort_variety_stats['mean']:.3f}ms ± {ort_variety_stats['std']:.3f}")
    logger.info(
        f"Text Variety - OpenVINO: {ov_variety_stats['mean']:.3f}ms ± {ov_variety_stats['std']:.3f}")

    # --- Final Summary ---
    logger.info("=== FINAL PERFORMANCE SUMMARY ===")
    logger.info(f"Standard Latency Benchmark:")
    logger.info(
        f"  ONNX Runtime - Mean: {ort_stats['mean']:.3f}ms, Median: {ort_stats['median']:.3f}ms, P95: {ort_stats['p95']:.3f}ms")
    logger.info(
        f"  OpenVINO - Mean: {ov_stats['mean']:.3f}ms, Median: {ov_stats['median']:.3f}ms, P95: {ov_stats['p95']:.3f}ms")
    logger.info(f"  Speedup (Mean): {ort_stats['mean']/ov_stats['mean']:.2f}x")
    logger.info(
        f"  Speedup (Median): {ort_stats['median']/ov_stats['median']:.2f}x")
    logger.info(f"  Speedup (P95): {ort_stats['p95']/ov_stats['p95']:.2f}x")

    logger.info(f"Text Variety Benchmark:")
    logger.info(
        f"  ONNX Runtime - Mean: {ort_variety_stats['mean']:.3f}ms, P95: {ort_variety_stats['p95']:.3f}ms")
    logger.info(
        f"  OpenVINO - Mean: {ov_variety_stats['mean']:.3f}ms, P95: {ov_variety_stats['p95']:.3f}ms")
    logger.info(
        f"  Speedup (Mean): {ort_variety_stats['mean']/ov_variety_stats['mean']:.2f}x")

    logger.info(f"Cold Start Performance:")
    logger.info(
        f"  ONNX Runtime - Mean: {statistics.mean(ort_cold_times):.3f}ms")
    logger.info(f"  OpenVINO - Mean: {statistics.mean(ov_cold_times):.3f}ms")
    logger.info(
        f"  Cold Start Speedup: {statistics.mean(ort_cold_times)/statistics.mean(ov_cold_times):.2f}x")

    logger.info(f"Throughput Comparison:")
    logger.info(f"  ONNX Runtime: {ort_throughput:.2f} inferences/sec")
    logger.info(f"  OpenVINO: {ov_throughput:.2f} inferences/sec")
    logger.info(f"  Throughput Speedup: {ov_throughput/ort_throughput:.2f}x")

    # --- Save Results ---
    results = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device': device,
        'model_path': BASE_DIR,
        'onnx_runtime_version': ort.__version__,
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3)
        },
        'latency_benchmark': {
            'onnx_runtime': ort_stats,
            'openvino': ov_stats,
            'speedup': ort_stats['mean']/ov_stats['mean']
        },
        'text_variety_benchmark': {
            'onnx_runtime': ort_variety_stats,
            'openvino': ov_variety_stats,
            'speedup': ort_variety_stats['mean']/ov_variety_stats['mean']
        },
        'cold_start_benchmark': {
            'onnx_runtime_times': ort_cold_times,
            'openvino_times': ov_cold_times,
            'onnx_runtime_mean': statistics.mean(ort_cold_times),
            'openvino_mean': statistics.mean(ov_cold_times),
            'speedup': statistics.mean(ort_cold_times)/statistics.mean(ov_cold_times)
        },
        'throughput_benchmark': {
            'onnx_runtime_throughput': ort_throughput,
            'openvino_throughput': ov_throughput,
            'speedup': ov_throughput/ort_throughput
        }
    }

    save_results_to_json(results)

    logger.info("=== BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    main()
