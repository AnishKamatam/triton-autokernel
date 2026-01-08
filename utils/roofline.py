import torch

# A100 GPU hardware specifications
A100_MEMORY_BANDWIDTH_GBPS = 1939.0
A100_PEAK_TFLOPS_FP16 = 312.0

def calculate_arithmetic_intensity(M, N, K, dtype=torch.float16, fused=False):
    # Arithmetic intensity = operations / bytes (higher = more compute-bound)
    bytes_per_element = 2 if dtype == torch.float16 else 4
    total_ops = 2.0 * M * N * K  # 2 ops per multiply-add
    total_bytes = (M * K + K * N + M * N) * bytes_per_element  # Input + output matrices
    arithmetic_intensity = total_ops / total_bytes
    return arithmetic_intensity, total_ops, total_bytes

def calculate_bandwidth_utilization(M, N, K, ms, dtype=torch.float16):
    # Calculate how much of peak memory bandwidth is actually used
    _, _, total_bytes = calculate_arithmetic_intensity(M, N, K, dtype)
    time_seconds = ms * 1e-3
    actual_bandwidth_gbps = (total_bytes / time_seconds) / 1e9
    utilization_percent = (actual_bandwidth_gbps / A100_MEMORY_BANDWIDTH_GBPS) * 100
    return actual_bandwidth_gbps, utilization_percent

def analyze_roofline(M, N, K, ms, tflops, dtype=torch.float16, fused=False, kernel_name=""):
    # Roofline model: performance is limited by either memory or compute bandwidth
    ai, total_ops, total_bytes = calculate_arithmetic_intensity(M, N, K, dtype, fused)
    actual_bandwidth, bandwidth_util = calculate_bandwidth_utilization(M, N, K, ms, dtype)
    
    # Compute theoretical limits
    memory_bound_perf = A100_MEMORY_BANDWIDTH_GBPS * ai / 1e3  # TFLOPS limited by memory
    compute_bound_perf = A100_PEAK_TFLOPS_FP16  # TFLOPS limited by compute
    roofline_limit = min(memory_bound_perf, compute_bound_perf)  # Actual limit is the bottleneck
    is_compute_bound = memory_bound_perf > compute_bound_perf
    
    return {
        "kernel_name": kernel_name,
        "arithmetic_intensity": ai,
        "total_ops": total_ops,
        "total_bytes": total_bytes,
        "actual_bandwidth_gbps": actual_bandwidth,
        "bandwidth_utilization_percent": bandwidth_util,
        "memory_bound_limit_tflops": memory_bound_perf,
        "compute_bound_limit_tflops": compute_bound_perf,
        "roofline_limit_tflops": roofline_limit,
        "is_compute_bound": is_compute_bound,
        "actual_tflops": tflops,
        "efficiency_percent": (tflops / roofline_limit) * 100
    }

def print_roofline_analysis(analysis):
    print(f"\n--- Roofline Analysis: {analysis['kernel_name']} ---")
    print(f"Arithmetic Intensity: {analysis['arithmetic_intensity']:.2f} Ops/Byte")
    print(f"Total Operations: {analysis['total_ops']/1e12:.2f} TFLOP")
    print(f"Total Memory: {analysis['total_bytes']/1e9:.2f} GB")
    print(f"\nMemory Bandwidth:")
    print(f"  Actual: {analysis['actual_bandwidth_gbps']:.2f} GB/s")
    print(f"  Peak (A100): {A100_MEMORY_BANDWIDTH_GBPS:.2f} GB/s")
    print(f"  Utilization: {analysis['bandwidth_utilization_percent']:.1f}%")
    print(f"\nPerformance Limits:")
    print(f"  Memory-bound limit: {analysis['memory_bound_limit_tflops']:.2f} TFLOPS")
    print(f"  Compute-bound limit: {analysis['compute_bound_limit_tflops']:.2f} TFLOPS")
    print(f"  Roofline limit: {analysis['roofline_limit_tflops']:.2f} TFLOPS")
    print(f"\nActual Performance:")
    print(f"  Achieved: {analysis['actual_tflops']:.2f} TFLOPS")
    print(f"  Efficiency: {analysis['efficiency_percent']:.1f}%")
    print(f"  Bound Type: {'COMPUTE-BOUND' if analysis['is_compute_bound'] else 'MEMORY-BOUND'}")
