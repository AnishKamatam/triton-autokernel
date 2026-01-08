import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.registry import registry
from utils.timing import measure_runtime
from utils.logger import log_result
from utils.roofline import analyze_roofline, print_roofline_analysis

def run_benchmark(M=4096, N=4096, K=4096):
    # Benchmark all kernel configurations in the registry
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ref_out = torch.matmul(a, b)

    results = []

    for candidate in registry.candidates:
        name = candidate["name"]
        launcher = registry.get_launcher(name)
        
        try:
            # Verify correctness first
            out = launcher(a, b)
            if not torch.allclose(ref_out, out, atol=1e-2, rtol=1e-2):
                print(f"FAILED {name}: Correctness check failed")
                log_result(name, candidate["configs"], None, None, status="failed_correctness")
                continue
            
            # Measure performance
            ms = measure_runtime(launcher, (a, b), {})
            tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
            results.append((name, ms, tflops))
            print(f"PASSED {name}: {ms:.3f} ms | {tflops:.2f} TFLOPS")
            log_result(name, candidate["configs"], ms, tflops, status="success")
        
        except Exception as e:
            print(f"ERROR {name}: {e}")
            log_result(name, candidate["configs"], None, None, status=f"error: {str(e)}")

    # Sort by runtime (fastest first)
    results.sort(key=lambda x: x[1])
    print("\n--- Final Rankings ---")
    for i, (name, ms, tflops) in enumerate(results):
        print(f"{i+1}. {name}: {tflops:.2f} TFLOPS")
    
    # Roofline analysis for the best performing kernel
    if results:
        best_name, best_ms, best_tflops = results[0]
        print("\n" + "="*60)
        print(f"ROOFLINE ANALYSIS: {best_name} (Best Kernel)")
        print("="*60)
        best_analysis = analyze_roofline(M, N, K, best_ms, best_tflops, dtype=torch.float16, fused=False, kernel_name=best_name)
        print_roofline_analysis(best_analysis)

if __name__ == "__main__":
    run_benchmark()
