import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_registry import fused_registry
from utils.timing import measure_runtime
from utils.logger import log_result

def run_benchmark_fused(M=4096, N=4096, K=4096):
    # Benchmark fused matmul+activation kernels
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)

    results = []

    for candidate in fused_registry.candidates:
        name = candidate["name"]
        configs = candidate["configs"]
        launcher = fused_registry.get_launcher(name)
        
        # Get activation type from config, default to leaky_relu
        activation = configs.get("ACTIVATION", "leaky_relu")
        
        # Generate reference output based on activation type
        if activation == "leaky_relu":
            ref_out = torch.nn.functional.leaky_relu(torch.matmul(a, b))
        elif activation == "relu":
            ref_out = torch.nn.functional.relu(torch.matmul(a, b))
        else:
            ref_out = torch.matmul(a, b)
        
        try:
            out = launcher(a, b)
            if not torch.allclose(ref_out, out, atol=1e-2, rtol=1e-2):
                print(f"FAILED {name}: Correctness check failed")
                log_result(name, configs, None, None, status="failed_correctness")
                continue
            
            ms = measure_runtime(launcher, (a, b), {})
            tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
            results.append((name, ms, tflops))
            print(f"PASSED {name}: {ms:.3f} ms | {tflops:.2f} TFLOPS")
            log_result(name, configs, ms, tflops, status="success")
        
        except Exception as e:
            print(f"ERROR {name}: {e}")
            log_result(name, configs, None, None, status=f"error: {str(e)}")

    results.sort(key=lambda x: x[1])
    print("\n--- Final Rankings ---")
    for i, (name, ms, tflops) in enumerate(results):
        print(f"{i+1}. {name}: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    run_benchmark_fused()
