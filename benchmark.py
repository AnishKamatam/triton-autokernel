import torch
from kernels.registry import registry
from utils.timing import measure_runtime

def run_benchmark(M=4096, N=4096, K=4096):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    ref_out = torch.matmul(a, b)

    results = []

    for candidate in registry.candidates:
        name = candidate["name"]
        launcher = registry.get_launcher(name)
        
        try:
            out = launcher(a, b)
            if not torch.allclose(ref_out, out, atol=1e-2, rtol=1e-2):
                print(f"FAILED {name}: Correctness check failed")
                continue
            
            ms = measure_runtime(launcher, (a, b), {})
            tflops = (2.0 * M * N * K) / (ms * 1e-3) / 1e12
            results.append((name, ms, tflops))
            print(f"PASSED {name}: {ms:.3f} ms | {tflops:.2f} TFLOPS")
        
        except Exception as e:
            print(f"ERROR {name}: {e}")

    results.sort(key=lambda x: x[1])
    print("\n--- Final Rankings ---")
    for i, (name, ms, tflops) in enumerate(results):
        print(f"{i+1}. {name}: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    run_benchmark()
