import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.fused_matmul_kernel import triton_fused_matmul
from utils.timing import measure_runtime
from utils.roofline import analyze_roofline, print_roofline_analysis

def run_comparison(M=4096, N=4096, K=4096):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    ref_out = torch.nn.functional.leaky_relu(torch.matmul(a, b))
    
    def pytorch_non_fused(a, b):
        return torch.nn.functional.leaky_relu(torch.matmul(a, b))
    
    pt_ms = measure_runtime(pytorch_non_fused, (a, b), {})
    tri_default_ms = measure_runtime(lambda a, b: triton_fused_matmul(a, b, use_optimized=False), (a, b), {})
    tri_optimized_ms = measure_runtime(lambda a, b: triton_fused_matmul(a, b, use_optimized=True), (a, b), {})
    
    triton_out = triton_fused_matmul(a, b, use_optimized=True)
    if torch.allclose(ref_out, triton_out, atol=1e-2, rtol=1e-2):
        print("Correctness Check: PASSED")
    else:
        print("Correctness Check: FAILED")
        return
    
    print(f"PyTorch (Non-fused): {pt_ms:.3f} ms")
    print(f"Triton Fused (Default): {tri_default_ms:.3f} ms | Speedup: {pt_ms / tri_default_ms:.2f}x")
    print(f"Triton Fused (Optimized): {tri_optimized_ms:.3f} ms | Speedup: {pt_ms / tri_optimized_ms:.2f}x")
    
    pt_tflops = (2.0 * M * N * K) / (pt_ms * 1e-3) / 1e12
    default_tflops = (2.0 * M * N * K) / (tri_default_ms * 1e-3) / 1e12
    optimized_tflops = (2.0 * M * N * K) / (tri_optimized_ms * 1e-3) / 1e12
    
    print(f"\nPerformance:")
    print(f"PyTorch: {pt_tflops:.2f} TFLOPS")
    print(f"Default: {default_tflops:.2f} TFLOPS")
    print(f"Optimized: {optimized_tflops:.2f} TFLOPS")
    
    print("\n" + "="*60)
    print("MEMORY ROOFLINE ANALYSIS")
    print("="*60)
    
    pt_analysis = analyze_roofline(M, N, K, pt_ms, pt_tflops, dtype=torch.float16, fused=False, kernel_name="PyTorch (Non-fused)")
    print_roofline_analysis(pt_analysis)
    
    opt_analysis = analyze_roofline(M, N, K, tri_optimized_ms, optimized_tflops, dtype=torch.float16, fused=True, kernel_name="Triton Fused (Optimized)")
    print_roofline_analysis(opt_analysis)
    
    print("\n" + "="*60)
    print("ARITHMETIC INTENSITY COMPARISON")
    print("="*60)
    print(f"PyTorch (Non-fused) AI: {pt_analysis['arithmetic_intensity']:.2f} Ops/Byte")
    print(f"Triton Fused AI: {opt_analysis['arithmetic_intensity']:.2f} Ops/Byte")
    print(f"\nKey Insight:")
    print(f"  Both kernels have the same arithmetic intensity (same memory access pattern)")
    print(f"  But fused kernel avoids 1 memory round-trip, improving effective bandwidth utilization")
    if tri_optimized_ms < pt_ms:
        print(f"  This explains the {((pt_ms / tri_optimized_ms - 1) * 100):.1f}% speedup!")
    else:
        print(f"  PyTorch's cuBLAS optimization still provides better performance.")
    
    if tri_optimized_ms < pt_ms:
        print(f"\nFused kernel beats PyTorch by {(pt_ms / tri_optimized_ms - 1) * 100:.1f}%")
    else:
        print(f"\nPyTorch is still {(tri_optimized_ms / pt_ms - 1) * 100:.1f}% faster")
        print("Note: PyTorch uses highly optimized cuBLAS. Fusion benefits are larger with multiple operations.")

if __name__ == "__main__":
    run_comparison()
