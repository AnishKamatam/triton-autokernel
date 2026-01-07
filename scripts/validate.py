import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kernels.matmul_template import triton_matmul
from utils.timing import measure_runtime

def run_validation(M=512, N=512, K=512):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    
    torch_output = torch.matmul(a, b)
    triton_output = triton_matmul(a, b)
    
    if torch.allclose(torch_output, triton_output, atol=1e-2, rtol=1e-2):
        print("Correctness Check: PASSED")
    else:
        print("Correctness Check: FAILED")
        return
    
    ms = measure_runtime(triton_matmul, (a, b), {})
    tflops = 2 * M * N * K / (ms * 1e-3) / 1e12
    print(f"Performance: {ms:.3f} ms ({tflops:.2f} TFLOPS)")

if __name__ == "__main__":
    run_validation()

