import torch
import triton
import triton.language as tl
from kernels.fused_matmul_kernel import fused_matmul_kernel

class FusedKernelRegistry:
    def __init__(self):
        self.candidates = []

    def add_candidate(self, name, configs):
        self.candidates.append({
            "name": name,
            "configs": configs
        })

    def get_launcher(self, name):
        for c in self.candidates:
            if c["name"] == name:
                return self._create_launcher(c["configs"])
        return None

    def _create_launcher(self, configs):
        def launcher(a, b):
            M, K = a.shape
            _, N = b.shape
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            grid = lambda META: (
                triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
            )
            
            launch_metadata = {}
            if "num_warps" in configs:
                launch_metadata["num_warps"] = configs["num_warps"]
            if "num_stages" in configs:
                launch_metadata["num_stages"] = configs["num_stages"]
            
            kernel_params = {k: v for k, v in configs.items() 
                           if k not in ["num_warps", "num_stages"]}
            
            if "ACTIVATION" not in kernel_params:
                kernel_params["ACTIVATION"] = "leaky_relu"
            
            fused_matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                **kernel_params,
                **launch_metadata
            )
            return c
        return launcher

fused_registry = FusedKernelRegistry()
