import torch
import triton
import triton.language as tl
from kernels.matmul_template import matmul_kernel

class KernelRegistry:
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
            matmul_kernel[grid](
                a, b, c,
                M, N, K,
                a.stride(0), a.stride(1),
                b.stride(0), b.stride(1),
                c.stride(0), c.stride(1),
                **configs
            )
            return c
        return launcher

registry = KernelRegistry()
registry.add_candidate("small_tiles", {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8})
registry.add_candidate("med_tiles", {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32, "GROUP_SIZE_M": 8})
registry.add_candidate("large_tiles", {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8})
