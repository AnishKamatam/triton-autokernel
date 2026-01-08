import torch
import triton
import triton.language as tl
import json
import os

@triton.jit
def fused_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Compute program ID and tile coordinates using grouped tiling
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Grouped tiling: improves L2 cache reuse by processing nearby M tiles together
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    # Compute memory offsets for this tile
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Compute pointer arrays for coalesced memory access
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # Accumulate dot products across K dimension (blocked matmul)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Apply activation function in-place (fused operation avoids memory round-trip)
    if ACTIVATION == "leaky_relu":
        accumulator = tl.where(accumulator >= 0.0, accumulator, accumulator * 0.01)
    elif ACTIVATION == "relu":
        accumulator = tl.maximum(accumulator, 0.0)
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def triton_fused_matmul(a, b, activation="leaky_relu", use_optimized=True):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 128, 32
    GROUP_SIZE_M = 8
    num_warps, num_stages = 4, 4
    
    # Load optimized configuration if available (from auto-tuning)
    if use_optimized:
        try:
            config_path = os.path.join(os.path.dirname(__file__), "best_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    best_config = json.load(f)
                BLOCK_SIZE_M = best_config.get("BLOCK_SIZE_M", 128)
                BLOCK_SIZE_N = best_config.get("BLOCK_SIZE_N", 128)
                BLOCK_SIZE_K = best_config.get("BLOCK_SIZE_K", 32)
                GROUP_SIZE_M = best_config.get("GROUP_SIZE_M", 8)
                num_warps = best_config.get("num_warps", 4)
                num_stages = best_config.get("num_stages", 4)
        except (json.JSONDecodeError, IOError, KeyError):
            # Fall back to defaults if config file is missing or invalid
            pass
    
    fused_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ACTIVATION=activation,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return c
