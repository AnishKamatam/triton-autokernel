# Triton Autokernel: LLM-Powered GPU Kernel Optimization

A GPU kernel benchmarking and auto-tuning framework that uses Large Language Models (LLMs) to discover Triton kernel configurations. This project explores how LLM-driven optimization compares to manual tuning and standard implementations like cuBLAS.

## Results Summary

- **Performance**: Optimized fused kernels achieve **204.71 TFLOPS** vs PyTorch's 199.13 TFLOPS (3% improvement)
- **Hardware efficiency**: 65.6% of roofline limit on NVIDIA A100 GPUs
- **Autotuning**: LLM-driven configuration search across 151+ benchmark runs
- **Validation**: Correctness checks, roofline analysis, and performance tracking

## Performance Highlights

### Speedup Baseline Comparison

| Implementation | Latency (ms) | TFLOPS | Speedup vs PyTorch |
|----------------|--------------|--------|-------------------|
| **PyTorch cuBLAS** (Non-fused) | 0.690 | 199.13 | Baseline |
| Triton Fused (Default) | 0.740 | 182.69 | 0.93x |
| **Triton Fused (Optimized)** | **0.671** | **204.71** | **1.03x** |

### Best Configurations Found

**Standard MatMul**: `llm_variant_1` - **222.21 TFLOPS**
- Config: `BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, GROUP_SIZE_M=8, num_warps=4, num_stages=4`

**Fused MatMul+Activation**: `fused_gen_2_var_1` - **201.80 TFLOPS**
- Config: `BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64, GROUP_SIZE_M=8, ACTIVATION='leaky_relu', num_warps=16, num_stages=4`

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env` file for LLM-powered autotuning:

```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Basic Usage

```bash
# Validate kernel correctness
python scripts/validate.py

# Benchmark all registered kernels
python scripts/benchmark.py

# Compare fused vs non-fused performance
python scripts/bench_fusion.py

# LLM-powered autotuning (single generation)
python scripts/run_auto.py

# Iterative feedback loop optimization
python scripts/run_loop.py          # Standard kernels
python scripts/run_loop_fused.py     # Fused kernels

# Export best configuration
python scripts/export_best.py
```

## Project Structure

```
triton-autokernel/
├── kernels/              # Kernel implementations
│   ├── matmul_template.py      # Standard matmul kernel
│   ├── fused_matmul_kernel.py  # Fused matmul+activation kernel
│   ├── registry.py             # Kernel configuration registry
│   ├── fused_registry.py       # Fused kernel registry
│   └── best_config.json        # Auto-discovered optimal config
├── utils/               # Performance analysis utilities
│   ├── timing.py              # GPU timing measurement
│   ├── roofline.py            # Roofline model analysis
│   └── logger.py              # Benchmark result logging
├── llm/                 # LLM integration
│   └── provider.py            # OpenAI API wrapper for config generation
├── scripts/             # Execution scripts
│   ├── benchmark.py           # Standard kernel benchmarking
│   ├── benchmark_fused.py      # Fused kernel benchmarking
│   ├── bench_fusion.py         # Performance comparison
│   ├── run_auto.py             # Single-generation LLM tuning
│   ├── run_loop.py             # Iterative LLM tuning (standard)
│   ├── run_loop_fused.py       # Iterative LLM tuning (fused)
│   ├── export_best.py          # Export best config to JSON
│   └── validate.py             # Correctness validation
├── results/             # Benchmark results
│   └── history.csv             # Complete benchmark history (151+ entries)
└── docs/                # Documentation
    ├── BENCHMARKS.md           # Detailed benchmark results
    ├── ARCHITECTURE.md         # Technical architecture
    └── PERFORMANCE.md          # Performance analysis
```

## Technical Analysis

### A. The Speedup Baseline

Optimized fused kernels achieve **204.71 TFLOPS**, compared to PyTorch's cuBLAS implementation at 199.13 TFLOPS (3% improvement). Observations:

1. Manual Triton optimization can achieve performance comparable to standard libraries
2. LLM-driven autotuning can discover configurations that perform well
3. Fusion optimization eliminates memory round-trips, improving effective bandwidth utilization

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for complete performance breakdown.

### B. LLM Learning Curve

The autotuning system shows learning behavior across generations:

- **Early generations**: Higher failure rates due to incorrect configurations
- **Later generations**: More stable performance as the system incorporates historical data
- **Performance progression**: Peak performance improves from 161.87 TFLOPS (Gen 0) to 201.80 TFLOPS (Gen 2)

The system incorporates feedback from:
- **Correctness failures**: Avoids configurations causing numerical errors
- **Performance data**: Adjusts successful configs by modifying `num_stages` and `num_warps`
- **Hardware constraints**: Learns to avoid shared memory OOM errors

### C. Hardware Limit (Roofline Analysis)

**Arithmetic Intensity**: 1365.33 Ops/Byte

The kernels are **compute-bound**, meaning performance is limited by GPU compute capability rather than memory bandwidth:

- **Memory Bandwidth Utilization**: 7.7% (149.93 GB/s of 1939 GB/s peak)
- **Efficiency**: 65.6% of roofline limit
- **Bound Type**: COMPUTE-BOUND

The high arithmetic intensity suggests efficient memory access patterns and use of GPU registers and caches.

### D. Fusion Advantage

While TFLOPS appear similar, the fused kernel achieves better latency by:

1. **Eliminating memory round-trip**: Fused activation avoids writing intermediate matmul result to global memory
2. **Improved cache locality**: Activation computed immediately after matmul while data is still in registers
3. **Reduced memory pressure**: One less global memory transaction per element

## Technical Insights

### Register Pressure Management

The system balances larger `BLOCK_SIZE` values against register usage. Larger blocks improve parallelism but increase register pressure, especially with fused activation logic (LeakyReLU). The selected configuration (`BLOCK_SIZE_M=256, BLOCK_SIZE_N=128`) represents a trade-off between throughput and register usage.

### L2 Cache Locality

**Grouped Program ID Mapping (Swizzling)** contributed to a **12.1% performance improvement** from default to optimized configuration. By processing nearby M-dimension tiles together (`GROUP_SIZE_M=8`), L2 cache hit rates improve, reducing memory latency for subsequent tile loads.

### Numerical Drift

Early fused kernel attempts (`fused_gen_0_var_0`) failed correctness checks due to floating-point error accumulation. The fused operator chain (matmul → activation) accumulates small numerical differences that can exceed tolerance thresholds. The validation harness (`scripts/validate.py`) uses `torch.allclose()` with `atol=1e-2, rtol=1e-2` to detect these issues.

## Benchmark Results Summary

- **Total Benchmark Entries**: 151+ configurations tested
- **Best Standard MatMul**: 222.21 TFLOPS (24% better than manual baseline)
- **Best Fused Kernel**: 204.71 TFLOPS (3% faster than PyTorch)
- **LLM Convergence**: Successfully learned from failures and successes
- **L2 Cache Optimization**: 12.1% improvement from grouped tiling

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for complete benchmark data and analysis.

## Requirements

- Python 3.8+
- CUDA-capable GPU (tested on NVIDIA A100)
- PyTorch with CUDA support
- Triton compiler
- OpenAI API key (for LLM autotuning)

## Documentation

- [Complete Benchmark Results](docs/BENCHMARKS.md) - Detailed performance analysis
- [Architecture Overview](docs/ARCHITECTURE.md) - Technical design and implementation
- [Performance Analysis](docs/PERFORMANCE.md) - Roofline model and optimization insights

## Contributing

Contributions welcome for:
- Additional kernel types (convolution, attention, etc.)
- Alternative LLM providers
- Performance visualization tools
- Extended hardware support

## License

MIT License - see LICENSE file for details.

---

**Built with**: Triton, PyTorch, OpenAI GPT-4o | **Tested on**: NVIDIA A100 GPU
