# Architecture Overview

This document describes the technical architecture and design decisions of the Triton Autokernel framework.

## System Overview

Triton Autokernel is a GPU kernel optimization framework that combines:
- **Triton**: GPU kernel programming language
- **LLM Integration**: Automated configuration generation
- **Performance Analysis**: Roofline model and benchmarking
- **Iterative Learning**: Feedback-driven optimization

## Core Components

### 1. Kernel Implementations

#### Standard MatMul Kernel (`kernels/matmul_template.py`)

A blocked matrix multiplication kernel using Triton's JIT compilation:

```python
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
)
```

**Features**:
- **Grouped tiling**: Processes nearby M-dimension tiles together for L2 cache efficiency
- **Coalesced memory access**: Pointer arithmetic designed for GPU memory coalescing
- **Blocked computation**: Accumulates dot products across K dimension

#### Fused MatMul+Activation Kernel (`kernels/fused_matmul_kernel.py`)

Extends the standard matmul with in-place activation:

```python
@triton.jit
def fused_matmul_kernel(
    ...
    ACTIVATION: tl.constexpr,
)
```

**Features**:
- **Fused operations**: MatMul and activation in single kernel
- **Register-level fusion**: Activation computed immediately after matmul
- **Multiple activations**: Supports ReLU and LeakyReLU

**Fusion characteristics**:
- Eliminates one memory round-trip
- May improve cache locality
- Reduces memory bandwidth requirements

### 2. Registry System

#### Kernel Registry (`kernels/registry.py`)

Manages multiple kernel configurations for auto-tuning:

```python
class KernelRegistry:
    def __init__(self):
        self.candidates = []
    
    def add_candidate(self, name, configs):
        # Register a kernel configuration
    
    def get_launcher(self, name):
        # Return a launcher function for a configuration
```

**Design**:
- **Configuration-driven**: Separates kernel parameters from launch metadata
- **Dynamic launcher creation**: Generates launcher functions at runtime
- **Extensible**: Supports adding new kernel types

#### Fused Registry (`kernels/fused_registry.py`)

Specialized registry for fused kernels with activation support.

### 3. LLM Integration (`llm/provider.py`)

Uses OpenAI GPT-5-2025-08-07 to generate kernel configurations:

```python
class KernelGenerator:
    def generate_configs(self, n=5, history_context=None, kernel_type="matmul"):
        # Generate configurations using LLM
        # Optionally informed by benchmark history
```

**Features**:
- **Context-aware**: Incorporates previous benchmark results
- **Failure avoidance**: Analyzes error patterns to avoid problematic configs
- **Performance refinement**: Adjusts successful configs based on performance data

**Prompt Engineering**:
- Provides hardware specifications (A100)
- Includes parameter constraints (power of 2, valid ranges)
- Incorporates feedback from benchmark history

### 4. Performance Analysis

#### Timing (`utils/timing.py`)

Accurate GPU timing using CUDA events:

```python
def measure_runtime(kernel_fn, args, kwargs, n_warmup=10, n_repeat=100):
    # Warmup runs for steady state
    # CUDA events for accurate timing
    # Average over multiple runs
```

**Implementation details**:
- Warmup runs to ensure GPU is in steady state
- CUDA events for GPU-side timing
- Multiple runs to reduce measurement variance

#### Roofline Analysis (`utils/roofline.py`)

Implements the roofline performance model:

```python
def analyze_roofline(M, N, K, ms, tflops, dtype, fused, kernel_name):
    # Calculate arithmetic intensity
    # Determine memory vs compute bound
    # Compute efficiency metrics
```

**Roofline Model**:
- **Arithmetic Intensity**: Operations per byte (Ops/Byte)
- **Memory-bound limit**: `Bandwidth × Arithmetic Intensity`
- **Compute-bound limit**: Peak TFLOPS
- **Roofline limit**: `min(memory_bound, compute_bound)`

**Hardware Specifications** (A100):
- Memory Bandwidth: 1939 GB/s
- Peak TFLOPS (FP16): 312 TFLOPS

#### Logging (`utils/logger.py`)

CSV-based benchmark result tracking:

```python
def log_result(name, config, ms, tflops, status="success"):
    # Append to results/history.csv
    # Track timestamp, performance, status
```

**Data Schema**:
- `timestamp`: When benchmark ran
- `name`: Kernel configuration name
- `ms`: Latency in milliseconds
- `tflops`: Performance in TFLOPS
- `status`: success/failed_correctness/error
- `config`: Kernel configuration dictionary

### 5. Benchmarking Scripts

#### Standard Benchmarks (`scripts/benchmark.py`)

Benchmarks all registered kernel configurations:

1. **Correctness validation**: Compare against PyTorch reference
2. **Performance measurement**: Time each configuration
3. **Result logging**: Save to CSV
4. **Roofline analysis**: Analyze best performer

#### Fused Benchmarks (`scripts/benchmark_fused.py`)

Specialized for fused kernels with activation-specific reference generation.

#### Comparison (`scripts/bench_fusion.py`)

Direct comparison between PyTorch and Triton implementations:
- PyTorch non-fused baseline
- Triton default configuration
- Triton optimized configuration

### 6. Autotuning Scripts

#### Single Generation (`scripts/run_auto.py`)

One-shot LLM configuration generation and benchmarking.

#### Iterative Loop (`scripts/run_loop.py`, `scripts/run_loop_fused.py`)

Multi-generation optimization:

1. **Generate configs**: LLM suggests configurations
2. **Benchmark**: Test all configurations
3. **Analyze results**: Extract top performers and failures
4. **Feedback**: Provide history to LLM for next generation
5. **Repeat**: Continue until convergence

**Learning Mechanism**:
- Tracks successful configurations
- Identifies failure patterns
- Refines parameters based on performance data

## Design Decisions

### 1. Register Pressure Management

**Challenge**: Larger block sizes improve parallelism but increase register usage.

**Solution**: The LLM learns to balance:
- `BLOCK_SIZE_M × BLOCK_SIZE_N`: Tile size
- `num_warps`: Number of warps per block
- Activation logic complexity

**Optimal Configuration**: `BLOCK_SIZE_M=256, BLOCK_SIZE_N=128` represents careful trade-off.

### 2. L2 Cache Locality

**Challenge**: Improve cache hit rates for memory-bound operations.

**Solution**: **Grouped Program ID Mapping (Swizzling)**:
- Process nearby M-dimension tiles together (`GROUP_SIZE_M=8`)
- Improves L2 cache reuse
- Reduces memory latency

**Impact**: 12.1% performance improvement from default to optimized.

### 3. Numerical Stability

**Challenge**: Fused operators accumulate floating-point errors.

**Solution**: Comprehensive correctness validation:
- `torch.allclose()` with `atol=1e-2, rtol=1e-2`
- Catches numerical drift in fused kernels
- Ensures production-ready implementations

**Example**: `fused_gen_0_var_0` failed due to numerical drift, system learned to avoid similar configs.

### 4. Configuration Management

**Challenge**: Track and export optimal configurations.

**Solution**: JSON-based configuration storage:
- `kernels/best_config.json`: Auto-discovered optimal config
- Loaded automatically by optimized kernels
- Easy to version control and share

## Data Flow

```
┌─────────────────┐
│  LLM Provider   │
│  (GPT-5-2025-08-07) │
└────────┬────────┘
         │ Generate configs
         ▼
┌─────────────────┐
│   Registry      │
│  (Configs)      │
└────────┬────────┘
         │ Register candidates
         ▼
┌─────────────────┐
│   Benchmark     │
│   Scripts       │
└────────┬────────┘
         │ Execute kernels
         ▼
┌─────────────────┐
│  Performance    │
│  Analysis       │
│  (Timing,       │
│   Roofline)     │
└────────┬────────┘
         │ Log results
         ▼
┌─────────────────┐
│  History CSV     │
│  (151+ entries)  │
└────────┬────────┘
         │ Feedback
         ▼
┌─────────────────┐
│  LLM Provider    │
│  (Next Gen)     │
└─────────────────┘
```

## Extension Points

### Adding New Kernel Types

1. Create kernel implementation in `kernels/`
2. Create registry class (inherit from `KernelRegistry`)
3. Add benchmark script in `scripts/`
4. Update LLM prompts in `llm/provider.py`

### Adding New LLM Providers

1. Implement `generate_configs()` method
2. Support history context format
3. Return JSON list of configurations

### Adding New Analysis Tools

1. Create utility in `utils/`
2. Integrate with benchmark scripts
3. Add to logging system

## Performance Considerations

### Memory Access Patterns

- **Coalesced access**: Adjacent threads access adjacent memory
- **Blocked computation**: Reduces memory traffic
- **Grouped tiling**: Improves cache locality

### Compute Optimization

- **Register usage**: Balance tile size vs register pressure
- **Warp scheduling**: Optimal `num_warps` for occupancy
- **Pipeline stages**: `num_stages` for instruction-level parallelism

### Fusion Benefits

- **Register-level fusion**: No intermediate memory writes
- **Cache efficiency**: Data stays in registers/cache
- **Reduced bandwidth**: Fewer memory transactions

## Future Enhancements

1. **Multi-GPU support**: Scale across multiple GPUs
2. **Additional kernels**: Convolution, attention, etc.
3. **Visualization**: Performance charts and roofline plots
4. **Distributed tuning**: Parallel configuration testing
5. **Hardware abstraction**: Support for different GPU architectures

