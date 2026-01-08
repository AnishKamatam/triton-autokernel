# Triton Autokernel

GPU kernel benchmarking harness with LLM-powered configuration generation.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env` file for LLM features.

## Usage

### Standard MatMul Kernels

```bash
# Single kernel validation
python scripts/validate.py

# Benchmark all registered kernels
python scripts/benchmark.py

# Generate LLM configs and benchmark
python scripts/run_auto.py

# Iterative feedback loop optimization
python scripts/run_loop.py
```

### Fused Kernels (MatMul + Activation)

```bash
# Compare fused vs non-fused performance
python scripts/bench_fusion.py

# Benchmark fused kernel configs
python scripts/benchmark_fused.py

# LLM autotune fused kernels
python scripts/run_loop_fused.py
```

### Utilities

```bash
# Export best configuration
python scripts/export_best.py
```

## Structure

```
triton-autokernel/
├── kernels/          # Kernel implementations and registries
├── utils/            # Timing, logging, and roofline analysis
├── llm/              # LLM configuration generator
├── scripts/          # Execution scripts
├── results/          # Benchmark history (CSV)
└── requirements.txt
```
