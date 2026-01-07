# Triton Autokernel

GPU kernel benchmarking harness with LLM-powered configuration generation.

## Setup

```bash
pip install -r requirements.txt
```

Set `OPENAI_API_KEY` in `.env` file for LLM features.

## Usage

```bash
# Single kernel validation
python scripts/validate.py

# Benchmark all registered kernels
python scripts/benchmark.py

# Generate LLM configs and benchmark
python scripts/run_auto.py

# Export best configuration
python scripts/export_best.py
```

## Structure

```
triton-autokernel/
├── kernels/          # Kernel implementations and registry
├── utils/            # Timing and logging utilities
├── llm/              # LLM configuration generator
├── scripts/          # Execution scripts
├── results/          # Benchmark history (CSV)
└── requirements.txt
```
