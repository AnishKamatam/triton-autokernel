import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from llm.provider import KernelGenerator
from kernels.registry import registry
from scripts.benchmark import run_benchmark

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Please set OPENAI_API_KEY in .env file or environment variable.")
    exit(1)

gen = KernelGenerator(api_key)
print("Requesting kernel configurations from LLM...")
new_configs = gen.generate_configs(n=5)

for i, cfg in enumerate(new_configs):
    registry.add_candidate(f"llm_variant_{i}", cfg)

print(f"Added {len(new_configs)} LLM variants to registry. Starting benchmark...\n")
run_benchmark(M=4096, N=4096, K=4096)
