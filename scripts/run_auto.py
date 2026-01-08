import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from llm.provider import KernelGenerator
from kernels.registry import registry
from scripts.benchmark import run_benchmark

load_dotenv()

def get_history_summary(top_n=5):
    # Extract top performers and recent failures from benchmark history
    if not os.path.exists("results/history.csv"):
        return None
    
    df = pd.read_csv("results/history.csv")
    
    # Get top performers
    winners = df[df['status'] == 'success'].sort_values(by='tflops', ascending=False).head(top_n)
    # Get recent failures to avoid repeating mistakes
    failures = df[df['status'].str.contains('error', na=False)].tail(top_n)
    
    summary = "TOP PERFORMERS:\n" + winners[['tflops', 'config']].to_string()
    summary += "\n\nRECENT FAILURES:\n" + failures[['status', 'config']].to_string()
    
    return summary

# Generate kernel configs using LLM and benchmark them
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Please set OPENAI_API_KEY in .env file or environment variable.")
    exit(1)

gen = KernelGenerator(api_key)
print("Requesting kernel configurations from LLM...")

# Always use history if available
history = get_history_summary()
if history:
    print("Using benchmark history to inform configuration generation...")
    new_configs = gen.generate_configs(n=5, history_context=history)
else:
    print("No benchmark history found, generating initial configurations...")
    new_configs = gen.generate_configs(n=5)

# Add LLM-generated configs to registry
for i, cfg in enumerate(new_configs):
    registry.add_candidate(f"llm_variant_{i}", cfg)

print(f"Added {len(new_configs)} LLM variants to registry. Starting benchmark...\n")
run_benchmark(M=4096, N=4096, K=4096)
