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

def get_overall_best():
    # Find the best performing configuration across all benchmarks
    if not os.path.exists("results/history.csv"):
        return None
    
    df = pd.read_csv("results/history.csv")
    best = df[df['status'] == 'success'].sort_values(by='tflops', ascending=False).iloc[0]
    return best

def main():
    # Iterative auto-tuning: use LLM to generate configs based on previous results
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env file or environment variable.")
        exit(1)
    
    gen = KernelGenerator(api_key)
    iterations = 5
    
    for i in range(iterations):
        print(f"\n--- Feedback Generation {i+1} ---")
        
        # Get history to inform next generation
        history = get_history_summary()
        new_configs = gen.generate_configs(n=3, history_context=history)
        
        # Clear registry to focus on new variants
        registry.candidates = []
        for j, cfg in enumerate(new_configs):
            registry.add_candidate(f"gen_{i}_var_{j}", cfg)
        
        print(f"LLM suggested {len(new_configs)} refined configs based on history.")
        run_benchmark(M=4096, N=4096, K=4096)
        
        # Show overall best so far
        overall_best = get_overall_best()
        if overall_best is not None:
            print(f"\nOverall Best So Far: {overall_best['name']} - {overall_best['tflops']} TFLOPS")

if __name__ == "__main__":
    main()
