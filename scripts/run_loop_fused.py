import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from llm.provider import KernelGenerator
from kernels.fused_registry import fused_registry
from scripts.benchmark_fused import run_benchmark_fused

load_dotenv()

def get_history_summary(top_n=5, kernel_type="fused"):
    csv_path = "results/history.csv"
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    
    # Filter by kernel type if needed (you could add a 'type' column)
    winners = df[df['status'] == 'success'].sort_values(by='tflops', ascending=False).head(top_n)
    failures = df[df['status'].str.contains('error', na=False)].tail(top_n)
    
    summary = "TOP PERFORMERS:\n" + winners[['tflops', 'config']].to_string()
    summary += "\n\nRECENT FAILURES:\n" + failures[['status', 'config']].to_string()
    
    return summary

def get_overall_best():
    csv_path = "results/history.csv"
    if not os.path.exists(csv_path):
        return None
    
    df = pd.read_csv(csv_path)
    best = df[df['status'] == 'success'].sort_values(by='tflops', ascending=False).iloc[0]
    return best

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY in .env file or environment variable.")
        exit(1)
    
    gen = KernelGenerator(api_key)
    iterations = 5
    
    for i in range(iterations):
        print(f"\n--- Fused Kernel Generation {i+1} ---")
        
        history = get_history_summary(kernel_type="fused")
        new_configs = gen.generate_configs(n=3, history_context=history, kernel_type="fused")
        
        fused_registry.candidates = []
        for j, cfg in enumerate(new_configs):
            fused_registry.add_candidate(f"fused_gen_{i}_var_{j}", cfg)
        
        print(f"LLM suggested {len(new_configs)} fused kernel configs based on history.")
        run_benchmark_fused(M=4096, N=4096, K=4096)
        
        overall_best = get_overall_best()
        if overall_best is not None:
            print(f"\nOverall Best So Far: {overall_best['name']} - {overall_best['tflops']} TFLOPS")

if __name__ == "__main__":
    main()

