import pandas as pd
import json
import os
import sys
import ast
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def export_best():
    # Export the best performing kernel configuration to JSON
    csv_path = "results/history.csv"
    
    if not os.path.exists(csv_path):
        print("No results found. Run benchmark first.")
        return
    
    df = pd.read_csv(csv_path)
    
    if df.empty or df[df['status'] == 'success'].empty:
        print("No successful results found.")
        return
    
    # Find best configuration by TFLOPS
    best = df[df['status'] == 'success'].sort_values(by='tflops', ascending=False).iloc[0]
    
    print(f"Best Kernel Found: {best['name']}")
    print(f"Performance: {best['tflops']} TFLOPS")
    print(f"Config: {best['config']}")
    
    os.makedirs("kernels", exist_ok=True)
    
    try:
        # Parse config string and save as JSON
        config_dict = ast.literal_eval(best['config'])
        with open("kernels/best_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
        print("\nBest configuration saved to kernels/best_config.json")
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing config: {e}")
    except Exception as e:
        print(f"Error saving config: {e}")

if __name__ == "__main__":
    export_best()
