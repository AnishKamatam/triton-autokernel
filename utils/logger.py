import csv
import os
from datetime import datetime

def log_result(name, config, ms, tflops, status="success"):
    # Log benchmark results to CSV for tracking and analysis
    file_path = "results/history.csv"
    os.makedirs("results", exist_ok=True)
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header if file is new
        if not file_exists:
            writer.writerow(["timestamp", "name", "ms", "tflops", "status", "config"])
        
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            name,
            f"{ms:.4f}" if ms else "N/A",
            f"{tflops:.2f}" if tflops else "0",
            status,
            str(config)
        ])
