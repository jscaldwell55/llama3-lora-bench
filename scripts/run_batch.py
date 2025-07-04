#!/usr/bin/env python3
import yaml
import subprocess
import time
from pathlib import Path

def run_batch_benchmarks(config_path, model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
    """Run multiple benchmark configurations"""
    
    with open(config_path, 'r') as f:
        batch_config = yaml.safe_load(f)
    
    results_dir = f"results/{batch_config['name']}"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Running {len(batch_config['configurations'])} configurations...")
    
    for i, config in enumerate(batch_config['configurations']):
        print(f"\n[{i+1}/{len(batch_config['configurations'])}] Running {config['name']}...")
        
        # Create temp config file
        temp_config = {
            "name": config['name'],
            "lora": config['lora'],
            "training": batch_config['training'],
            "dataset": batch_config.get('dataset', {"max_samples": 100})
        }
        
        temp_path = f"configs/temp_{config['name']}.yaml"
        with open(temp_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        # Run benchmark
        cmd = [
            "python", "scripts/benchmark_working.py",
            "--config", temp_path,
            "--model_name", model_name,
            "--output_dir", results_dir
        ]
        
        subprocess.run(cmd)
        time.sleep(2)  # Brief pause between runs
    
    print(f"\nAll benchmarks complete! Results in {results_dir}/")

if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/comprehensive_test.yaml"
    run_batch_benchmarks(config_path)
