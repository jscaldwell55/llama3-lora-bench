# LoRA Benchmark Results Analysis

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

## Load Results

def load_all_results(results_dir="results"):
    """Load all benchmark results from directory"""
    results = []
    
    for json_file in Path(results_dir).rglob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
                results.append(data)
        except:
            print(f"Error loading {json_file}")
            
    return pd.DataFrame(results)

# Load results
df = load_all_results()
print(f"Loaded {len(df)} benchmark results")

# Extract LoRA parameters into separate columns
if len(df) > 0:
    df['lora_r'] = df['config'].apply(lambda x: x['lora']['r'])
    df['lora_alpha'] = df['config'].apply(lambda x: x['lora']['lora_alpha'])
    df['lora_dropout'] = df['config'].apply(lambda x: x['lora']['lora_dropout'])
    df['target_modules'] = df['config'].apply(lambda x: len(x['lora']['target_modules']))

## Summary Statistics

if len(df) > 0:
    print("\n📊 Summary Statistics:")
    print(f"Total benchmarks run: {len(df)}")
    print(f"Models tested: {df['model'].nunique()}")
    print(f"LoRA ranks tested: {sorted(df['lora_r'].unique())}")
    print(f"\nPerformance ranges:")
    print(f"  Training time: {df['training_time'].min():.1f}s - {df['training_time'].max():.1f}s")
    print(f"  Final loss: {df['final_loss'].min():.3f} - {df['final_loss'].max():.3f}")
    print(f"  Peak memory: {df['peak_gpu_memory_gb'].min():.2f}GB - {df['peak_gpu_memory_gb'].max():.2f}GB")

## Visualizations

### 1. Memory Usage vs LoRA Rank

plt.figure(figsize=(10, 6))
if len(df) > 1:
    sns.lineplot(data=df, x='lora_r', y='peak_gpu_memory_gb', marker='o', markersize=10)
    plt.title('GPU Memory Usage vs LoRA Rank', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Peak GPU Memory (GB)', fontsize=14)
    plt.grid(True, alpha=0.3)
else:
    plt.scatter(df['lora_r'], df['peak_gpu_memory_gb'], s=100)
    plt.title('GPU Memory Usage', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Peak GPU Memory (GB)', fontsize=14)
plt.tight_layout()
plt.show()

### 2. Training Time vs LoRA Rank

plt.figure(figsize=(10, 6))
if len(df) > 1:
    sns.lineplot(data=df, x='lora_r', y='training_time', marker='o', markersize=10)
    plt.title('Training Time vs LoRA Rank', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
    plt.grid(True, alpha=0.3)
else:
    plt.scatter(df['lora_r'], df['training_time'], s=100)
    plt.title('Training Time', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Training Time (seconds)', fontsize=14)
plt.tight_layout()
plt.show()

### 3. Final Loss Comparison

plt.figure(figsize=(10, 6))
if len(df) > 1:
    df_sorted = df.sort_values('lora_r')
    plt.bar(df_sorted['lora_r'].astype(str), df_sorted['final_loss'])
    plt.title('Final Training Loss by LoRA Rank', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Final Loss', fontsize=14)
    plt.xticks(rotation=45)
else:
    plt.bar([str(df.iloc[0]['lora_r'])], [df.iloc[0]['final_loss']])
    plt.title('Final Training Loss', fontsize=16)
    plt.xlabel('LoRA Rank (r)', fontsize=14)
    plt.ylabel('Final Loss', fontsize=14)
plt.tight_layout()
plt.show()

### 4. Efficiency Analysis

# Calculate efficiency metrics
if len(df) > 0:
    df['params_per_gb'] = df['trainable_parameters'] / df['peak_gpu_memory_gb'] / 1e6  # Millions of params per GB
    df['samples_per_gb'] = df['samples_per_second'] / df['peak_gpu_memory_gb']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Parameters per GB
    if len(df) > 1:
        ax1.plot(df['lora_r'], df['params_per_gb'], marker='o', markersize=10)
    else:
        ax1.scatter(df['lora_r'], df['params_per_gb'], s=100)
    ax1.set_title('Memory Efficiency: Parameters per GB', fontsize=14)
    ax1.set_xlabel('LoRA Rank (r)', fontsize=12)
    ax1.set_ylabel('Million Parameters per GB', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Throughput per GB
    if len(df) > 1:
        ax2.plot(df['lora_r'], df['samples_per_gb'], marker='o', markersize=10)
    else:
        ax2.scatter(df['lora_r'], df['samples_per_gb'], s=100)
    ax2.set_title('Compute Efficiency: Throughput per GB', fontsize=14)
    ax2.set_xlabel('LoRA Rank (r)', fontsize=12)
    ax2.set_ylabel('Samples/sec per GB', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

### 5. Comprehensive Comparison Table

if len(df) > 0:
    # Create summary table
    summary_df = df[['lora_r', 'lora_alpha', 'trainable_parameters', 
                     'peak_gpu_memory_gb', 'training_time', 'final_loss', 
                     'samples_per_second']].copy()
    
    summary_df.columns = ['R', 'Alpha', 'Trainable Params', 'Peak GPU (GB)', 
                          'Time (s)', 'Final Loss', 'Samples/sec']
    
    summary_df = summary_df.sort_values('R')
    
    print("\n📊 Detailed Results Table:")
    print(summary_df.to_string(index=False, float_format='%.3f'))

## Generate Recommendations

if len(df) > 0:
    print("\n🎯 Recommendations based on benchmarks:")
    
    # Best for memory efficiency
    best_memory = df.loc[df['params_per_gb'].idxmax()]
    print(f"\n1. Best Memory Efficiency:")
    print(f"   - LoRA rank: {best_memory['lora_r']}")
    print(f"   - {best_memory['params_per_gb']:.1f}M parameters per GB")
    
    # Best for speed
    fastest = df.loc[df['samples_per_second'].idxmax()]
    print(f"\n2. Fastest Training:")
    print(f"   - LoRA rank: {fastest['lora_r']}")
    print(f"   - {fastest['samples_per_second']:.1f} samples/second")
    
    # Best loss (if multiple runs)
    if len(df) > 1:
        best_loss = df.loc[df['final_loss'].idxmin()]
        print(f"\n3. Best Final Loss:")
        print(f"   - LoRA rank: {best_loss['lora_r']}")
        print(f"   - Final loss: {best_loss['final_loss']:.4f}")

## Export Results

# Save processed results
if len(df) > 0:
    # Create reports directory
    Path("reports").mkdir(exist_ok=True)
    
    # Save to CSV
    summary_df.to_csv("reports/benchmark_summary.csv", index=False)
    print("\n💾 Results saved to reports/benchmark_summary.csv")
    
    # Create markdown report
    with open("reports/benchmark_report.md", "w") as f:
        f.write(f"# LoRA Benchmark Report\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## Summary\n")
        f.write(f"- Total configurations tested: {len(df)}\n")
        f.write(f"- Best memory efficiency: r={best_memory['lora_r']} ")
        f.write(f"({best_memory['params_per_gb']:.1f}M params/GB)\n")
        f.write(f"- Fastest configuration: r={fastest['lora_r']} ")
        f.write(f"({fastest['samples_per_second']:.1f} samples/sec)\n")
        f.write(f"\n## Detailed Results\n\n")
        f.write(summary_df.to_markdown(index=False))
    
    print("📄 Markdown report saved to reports/benchmark_report.md")
