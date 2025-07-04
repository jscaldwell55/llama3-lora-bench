# llama3-lora-bench

A comprehensive, reproducible benchmark suite for LoRA configurations on Llama 3 models (1B, 3B, 8B, 70B).

## 🎯 Mission

Fill the gap in systematic LoRA hyperparameter benchmarking for Llama 3 models, saving developers time and compute resources.

## 📊 What We Benchmark

- **Models**: Llama 3.2 (1B, 3B), Llama 3 (8B, 70B)
- **Parameters**: r, alpha, dropout, target_modules
- **Metrics**: Training loss, validation perplexity, memory usage, tokens/sec
- **Datasets**: Alpaca, OpenAssistant, custom

## 🚀 Quick Start

```bash
# Run minimal benchmark
python scripts/benchmark.py --config configs/quick_start/best_quality.yaml

# Run full benchmark  
python scripts/benchmark.py --config configs/comprehensive/llama3_8b.yaml

📈 Results
Coming soon! Follow for updates.
🤝 Contributing
Have a configuration you want tested? Open an issue!
📝 Citation
If you use these benchmarks, please cite:
@misc{llama3-lora-bench,
  author = {Your Name},
  title = {Llama 3 LoRA Benchmarks},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/yourusername/llama3-lora-bench}
}