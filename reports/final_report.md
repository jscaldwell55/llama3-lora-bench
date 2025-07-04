# LoRA Benchmark Results

## Summary
- Tested 4 configurations on TinyLlama/TinyLlama-1.1B-Chat-v1.0
- LoRA ranks tested: [np.int64(4), np.int64(8), np.int64(16), np.int64(32)]
- GPU: NVIDIA A100-SXM4-40GB

## Key Findings

### 1. Memory Scaling
- Memory usage increases sub-linearly with rank
- r=4: 3.11GB
- r=32: 3.38GB
- Only 9% increase for 8x parameters

### 2. Performance Impact
- Higher ranks show better final loss
- Training time scales linearly with parameters
- Throughput remains relatively stable

### 3. Recommendations
- **Memory-constrained**: Use r=4-8
- **Quality-focused**: Use r=32
- **Balanced**: Use r=16

## Next Steps
1. Test with larger models (3B, 8B)
2. Evaluate on downstream tasks
3. Test different target module combinations
