# UTIO Benchmark Suite

This directory contains comprehensive benchmarks for UTIO performance on real hardware.

## Quick Start

```bash
# Run default benchmark suite
python benchmarks/benchmark_suite.py

# Custom configuration
python benchmarks/benchmark_suite.py \
    --model llama-7b \
    --batch-sizes 32 64 128 256 \
    --prompt-lengths 512 2048 8192 \
    --output-lengths 64 256 \
    --output benchmarks/results.json
```

## Benchmark Models

Currently supported models (mock implementations):
- `llama-7b`: 32 layers, 4096 dim
- `llama-13b`: 40 layers, 5120 dim
- `llama-70b`: 80 layers, 8192 dim
- `mistral-7b`: 32 layers, 4096 dim

## Metrics Collected

- **Latency**: P50/P95 latency per request
- **Throughput**: Tokens per second
- **Speedup**: UTIO vs baseline
- **Overhead**: TIS computation and bucketing overhead
- **Quality**: Perplexity/accuracy deltas (when available)

## Output Format

Results are saved as JSON with:
- Individual run results
- Summary statistics
- Configuration used

## Integration with Real Models

To benchmark with actual models (e.g., via vLLM):

1. Install vLLM: `pip install vllm`
2. Use the vLLM integration wrapper (see `utio/vllm_integration.py`)
3. Modify `benchmark_suite.py` to use real model instances

## Hardware Requirements

- CUDA-capable GPU (recommended: A100, H100, 4090, L40S)
- PyTorch with CUDA support
- Sufficient VRAM for model + batch sizes

## Example Results

```
Average Speedup: 1.25x
Average Latency Reduction: 20.0%
Average TIS Overhead: 0.15ms
Average Bucketing Overhead: 0.05ms
```
