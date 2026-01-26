# UTIO Quick Start Guide

Get UTIO running in 5 minutes.

## Installation

```bash
git clone <your-repo>
cd Beyond-Early-Exit
pip install -e ".[dev]"
```

## Basic Usage

```python
from utio import surrogate_importance, bucket_tokens, run_buckets
import torch

# 1. Compute importance scores
tis = surrogate_importance(logits, attn_heat, embeddings)

# 2. Bucket tokens by depth
buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32))

# 3. Execute bucketed forward passes
outputs = run_buckets(model, inputs, buckets, order, inv)
```

See `example.py` for a complete working example.

## vLLM Integration

```python
from vllm import LLM
from utio.vllm_integration import patch_vllm_model_executor

llm = LLM(model="meta-llama/Llama-2-7b-hf")
patch_vllm_model_executor(llm.llm_engine.model_executor, use_utio=True)

# Now use normally - UTIO is active
outputs = llm.generate(prompts)
```

## Benchmarking

```bash
# Run benchmarks
python benchmarks/benchmark_suite.py --model llama-7b

# Generate reports
python benchmarks/compare_results.py --results benchmarks/results.json
```

## Testing

```bash
pytest tests/
```

## Next Steps

1. **Run the example**: `python example.py`
2. **Run benchmarks**: `python benchmarks/benchmark_suite.py`
3. **Integrate with vLLM**: See `examples/vllm_example.py`
4. **Read the paper**: See `README.md` for the full architecture

## Troubleshooting

- **CUDA errors**: Ensure PyTorch has CUDA support
- **Import errors**: Run `pip install -e .` to install in development mode
- **vLLM integration**: Requires vLLM installed separately
