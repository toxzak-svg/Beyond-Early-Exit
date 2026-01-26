# UTIO Examples

This directory contains example usage of UTIO in different scenarios.

## Basic Usage

See `../example.py` for a simple standalone example.

## vLLM Integration

See `vllm_example.py` for how to integrate UTIO into vLLM.

**Requirements:**
```bash
pip install vllm
```

**Usage:**
```python
from vllm import LLM
from utio.vllm_integration import patch_vllm_model_executor

llm = LLM(model="meta-llama/Llama-2-7b-hf")
patch_vllm_model_executor(llm.llm_engine.model_executor, use_utio=True)

# Now use llm.generate() as normal - UTIO routing is active
```

## Custom Integration

To integrate UTIO into your own inference engine:

1. Compute TIS scores using `surrogate_importance()`
2. Bucket tokens using `bucket_tokens()`
3. Execute bucketed forward passes using `run_buckets()`

See the main `example.py` for a complete working example.
