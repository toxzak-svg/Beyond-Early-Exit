# vLLM Integration Notes for UTIO

This document describes where and how to hook UTIO into vLLM so it can be submitted as a PR.

## 1. Where to hook

- **Model executor forward**: The main integration point is the path that runs the transformer forward for a batch of requests. In vLLM this is typically inside the model executor (e.g. code that runs the stacked transformer layers for the current batch).
- **Ideal hook**: Wrap the function that:
  - Takes hidden states (or equivalent) for the current batch.
  - Runs all layers (or a range of layers) for that batch.
  - Returns the updated hidden states (and possibly logits).

So: **before** that forward, run UTIO routing (compute TIS, bucket, permute); run forward **per bucket** with a depth limit; **after** all buckets, unpermute to original order.

## 2. What you need from vLLM

- **Inputs for TIS** (all at batch dimension `B`):
  - Logits or last-layer logits for the current step (for entropy).
  - Some notion of “attention heat” per token (e.g. mean attention received); if not available, use a constant or a simple proxy (e.g. norm of hidden state).
  - Hidden state or embedding for each token (for cosine outlier vs batch mean).
- **Forward API**:
  - A way to run the same model forward on a **subset** of the batch (the bucket) with a **maximum depth** (layer index). If vLLM does not expose `max_depth`, you have two options:
    - Add an optional `max_depth` (or `num_layers`) argument to the internal forward and stop after that layer; or
    - Run the full stack but only for the subset of batch indices in the bucket (and combine results with other buckets).
- **Output reordering**: After running per-bucket forwards, concatenate/stack outputs and apply the inverse permutation so that output position `i` corresponds to the original request `i`.

## 3. Suggested file changes (conceptual)

- **Config / CLI**  
  - Add a flag, e.g. `--enable-utio`, and optionally `--utio-min-bucket`, `--utio-min-depth`, `--utio-bucket-cutoffs` (e.g. `6,16,32`).
- **Model executor**  
  - In the file that performs the main transformer forward (e.g. one of the model executor or worker files that runs the model):
  - If `enable_utio` is False, call the existing forward as today.
  - If True:
    - Compute TIS from current logits, attention heat, and hidden states.
    - Call a small “bucket + permute” routine (or use the `utio` package’s `bucket_tokens`).
    - For each bucket, run the model forward on the bucket’s indices with the bucket’s depth; collect outputs.
    - Unpermute to original order and pass that result to the rest of the pipeline.
- **Dependency**  
  - Either:
    - Vendor a minimal copy of the UTIO routing logic (signal + bucket + runner), or
    - Add an optional dependency on the `utio` package and call it from vLLM.

## 4. Testing strategy

- **Correctness**: Compare outputs (hidden states or logits) for a fixed batch with and without UTIO (same seed, same inputs). Allow small numerical differences; check that order is correct and values are close.
- **Performance**: Benchmark latency and throughput with UTIO on vs off on representative models (e.g. Llama-2 7B/13B) and batch sizes. Report speedup and any overhead.
- **Quality**: If possible, compare perplexity (or a downstream metric) with UTIO vs baseline to show that the surrogate routing does not hurt quality.

## 5. Fallbacks

- Disable UTIO for batch size below a threshold (e.g. &lt; 16).
- If only one bucket is formed, skip UTIO and run the standard forward.
- Enforce a minimum depth (e.g. 6 layers) for all tokens to limit damage from misrouting.

## 6. Reference implementation

The standalone implementation is in the [Beyond-Early-Exit](https://github.com/yourusername/Beyond-Early-Exit) repo:

- `utio/signal.py`: surrogate importance score.
- `utio/routing.py`: bucket creation and permutation.
- `utio/runner.py`: run model per bucket and unpermute.
- `utio/vllm_integration.py`: wrapper and `patch_vllm_model_executor()` for out-of-tree use.

You can use `utio.vllm_integration.UTIOvLLMWrapper` as a reference for the wrapper interface; adapt it to vLLM’s actual executor API (tensor shapes, kwargs, and where forward is called).
