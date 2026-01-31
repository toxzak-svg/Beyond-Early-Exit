"""
Example patch logic for vLLM: where to call UTIO.

This is NOT a literal patch file. It shows the pattern to implement
inside vLLM's codebase when adding UTIO support.

Assume we are inside the model executor's forward path, with:
  - hidden_states: [B, seq_len, D]
  - We have access to logits (or can get them) and attention cache.

Pseudocode:
"""

# 1) At the start of the forward path (e.g. in model_executor or worker):

# if not enable_utio:
#     return original_forward(hidden_states, ...)

# 2) Compute TIS (use utio.signal.surrogate_importance)
#    Need: logits [B, V], attn_heat [B] or [B,H], embeddings [B, D]
# tis = surrogate_importance(logits, attn_heat, embeddings)

# 3) Bucket tokens (use utio.routing.bucket_tokens)
# buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6,16,32), min_bucket=8)

# 4) If only one bucket, skip UTIO
# if len(buckets) <= 1:
#     return original_forward(hidden_states, ...)

# 5) Reorder hidden_states by order
# hidden_reordered = hidden_states[order]  # or index_select along batch dim

# 6) Run forward per bucket with max_depth
# outputs = []
# for depth, bucket_indices in buckets:
#     sub = hidden_reordered[bucket_indices]  # subset of reordered batch
#     out = original_forward(sub, max_depth=depth, ...)  # requires max_depth support
#     outputs.append((bucket_indices, out))

# 7) Unpermute: stack outputs in original order using inv
# out_stacked = stack_and_unpermute(outputs, inv)
# return out_stacked

# 8) Original forward must accept optional max_depth and run only that many layers.
#    If vLLM doesn't support it, add a loop that runs layers 0..max_depth-1 and returns.
