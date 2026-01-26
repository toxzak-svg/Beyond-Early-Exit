"""vLLM integration shim for UTIO routing."""

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .routing import bucket_tokens
from .runner import run_buckets
from .signal import surrogate_importance


class UTIOvLLMWrapper:
    """
    Wrapper for vLLM that adds UTIO routing to model execution.

    This class hooks into vLLM's model executor to add micro-batched routing
    before forward passes. It's designed to be a drop-in replacement that
    can be enabled with a flag.

    Usage:
        # In vLLM's model_executor.py or similar:
        if use_utio:
            wrapper = UTIOvLLMWrapper(model_executor)
            model_executor.forward = wrapper.forward
    """

    def __init__(
        self,
        model_executor,
        bucket_cutoffs: Tuple[int, ...] = (6, 16, 32),
        min_bucket: int = 8,
        tis_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25),
        min_depth: int = 6,
        enable_utio: bool = True,
    ):
        """
        Initialize UTIO wrapper for vLLM.

        Args:
            model_executor: vLLM model executor instance.
            bucket_cutoffs: Depth cutoffs for bucketing.
            min_bucket: Minimum tokens per bucket before merging.
            tis_weights: Weights for (entropy, attention, outlier) in TIS.
            min_depth: Minimum depth to enforce (safety floor).
            enable_utio: If False, bypasses UTIO and uses original forward.
        """
        self.model_executor = model_executor
        self.bucket_cutoffs = bucket_cutoffs
        self.min_bucket = min_bucket
        self.tis_weights = tis_weights
        self.min_depth = min_depth
        self.enable_utio = enable_utio

        # Cache for buckets (can be reused if batch doesn't change)
        self._cached_buckets: Optional[Tuple[List[Tuple[int, Tensor]], Tensor, Tensor]] = None
        self._cached_batch_size: Optional[int] = None

    def _extract_features(self, hidden_states: Tensor, logits: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Extract features needed for TIS computation from vLLM's internal state.

        This is a placeholder - actual implementation depends on vLLM's
        internal structure. You may need to adapt this based on vLLM version.

        Args:
            hidden_states: [B, D] hidden states from model.
            logits: [B, V] optional logits if available.

        Returns:
            (logits, attn_heat, embeddings) for TIS computation.
        """
        batch_size = hidden_states.shape[0]
        device = hidden_states.device
        embed_dim = hidden_states.shape[-1]

        # If logits not provided, use hidden states as proxy
        if logits is None:
            # Create dummy logits from hidden states (vLLM may provide these)
            vocab_size = 50000  # Default, should match model vocab
            logits = torch.randn(batch_size, vocab_size, device=device)

        # Extract attention heat from cached attention if available
        # This is a placeholder - vLLM stores attention differently
        attn_heat = torch.ones(batch_size, device=device)  # Default uniform

        # Use hidden states as embeddings
        embeddings = hidden_states

        return logits, attn_heat, embeddings

    def _should_use_utio(self, batch_size: int, hidden_states: Tensor) -> bool:
        """
        Determine if UTIO should be used for this batch.

        Args:
            batch_size: Current batch size.
            hidden_states: Input hidden states.

        Returns:
            True if UTIO should be applied.
        """
        if not self.enable_utio:
            return False

        # Skip UTIO for very small batches (overhead not worth it)
        if batch_size < self.min_bucket * 2:
            return False

        # Check if we have enough variance to benefit from routing
        # (This is a heuristic - can be improved)
        return True

    def forward(
        self,
        hidden_states: Tensor,
        kv_cache: Optional[Dict] = None,
        input_metadata: Optional[Dict] = None,
        **kwargs,
    ) -> Tensor:
        """
        UTIO-wrapped forward pass for vLLM.

        This method:
        1. Computes TIS scores
        2. Buckets tokens by depth
        3. Executes bucketed forward passes
        4. Restores original order

        Args:
            hidden_states: [B, D] input hidden states.
            kv_cache: Optional KV cache (vLLM format).
            input_metadata: Optional input metadata (vLLM format).
            **kwargs: Additional arguments passed to model forward.

        Returns:
            Output hidden states in original order.
        """
        batch_size = hidden_states.shape[0]

        # Check if we should use UTIO
        if not self._should_use_utio(batch_size, hidden_states):
            # Fallback to original forward
            return self.model_executor.forward(hidden_states, kv_cache=kv_cache, input_metadata=input_metadata, **kwargs)

        # Extract features for TIS
        logits, attn_heat, embeddings = self._extract_features(hidden_states)

        # Compute TIS
        tis = surrogate_importance(
            logits,
            attn_heat,
            embeddings,
            weights=self.tis_weights,
        )

        # Bucket tokens
        buckets, order, inv = bucket_tokens(
            tis,
            bucket_cutoffs=self.bucket_cutoffs,
            min_bucket=self.min_bucket,
        )

        # If only one bucket (all tokens same depth), skip UTIO overhead
        if len(buckets) <= 1:
            return self.model_executor.forward(hidden_states, kv_cache=kv_cache, input_metadata=input_metadata, **kwargs)

        # Define wrapped forward that respects depth limits
        def utio_forward(x: Tensor, max_depth: Optional[int] = None, **fw_kwargs) -> Tensor:
            # Enforce minimum depth
            depth = max(max_depth or self.bucket_cutoffs[-1], self.min_depth) if max_depth else None

            # Call original forward with depth hint if supported
            # Note: vLLM may not support max_depth directly - this needs adaptation
            result = self.model_executor.forward(
                x,
                kv_cache=kv_cache,
                input_metadata=input_metadata,
                max_depth=depth,
                **fw_kwargs,
            )
            return result

        # Execute bucketed passes
        outputs = run_buckets(
            utio_forward,
            hidden_states,
            buckets,
            order,
            inv,
            depth_kw="max_depth",
            **kwargs,
        )

        return outputs


def patch_vllm_model_executor(model_executor, use_utio: bool = True, **utio_kwargs) -> Optional[UTIOvLLMWrapper]:
    """
    Patch vLLM's model executor to use UTIO routing.

    This is a convenience function to add UTIO to an existing vLLM instance.

    Args:
        model_executor: vLLM model executor instance.
        use_utio: Whether to enable UTIO.
        **utio_kwargs: Additional arguments for UTIOvLLMWrapper.

    Returns:
        UTIO wrapper instance if enabled, None otherwise.

    Example:
        from vllm import LLM
        from utio.vllm_integration import patch_vllm_model_executor

        llm = LLM(model="meta-llama/Llama-2-7b-hf")
        patch_vllm_model_executor(llm.llm_engine.model_executor, use_utio=True)
    """
    if not use_utio:
        return None

    wrapper = UTIOvLLMWrapper(model_executor, enable_utio=True, **utio_kwargs)

    # Store original forward
    if not hasattr(model_executor, "_original_forward"):
        model_executor._original_forward = model_executor.forward

    # Replace forward with wrapped version
    model_executor.forward = wrapper.forward

    return wrapper


def create_vllm_utio_config(
    bucket_cutoffs: Tuple[int, ...] = (6, 16, 32),
    min_bucket: int = 8,
    tis_weights: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    min_depth: int = 6,
) -> Dict:
    """
    Create a configuration dict for UTIO in vLLM.

    This can be saved/loaded and passed to patch_vllm_model_executor.

    Args:
        bucket_cutoffs: Depth cutoffs for bucketing.
        min_bucket: Minimum tokens per bucket.
        tis_weights: TIS component weights.
        min_depth: Minimum depth floor.

    Returns:
        Configuration dictionary.
    """
    return {
        "bucket_cutoffs": bucket_cutoffs,
        "min_bucket": min_bucket,
        "tis_weights": tis_weights,
        "min_depth": min_depth,
    }
