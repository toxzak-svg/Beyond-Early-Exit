"""Surrogate importance signal used for UTIO routing."""

from typing import Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def surrogate_importance(
    logits: Tensor,
    attn_heat: Tensor,
    embeddings: Tensor,
    batch_mean: Optional[Tensor] = None,
    weights: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    eps: float = 1e-9,
) -> Tensor:
    """
    Compute a lightweight token importance score.

    Args:
        logits: [B, V] pre-softmax logits.
        attn_heat: [B] or [B, H] cached attention heat scores.
        embeddings: [B, D] token embeddings.
        batch_mean: [D] optional mean embedding; computed if omitted.
        weights: (entropy, attention, outlier) weighting.
        eps: numerical stability constant.

    Returns:
        Tensor [B] of importance scores (higher → deeper compute).
    """
    if logits.dim() != 2:
        raise ValueError("logits must be [B, V]")
    if embeddings.dim() != 2 or embeddings.shape[0] != logits.shape[0]:
        raise ValueError("embeddings must be [B, D] matching logits batch")

    w_entropy, w_attn, w_outlier = weights

    # Entropy term
    probs = F.softmax(logits.float(), dim=-1)
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)

    # Attention heat: mean over heads if provided per-head
    if attn_heat.shape[0] != logits.shape[0]:
        raise ValueError("attn_heat batch size must match logits batch")
    attn_term = attn_heat
    if attn_heat.dim() > 1:
        attn_term = attn_heat.mean(dim=-1)

    # Outlier term via cosine similarity to batch mean
    if batch_mean is None:
        batch_mean = embeddings.mean(dim=0)
    cos = F.cosine_similarity(embeddings.float(), batch_mean.float(), dim=-1)
    outlier = 1.0 - cos

    return w_entropy * entropy + w_attn * attn_term + w_outlier * outlier
