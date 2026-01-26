"""Routing utilities for depth bucketing."""

from typing import Iterable, List, Sequence, Tuple

import torch
from torch import Tensor


def bucket_tokens(
    tis: Tensor,
    bucket_cutoffs: Sequence[int] = (6, 16, 32),
    min_bucket: int = 8,
) -> Tuple[List[Tuple[int, Tensor]], Tensor, Tensor]:
    """
    Group tokens into depth buckets based on importance scores.

    Args:
        tis: [B] token importance scores.
        bucket_cutoffs: monotonically increasing depth cutoffs.
        min_bucket: minimum tokens to keep a bucket; else merged deeper.

    Returns:
        buckets: list of (depth, indices)
        order: permutation applied to inputs (concatenated bucket indices)
        inv: inverse permutation to restore original order
    """
    if tis.dim() != 1:
        raise ValueError("tis must be [B]")
    if any(bucket_cutoffs[i] > bucket_cutoffs[i + 1] for i in range(len(bucket_cutoffs) - 1)):
        raise ValueError("bucket_cutoffs must be non-decreasing")

    scores = tis.clone()
    consumed = torch.zeros_like(scores, dtype=torch.bool)
    buckets: List[Tuple[int, Tensor]] = []

    for depth in bucket_cutoffs:
        idx = (~consumed & (scores <= depth)).nonzero(as_tuple=False).flatten()
        if idx.numel() >= min_bucket:
            buckets.append((int(depth), idx))
            consumed[idx] = True

    # Remaining tokens go to the deepest bucket
    remaining = (~consumed).nonzero(as_tuple=False).flatten()
    if remaining.numel():
        buckets.append((int(bucket_cutoffs[-1]), remaining))

    if buckets:
        order = torch.cat([b[1] for b in buckets])
    else:
        order = torch.arange(scores.shape[0], device=scores.device)

    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device)

    return buckets, order, inv
