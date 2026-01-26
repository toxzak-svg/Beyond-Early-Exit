"""Bucketed execution helper."""

from typing import Callable, Iterable, List, Tuple

import torch
from torch import Tensor


def run_buckets(
    model_forward: Callable[..., Tensor],
    inputs: Tensor,
    buckets: Iterable[Tuple[int, Tensor]],
    order: Tensor,
    inv: Tensor,
    depth_kw: str = "max_depth",
    **forward_kwargs,
) -> Tensor:
    """
    Execute model forward passes per bucket, then restore order.

    Args:
        model_forward: callable accepting (batch, {depth_kw}=depth, **kwargs).
        inputs: [B, ...] input batch.
        buckets: iterable of (depth, indices) from bucket_tokens.
        order: permutation applied to inputs (concatenated bucket indices).
        inv: inverse permutation to restore original order.
        depth_kw: keyword name to pass the depth limit to the model.
        forward_kwargs: extra args forwarded to model_forward.

    Returns:
        Tensor with the same first-dimension ordering as the original inputs.
    """
    if inputs.shape[0] != order.shape[0] or order.shape[0] != inv.shape[0]:
        raise ValueError("order/inv must align with input batch")

    reordered = inputs.index_select(0, order)
    outputs: List[Tensor] = [None] * reordered.shape[0]  # type: ignore

    for depth, idx in buckets:
        if idx.numel() == 0:
            continue
        sub = reordered.index_select(0, idx)
        out = model_forward(sub, **{depth_kw: depth}, **forward_kwargs)
        if out.shape[0] != idx.numel():
            raise ValueError("model_forward returned mismatched batch size")
        for pos, tensor in zip(idx.tolist(), out):
            outputs[pos] = tensor

    if any(o is None for o in outputs):
        raise ValueError("missing outputs detected; check bucket construction")

    stacked = torch.stack(outputs)  # type: ignore[arg-type]
    return stacked.index_select(0, inv)
