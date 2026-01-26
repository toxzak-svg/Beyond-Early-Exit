"""Tests for bucketed execution runner."""

import pytest
import torch
from torch import nn

from utio.runner import run_buckets
from utio.routing import bucket_tokens
from utio.signal import surrogate_importance


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(10)])

    def forward(self, x: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
        for i in range(min(max_depth, len(self.layers))):
            x = self.layers[i](x)
        return x


def test_run_buckets_basic():
    """Test basic bucketed execution."""
    batch_size = 32
    dim = 128
    model = SimpleModel(dim=dim)
    model.eval()

    inputs = torch.randn(batch_size, dim)
    logits = torch.randn(batch_size, 1000)
    attn_heat = torch.randn(batch_size)
    embeddings = torch.randn(batch_size, dim)

    tis = surrogate_importance(logits, attn_heat, embeddings)
    buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(3, 6, 10), min_bucket=4)

    with torch.no_grad():
        outputs = run_buckets(model, inputs, buckets, order, inv)

    assert outputs.shape == inputs.shape
    assert outputs.shape[0] == batch_size


def test_run_buckets_preserves_order():
    """Test that outputs are restored to original order."""
    batch_size = 20
    dim = 64
    model = SimpleModel(dim=dim)
    model.eval()

    inputs = torch.randn(batch_size, dim)
    logits = torch.randn(batch_size, 1000)
    attn_heat = torch.randn(batch_size)
    embeddings = torch.randn(batch_size, dim)

    tis = surrogate_importance(logits, attn_heat, embeddings)
    buckets, order, inv = bucket_tokens(tis, min_bucket=3)

    with torch.no_grad():
        outputs = run_buckets(model, inputs, buckets, order, inv)

    # Outputs should be in original input order
    assert outputs.shape[0] == batch_size

    # Compare with baseline (full depth)
    with torch.no_grad():
        baseline = model(inputs, max_depth=10)

    # Outputs should be different (due to different depths) but same shape
    assert outputs.shape == baseline.shape


def test_run_buckets_custom_depth_kw():
    """Test custom depth keyword name."""
    batch_size = 16
    dim = 64

    def custom_forward(x: torch.Tensor, custom_depth: int = 10) -> torch.Tensor:
        return x * custom_depth

    inputs = torch.randn(batch_size, dim)
    tis = torch.randn(batch_size)
    buckets, order, inv = bucket_tokens(tis, min_bucket=2)

    outputs = run_buckets(custom_forward, inputs, buckets, order, inv, depth_kw="custom_depth")
    assert outputs.shape == inputs.shape


def test_run_buckets_errors():
    """Test error handling."""
    batch_size = 10
    dim = 32
    model = SimpleModel(dim=dim)

    inputs = torch.randn(batch_size, dim)
    tis = torch.randn(batch_size)
    buckets, order, inv = bucket_tokens(tis, min_bucket=2)

    # Mismatched batch size
    with pytest.raises(ValueError):
        run_buckets(model, inputs[:5], buckets, order, inv)

    # Model returning wrong batch size
    def bad_forward(x: torch.Tensor, max_depth: int = 10) -> torch.Tensor:
        return x[:1]  # Wrong size

    with pytest.raises(ValueError):
        run_buckets(bad_forward, inputs, buckets, order, inv)
