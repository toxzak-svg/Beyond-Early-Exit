"""Tests for routing and bucketing."""

import pytest
import torch

from utio.routing import bucket_tokens


def test_bucket_tokens_basic():
    """Test basic bucketing functionality."""
    batch_size = 100
    tis = torch.randn(batch_size) * 10 + 15  # Values around 15

    buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32), min_bucket=5)

    assert len(buckets) > 0
    assert order.shape == (batch_size,)
    assert inv.shape == (batch_size,)

    # Check inverse permutation
    restored = order[inv]
    expected = torch.arange(batch_size, device=order.device)
    assert torch.allclose(restored, expected)


def test_bucket_tokens_all_deep():
    """Test when all tokens go to deepest bucket."""
    batch_size = 50
    tis = torch.ones(batch_size) * 100  # All high

    buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32), min_bucket=5)

    # Should have at least one bucket (the deepest)
    assert len(buckets) >= 1
    assert buckets[-1][0] == 32  # Deepest cutoff


def test_bucket_tokens_min_bucket():
    """Test min_bucket merging behavior."""
    batch_size = 20
    tis = torch.randn(batch_size) * 5 + 3  # Mostly low values

    buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32), min_bucket=10)

    # Small buckets should be merged
    for depth, idx in buckets:
        assert idx.numel() >= 10 or depth == 32  # Or merged into deepest


def test_bucket_tokens_empty():
    """Test edge case with empty input."""
    tis = torch.tensor([])

    buckets, order, inv = bucket_tokens(tis, min_bucket=1)
    assert len(buckets) == 0
    assert order.shape == (0,)
    assert inv.shape == (0,)


def test_bucket_tokens_errors():
    """Test error handling."""
    tis = torch.randn(10)

    # Non-monotonic cutoffs
    with pytest.raises(ValueError):
        bucket_tokens(tis, bucket_cutoffs=(16, 6, 32))

    # Wrong input shape
    with pytest.raises(ValueError):
        bucket_tokens(tis.unsqueeze(0))


def test_bucket_tokens_permutation_correctness():
    """Test that permutation preserves all tokens."""
    batch_size = 200
    tis = torch.randn(batch_size)

    buckets, order, inv = bucket_tokens(tis, min_bucket=5)

    # All indices should appear exactly once
    all_indices = torch.cat([b[1] for b in buckets])
    assert all_indices.shape[0] == batch_size
    assert torch.allclose(torch.sort(all_indices)[0], torch.arange(batch_size, device=all_indices.device))
