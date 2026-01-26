"""Tests for surrogate importance signal."""

import pytest
import torch

from utio.signal import surrogate_importance


def test_surrogate_importance_shapes():
    """Test that TIS returns correct shapes."""
    batch_size = 64
    vocab_size = 50000
    embed_dim = 4096
    num_heads = 32

    logits = torch.randn(batch_size, vocab_size)
    attn_heat = torch.randn(batch_size, num_heads)
    embeddings = torch.randn(batch_size, embed_dim)

    tis = surrogate_importance(logits, attn_heat, embeddings)
    assert tis.shape == (batch_size,)
    assert torch.all(tis >= 0)


def test_surrogate_importance_1d_attn():
    """Test TIS with 1D attention heat."""
    batch_size = 32
    logits = torch.randn(batch_size, 1000)
    attn_heat = torch.randn(batch_size)  # 1D
    embeddings = torch.randn(batch_size, 512)

    tis = surrogate_importance(logits, attn_heat, embeddings)
    assert tis.shape == (batch_size,)


def test_surrogate_importance_custom_weights():
    """Test TIS with custom weights."""
    batch_size = 16
    logits = torch.randn(batch_size, 1000)
    attn_heat = torch.randn(batch_size, 8)
    embeddings = torch.randn(batch_size, 256)

    tis = surrogate_importance(logits, attn_heat, embeddings, weights=(0.7, 0.2, 0.1))
    assert tis.shape == (batch_size,)


def test_surrogate_importance_batch_mean():
    """Test TIS with provided batch mean."""
    batch_size = 32
    logits = torch.randn(batch_size, 1000)
    attn_heat = torch.randn(batch_size)
    embeddings = torch.randn(batch_size, 512)
    batch_mean = torch.randn(512)

    tis = surrogate_importance(logits, attn_heat, embeddings, batch_mean=batch_mean)
    assert tis.shape == (batch_size,)


def test_surrogate_importance_errors():
    """Test TIS error handling."""
    logits = torch.randn(10, 1000)
    attn_heat = torch.randn(10)
    embeddings = torch.randn(10, 512)

    # Wrong logits shape
    with pytest.raises(ValueError):
        surrogate_importance(logits.unsqueeze(0), attn_heat, embeddings)

    # Wrong batch size
    with pytest.raises(ValueError):
        surrogate_importance(logits, attn_heat[:5], embeddings)
