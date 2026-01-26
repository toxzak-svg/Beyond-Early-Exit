"""Benchmark harness for UTIO overhead and effectiveness."""

import time
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn

from .routing import bucket_tokens
from .runner import run_buckets
from .signal import surrogate_importance


class MockModel(nn.Module):
    """Mock model that simulates variable-depth computation."""

    def __init__(self, num_layers: int = 32, dim: int = 4096):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: Tensor, max_depth: Optional[int] = None) -> Tensor:
        """Forward pass with optional depth limit."""
        depth = max_depth if max_depth is not None else self.num_layers
        for i in range(min(depth, self.num_layers)):
            x = self.layers[i](x)
        return x


def benchmark_tis_overhead(
    batch_size: int = 1024,
    vocab_size: int = 50000,
    embed_dim: int = 4096,
    num_heads: int = 32,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Measure TIS computation overhead."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logits = torch.randn(batch_size, vocab_size, device=device)
    attn_heat = torch.randn(batch_size, num_heads, device=device)
    embeddings = torch.randn(batch_size, embed_dim, device=device)

    # Warmup
    for _ in range(10):
        _ = surrogate_importance(logits, attn_heat, embeddings)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = surrogate_importance(logits, attn_heat, embeddings)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    overhead_pct = (avg_ms / 1.0) * 100  # Assuming 1ms baseline forward pass

    return {
        "avg_time_ms": avg_ms,
        "overhead_pct": overhead_pct,
        "target_met": overhead_pct < 0.2,
    }


def benchmark_bucketing(
    batch_size: int = 1024,
    num_runs: int = 100,
) -> Dict[str, float]:
    """Measure bucketing overhead."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tis = torch.randn(batch_size, device=device)

    # Warmup
    for _ in range(10):
        _, _, _ = bucket_tokens(tis, min_bucket=8)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _, _, _ = bucket_tokens(tis, min_bucket=8)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / num_runs) * 1000
    return {"avg_time_ms": avg_ms}


def benchmark_end_to_end(
    batch_size: int = 256,
    embed_dim: int = 4096,
    num_layers: int = 32,
    num_runs: int = 50,
) -> Dict[str, float]:
    """Compare baseline vs UTIO execution time."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MockModel(num_layers=num_layers, dim=embed_dim).to(device)
    model.eval()

    inputs = torch.randn(batch_size, embed_dim, device=device)
    logits = torch.randn(batch_size, 50000, device=device)
    attn_heat = torch.randn(batch_size, 32, device=device)
    embeddings = torch.randn(batch_size, embed_dim, device=device)

    # Baseline: full depth
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(inputs, max_depth=num_layers)
        if device.type == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / num_runs

    # UTIO: bucketed execution
    with torch.no_grad():
        tis = surrogate_importance(logits, attn_heat, embeddings)
        buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32), min_bucket=8)

        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = run_buckets(model, inputs, buckets, order, inv)
        if device.type == "cuda":
            torch.cuda.synchronize()
        utio_time = (time.perf_counter() - start) / num_runs

    speedup = baseline_time / utio_time if utio_time > 0 else 0.0

    return {
        "baseline_ms": baseline_time * 1000,
        "utio_ms": utio_time * 1000,
        "speedup": speedup,
        "latency_reduction_pct": (1.0 - utio_time / baseline_time) * 100 if baseline_time > 0 else 0.0,
    }


if __name__ == "__main__":
    print("Running UTIO benchmarks...")
    print("\n1. TIS Overhead:")
    tis_results = benchmark_tis_overhead()
    for k, v in tis_results.items():
        print(f"   {k}: {v:.4f}")

    print("\n2. Bucketing Overhead:")
    bucket_results = benchmark_bucketing()
    for k, v in bucket_results.items():
        print(f"   {k}: {v:.4f}")

    print("\n3. End-to-End (Baseline vs UTIO):")
    e2e_results = benchmark_end_to_end()
    for k, v in e2e_results.items():
        print(f"   {k}: {v:.4f}")
