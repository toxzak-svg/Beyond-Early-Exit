"""Comprehensive benchmark suite for UTIO on real hardware."""

import argparse
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

# Import UTIO
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utio import bucket_tokens, run_buckets, surrogate_importance


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs."""

    model_name: str
    batch_sizes: List[int]
    prompt_lengths: List[int]
    output_lengths: List[int]
    bucket_cutoffs: Tuple[int, ...] = (6, 16, 32)
    min_bucket: int = 8
    num_warmup: int = 10
    num_runs: int = 50
    device: str = "cuda"


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config: Dict
    baseline_latency_ms: float
    utio_latency_ms: float
    speedup: float
    latency_reduction_pct: float
    tis_overhead_ms: float
    bucketing_overhead_ms: float
    quality_metric: Optional[float] = None
    quality_delta: Optional[float] = None


class MockLLM(nn.Module):
    """Mock LLM for benchmarking (simulates real model behavior)."""

    def __init__(self, num_layers: int = 32, dim: int = 4096, vocab_size: int = 50000):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.vocab_size = vocab_size

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, dim)

        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.ModuleDict({
                "attn": nn.MultiheadAttention(dim, num_heads=32, batch_first=True),
                "ffn": nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim),
                ),
                "ln1": nn.LayerNorm(dim),
                "ln2": nn.LayerNorm(dim),
            })
            self.layers.append(layer)

        # Output head
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: Tensor, max_depth: Optional[int] = None, return_logits: bool = False) -> Tensor:
        """Forward pass with optional depth limit."""
        x = self.embedding(input_ids)
        depth = max_depth if max_depth is not None else self.num_layers

        for i in range(min(depth, self.num_layers)):
            layer = self.layers[i]
            # Self-attention
            residual = x
            x = layer["ln1"](x)
            x_attn, _ = layer["attn"](x, x, x)
            x = residual + x_attn

            # FFN
            residual = x
            x = layer["ln2"](x)
            x = layer["ffn"](x)
            x = residual + x

        x = self.ln_out(x)

        if return_logits:
            return self.head(x)
        return x


def measure_tis_overhead(
    batch_size: int,
    vocab_size: int,
    embed_dim: int,
    num_heads: int,
    device: str,
    num_runs: int = 100,
) -> float:
    """Measure TIS computation overhead in milliseconds."""
    logits = torch.randn(batch_size, vocab_size, device=device)
    attn_heat = torch.randn(batch_size, num_heads, device=device)
    embeddings = torch.randn(batch_size, embed_dim, device=device)

    # Warmup
    for _ in range(10):
        _ = surrogate_importance(logits, attn_heat, embeddings)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _ = surrogate_importance(logits, attn_heat, embeddings)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return elapsed * 1000  # Convert to ms


def measure_bucketing_overhead(batch_size: int, device: str, num_runs: int = 100) -> float:
    """Measure bucketing overhead in milliseconds."""
    tis = torch.randn(batch_size, device=device)

    # Warmup
    for _ in range(10):
        _, _, _ = bucket_tokens(tis, min_bucket=8)

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(num_runs):
        _, _, _ = bucket_tokens(tis, min_bucket=8)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / num_runs

    return elapsed * 1000


def run_single_benchmark(
    model: nn.Module,
    batch_size: int,
    prompt_length: int,
    output_length: int,
    bucket_cutoffs: Tuple[int, ...],
    min_bucket: int,
    device: Union[str, torch.device],
    num_warmup: int,
    num_runs: int,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""
    model.eval()

    # Generate inputs (device may be str or torch.device)
    device_obj = device if isinstance(device, torch.device) else torch.device(device)
    input_ids = torch.randint(0, model.vocab_size, (batch_size, prompt_length), device=device_obj)
    target_length = prompt_length + output_length

    # Simulate features needed for TIS
    with torch.no_grad():
        # Get embeddings and logits from first forward
        hidden = model.embedding(input_ids[:, -1])  # Last token embeddings
        logits_full = model(input_ids, return_logits=True)
        logits = logits_full[:, -1, :]  # Last token logits

        # Simulate attention heat (would come from cached attention in real vLLM)
        attn_heat = torch.randn(batch_size, 32, device=device_obj)

    # Baseline: Full depth forward
    with torch.no_grad():
        if device_obj.type == "cuda":
            torch.cuda.synchronize()

        # Warmup
        for _ in range(num_warmup):
            _ = model(input_ids, max_depth=model.num_layers)

        # Measure
        start = time.perf_counter()
        for _ in range(num_runs):
            _ = model(input_ids, max_depth=model.num_layers)
        if device_obj.type == "cuda":
            torch.cuda.synchronize()
        baseline_time = (time.perf_counter() - start) / num_runs

    # UTIO: Bucketed forward
    with torch.no_grad():
        # Compute TIS
        tis = surrogate_importance(logits, attn_heat, hidden)
        buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=bucket_cutoffs, min_bucket=min_bucket)

        # If only one bucket, skip UTIO
        if len(buckets) <= 1:
            utio_time = baseline_time
        else:
            # Warmup
            for _ in range(num_warmup):
                _ = run_buckets(model, input_ids, buckets, order, inv, depth_kw="max_depth")

            # Measure
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(num_runs):
                _ = run_buckets(model, input_ids, buckets, order, inv, depth_kw="max_depth")
            if device_obj.type == "cuda":
                torch.cuda.synchronize()
            utio_time = (time.perf_counter() - start) / num_runs

    # Measure overheads
    device_str = device_obj.type if hasattr(device_obj, "type") else str(device_obj)
    tis_overhead = measure_tis_overhead(batch_size, model.vocab_size, model.dim, 32, device_str)
    bucketing_overhead = measure_bucketing_overhead(batch_size, device_str)

    speedup = baseline_time / utio_time if utio_time > 0 else 1.0
    latency_reduction = (1.0 - utio_time / baseline_time) * 100 if baseline_time > 0 else 0.0

    return BenchmarkResult(
        config={
            "batch_size": batch_size,
            "prompt_length": prompt_length,
            "output_length": output_length,
            "bucket_cutoffs": bucket_cutoffs,
            "min_bucket": min_bucket,
        },
        baseline_latency_ms=baseline_time * 1000,
        utio_latency_ms=utio_time * 1000,
        speedup=speedup,
        latency_reduction_pct=latency_reduction,
        tis_overhead_ms=tis_overhead,
        bucketing_overhead_ms=bucketing_overhead,
    )


def run_benchmark_suite(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run full benchmark suite."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Running benchmarks on: {device}")

    # Model configurations (simulating different model sizes)
    # Use "llama-7b-tiny" for fast CPU/smoke tests
    model_configs = {
        "llama-7b": {"num_layers": 32, "dim": 4096, "vocab_size": 32000},
        "llama-7b-tiny": {"num_layers": 8, "dim": 256, "vocab_size": 32000},
        "llama-13b": {"num_layers": 40, "dim": 5120, "vocab_size": 32000},
        "llama-70b": {"num_layers": 80, "dim": 8192, "vocab_size": 32000},
        "mistral-7b": {"num_layers": 32, "dim": 4096, "vocab_size": 32000},
    }

    if config.model_name not in model_configs:
        raise ValueError(f"Unknown model: {config.model_name}. Available: {list(model_configs.keys())}")

    model_cfg = model_configs[config.model_name]
    model = MockLLM(**model_cfg).to(device)

    results = []

    total_runs = len(config.batch_sizes) * len(config.prompt_lengths) * len(config.output_lengths)
    current_run = 0

    for batch_size in config.batch_sizes:
        for prompt_len in config.prompt_lengths:
            for output_len in config.output_lengths:
                current_run += 1
                print(f"\n[{current_run}/{total_runs}] Batch={batch_size}, Prompt={prompt_len}, Output={output_len}")

                result = run_single_benchmark(
                    model,
                    batch_size,
                    prompt_len,
                    output_len,
                    config.bucket_cutoffs,
                    config.min_bucket,
                    device,
                    config.num_warmup,
                    config.num_runs,
                )
                results.append(result)

                print(f"  Baseline: {result.baseline_latency_ms:.2f}ms")
                print(f"  UTIO: {result.utio_latency_ms:.2f}ms")
                print(f"  Speedup: {result.speedup:.2f}x ({result.latency_reduction_pct:.1f}% reduction)")

    return results


def save_results(results: List[BenchmarkResult], output_path: Path):
    """Save benchmark results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "results": [asdict(r) for r in results],
        "summary": {
            "avg_speedup": sum(r.speedup for r in results) / len(results) if results else 0,
            "avg_latency_reduction": sum(r.latency_reduction_pct for r in results) / len(results) if results else 0,
            "avg_tis_overhead_ms": sum(r.tis_overhead_ms for r in results) / len(results) if results else 0,
            "avg_bucketing_overhead_ms": sum(r.bucketing_overhead_ms for r in results) / len(results) if results else 0,
        },
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main():
    """Main benchmark entry point."""
    parser = argparse.ArgumentParser(description="UTIO Benchmark Suite")
    parser.add_argument("--model", type=str, default="llama-7b", help="Model to benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 64, 128], help="Batch sizes to test")
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[512, 2048], help="Prompt lengths to test")
    parser.add_argument("--output-lengths", type=int, nargs="+", default=[64, 256], help="Output lengths to test")
    parser.add_argument("--output", type=str, default="benchmarks/results.json", help="Output JSON path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    parser.add_argument("--num-runs", type=int, default=50, help="Number of runs per config")
    parser.add_argument("--num-warmup", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--fast", action="store_true", help="Fast run: tiny model, small batches, few runs (for CPU/smoke)")

    args = parser.parse_args()

    if args.fast:
        args.model = "llama-7b-tiny"
        args.batch_sizes = [8, 16]
        args.prompt_lengths = [64]
        args.output_lengths = [16]
        args.num_runs = 3
        args.num_warmup = 1
        args.device = "cpu"

    config = BenchmarkConfig(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        prompt_lengths=args.prompt_lengths,
        output_lengths=args.output_lengths,
        num_runs=args.num_runs,
        num_warmup=args.num_warmup,
        device=args.device,
    )

    results = run_benchmark_suite(config)
    save_results(results, Path(args.output))

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    avg_speedup = sum(r.speedup for r in results) / len(results) if results else 0
    avg_reduction = sum(r.latency_reduction_pct for r in results) / len(results) if results else 0
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Average Latency Reduction: {avg_reduction:.1f}%")
    print(f"Average TIS Overhead: {sum(r.tis_overhead_ms for r in results) / len(results):.3f}ms")
    print(f"Average Bucketing Overhead: {sum(r.bucketing_overhead_ms for r in results) / len(results):.3f}ms")


if __name__ == "__main__":
    main()
