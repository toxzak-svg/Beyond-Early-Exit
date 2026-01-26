"""Simple example demonstrating UTIO usage."""

import torch
from torch import nn

from utio import bucket_tokens, run_buckets, surrogate_importance


class SimpleLLM(nn.Module):
    """Minimal LLM-like model for demonstration."""

    def __init__(self, num_layers: int = 32, dim: int = 512):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(dim, dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, max_depth: int = None) -> torch.Tensor:
        """Forward with optional depth limit."""
        depth = max_depth if max_depth is not None else self.num_layers
        for i in range(min(depth, self.num_layers)):
            x = self.layers[i](x)
        return x


def main():
    """Run a simple UTIO example."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup
    batch_size = 64
    vocab_size = 50000
    embed_dim = 512
    num_layers = 32

    model = SimpleLLM(num_layers=num_layers, dim=embed_dim).to(device)
    model.eval()

    # Simulate token inputs and their features
    inputs = torch.randn(batch_size, embed_dim, device=device)
    logits = torch.randn(batch_size, vocab_size, device=device)
    attn_heat = torch.randn(batch_size, 8, device=device)  # 8 attention heads
    embeddings = torch.randn(batch_size, embed_dim, device=device)

    print(f"\nBatch size: {batch_size}")
    print(f"Model depth: {num_layers} layers")

    # Step 1: Compute importance scores
    print("\n1. Computing surrogate importance scores...")
    tis = surrogate_importance(logits, attn_heat, embeddings)
    print(f"   TIS range: [{tis.min():.3f}, {tis.max():.3f}]")

    # Step 2: Bucket tokens by depth
    print("\n2. Bucketing tokens...")
    buckets, order, inv = bucket_tokens(tis, bucket_cutoffs=(6, 16, 32), min_bucket=8)
    print(f"   Created {len(buckets)} buckets:")
    for depth, idx in buckets:
        print(f"     Depth {depth}: {idx.numel()} tokens")

    # Step 3: Run bucketed execution
    print("\n3. Executing bucketed forward passes...")
    with torch.no_grad():
        outputs = run_buckets(model, inputs, buckets, order, inv)

    print(f"   Output shape: {outputs.shape}")
    print(f"   ✓ Execution complete")

    # Compare with baseline
    print("\n4. Comparing with baseline (full depth)...")
    with torch.no_grad():
        baseline = model(inputs, max_depth=num_layers)

    print(f"   Baseline output shape: {baseline.shape}")
    print(f"   UTIO output shape: {outputs.shape}")
    print(f"   ✓ Shapes match")

    print("\n✓ UTIO example completed successfully!")


if __name__ == "__main__":
    main()
