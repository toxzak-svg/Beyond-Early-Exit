"""Compare benchmark results and generate reports."""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


def load_results(json_path: Path) -> Dict:
    """Load benchmark results from JSON."""
    with open(json_path, "r") as f:
        return json.load(f)


def create_summary_table(results: Dict) -> pd.DataFrame:
    """Create a summary table from results."""
    rows = []
    for r in results["results"]:
        rows.append({
            "Batch Size": r["config"]["batch_size"],
            "Prompt Length": r["config"]["prompt_length"],
            "Output Length": r["config"]["output_length"],
            "Baseline (ms)": f"{r['baseline_latency_ms']:.2f}",
            "UTIO (ms)": f"{r['utio_latency_ms']:.2f}",
            "Speedup": f"{r['speedup']:.2f}x",
            "Reduction %": f"{r['latency_reduction_pct']:.1f}",
            "TIS Overhead (ms)": f"{r['tis_overhead_ms']:.3f}",
        })

    return pd.DataFrame(rows)


def plot_speedup_by_batch_size(results: Dict, output_path: Path):
    """Plot speedup vs batch size."""
    df = pd.DataFrame(results["results"])

    # Group by batch size
    grouped = df.groupby(df["config"].apply(lambda x: x["batch_size"]))
    avg_speedup = grouped["speedup"].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(avg_speedup.index, avg_speedup.values, marker="o", linewidth=2, markersize=8)
    plt.xlabel("Batch Size", fontsize=12)
    plt.ylabel("Average Speedup (x)", fontsize=12)
    plt.title("UTIO Speedup by Batch Size", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved to: {output_path}")


def generate_report(results_path: Path, output_dir: Path):
    """Generate comprehensive benchmark report."""
    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(results_path)

    # Summary table
    df = create_summary_table(results)
    table_path = output_dir / "summary_table.csv"
    df.to_csv(table_path, index=False)
    print(f"Summary table saved to: {table_path}")

    # Plot
    plot_path = output_dir / "speedup_by_batch.png"
    plot_speedup_by_batch_size(results, plot_path)

    # Text report
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("UTIO BENCHMARK REPORT\n")
        f.write("=" * 60 + "\n\n")

        summary = results["summary"]
        f.write(f"Average Speedup: {summary['avg_speedup']:.2f}x\n")
        f.write(f"Average Latency Reduction: {summary['avg_latency_reduction']:.1f}%\n")
        f.write(f"Average TIS Overhead: {summary['avg_tis_overhead_ms']:.3f}ms\n")
        f.write(f"Average Bucketing Overhead: {summary['avg_bucketing_overhead_ms']:.3f}ms\n\n")

        f.write("=" * 60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(df.to_string(index=False))

    print(f"Report saved to: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare and report benchmark results")
    parser.add_argument("--results", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output-dir", type=str, default="benchmarks/reports", help="Output directory")

    args = parser.parse_args()

    generate_report(Path(args.results), Path(args.output_dir))


if __name__ == "__main__":
    main()
