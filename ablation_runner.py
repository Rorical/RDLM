import argparse
import csv
import datetime
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple


# (variant_name, hydra_exp, hydra_overrides)
# Defaults target LM1B checkpoints. Override with --variants if you want text8 configs.
# ODE ablations now keep only nucleus variants.
DEFAULT_VARIANTS: List[Tuple[str, str, List[str]]] = [
    ("sde", "sample_lm1b_sde", []),
    ("pfm_nucleus", "sample_lm1b_pfm_nucleus", []),
    ("pfm_nucleus_256", "sample_lm1b_pfm_nucleus", ["sampling.steps=256"]),
    # Single-sample MC nucleus: each ODE step samples 1 token direction
    ("pfm_nucleus_mc1", "sample_lm1b_pfm_nucleus", [
        "sampling.pfm.mc_samples=1",
        "sampling.pfm.stochastic=true",
        "sampling.steps=256"
    ]),
    # Posterior decode at t=1-eps (no final state argmax)
    ("pfm_nucleus_post", "sample_lm1b_pfm_nucleus_post", []),
    # Posterior decode + single-sample MC nucleus
    ("pfm_nucleus_post_mc1", "sample_lm1b_pfm_nucleus_post", [
        "sampling.pfm.mc_samples=1",
        "sampling.pfm.stochastic=true",
        "sampling.steps=256"
    ]),
]


def parse_metrics(log_path: str) -> Dict[str, float]:
    """Parse metric lines from run.log produced by run_sample/main."""
    if not os.path.exists(log_path):
        return {}

    pattern = re.compile(
        r"Step\s+(?P<step>\d+)\.\s+(?P<name>[A-Za-z0-9_]+)\s*:\s*(?P<value>[-+eE0-9\.]+)"
    )
    metrics: Dict[str, float] = {}
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                try:
                    metrics[m.group("name")] = float(m.group("value"))
                except ValueError:
                    continue
    return metrics


def run_variant(
    variant_name: str,
    exp_name: str,
    overrides: List[str],
    checkpoint: str,
    ngpus: int,
    base_dir: str,
) -> Tuple[str, str, Dict[str, float]]:
    """Run one ablation variant via main.py in sample mode."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(base_dir, f"{variant_name}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "main.py",
        "run_mode=sample",
        f"exp={exp_name}",
        f"model_path={checkpoint}",
        f"ngpus={ngpus}",
        f"hydra.run.dir={run_dir}",
        "hydra.output_subdir=null",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    cmd.extend(overrides)
    print(f"[{variant_name}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode != 0:
        print(f"[{variant_name}] Run failed (exit {result.returncode}). Output:\n{result.stdout}")
        return variant_name, run_dir, {}

    # Parse metrics
    log_path = os.path.join(run_dir, "run.log")
    metrics = parse_metrics(log_path)
    if not metrics:
        print(f"[{variant_name}] No metrics found in {log_path}")
    else:
        print(f"[{variant_name}] Metrics: {metrics}")
    return variant_name, run_dir, metrics


def print_table(results: List[Tuple[str, Dict[str, float]]]) -> None:
    all_metrics = sorted({m for _, metrics in results for m in metrics.keys()})
    header = ["variant"] + all_metrics
    rows = []
    for name, metrics in results:
        row = [name] + [f"{metrics.get(m, float('nan')):.3f}" if m in metrics else "-" for m in all_metrics]
        rows.append(row)

    col_widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    def fmt_row(vals):
        return " | ".join(v.ljust(col_widths[i]) for i, v in enumerate(vals))

    print("\nSummary table:")
    print(fmt_row(header))
    print("-+-".join("-" * w for w in col_widths))
    for r in rows:
        print(fmt_row(r))


def save_csv_json(
    base_dir: str, results: List[Tuple[str, Dict[str, float]]]
) -> str:
    all_metrics = sorted({m for _, metrics in results for m in metrics.keys()})
    csv_path = os.path.join(base_dir, "ablation_summary.csv")
    json_path = os.path.join(base_dir, "ablation_summary.json")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["variant"] + all_metrics)
        for name, metrics in results:
            writer.writerow([name] + [metrics.get(m, "") for m in all_metrics])

    summary_dict = {name: metrics for name, metrics in results}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    print(f"\nSaved summary to {csv_path} and {json_path}")
    return csv_path


def save_charts(base_dir: str, results: List[Tuple[str, Dict[str, float]]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Matplotlib not available, skipping charts. ({e})")
        return

    all_metrics = sorted({m for _, metrics in results for m in metrics.keys()})
    variants = [name for name, _ in results]

    for metric in all_metrics:
        values = [metrics.get(metric, float("nan")) for _, metrics in results]
        plt.figure(figsize=(6, 4))
        plt.bar(variants, values)
        plt.ylabel(metric)
        plt.title(f"{metric} by variant")
        plt.xticks(rotation=20)
        plt.tight_layout()
        out_path = os.path.join(base_dir, f"chart_{metric}.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"Saved chart: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Automate sampling ablation runs and aggregate results.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pth file.")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs to use per run.")
    parser.add_argument(
        "--base_dir", default="ablation_runs", help="Directory to store run outputs and summaries."
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=[
            ":".join(
                [name, exp] + ([",".join(overrides)] if overrides else [])
            )
            for name, exp, overrides in DEFAULT_VARIANTS
        ],
        help=(
            "Variants as name:exp[:override1,override2,...] "
            "(overrides are Hydra key=value strings)."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.base_dir, exist_ok=True)

    variants = []
    for v in args.variants:
        parts = v.split(":")
        if len(parts) < 2:
            parser.error(f"Variant '{v}' must be in name:exp format.")
        name, exp = parts[0], parts[1]
        overrides = []
        if len(parts) > 2:
            override_str = ":".join(parts[2:])  # allow colons in subsequent overrides
            overrides = [o for o in override_str.split(",") if o]
        variants.append((name, exp, overrides))

    run_results: List[Tuple[str, Dict[str, float]]] = []
    for name, exp, overrides in variants:
        variant, run_dir, metrics = run_variant(name, exp, overrides, args.checkpoint, args.ngpus, args.base_dir)
        run_results.append((variant, metrics))

    print_table(run_results)
    save_csv_json(args.base_dir, run_results)
    save_charts(args.base_dir, run_results)


if __name__ == "__main__":
    main()
