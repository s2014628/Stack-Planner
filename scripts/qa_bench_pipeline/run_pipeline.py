"""
QA Bench Experience Data Pipeline

End-to-end pipeline for generating experience data from QA benchmarks:
1. Load QA samples from specified datasets
2. Run benchmark through central agent + search agent
3. Evaluate results with per-dataset metrics
4. Generate experience summaries (factual + SOP)
5. Build final experience data (classified by experience type)

Experience Types:
- 事实性经验 (Factual Experience): TriviaQA, PopQA
  - Knowledge retrieval, search strategy, fact verification patterns
- SOP系统层经验 (SOP System Experience): GPQA, GSM8K, MATH
  - Step-by-step reasoning workflows, tool usage SOPs, error recovery

Usage:
    python -m scripts.qa_bench_pipeline.run_pipeline \\
        --datasets triviaqa popqa gpqa gsm8k math \\
        --max-samples 50 --num-runs 5
"""

import asyncio
import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.qa_bench_pipeline.data_loader import (
    load_qa_samples,
    save_samples,
    ALL_DATASETS,
    FACTUAL_DATASETS,
    SOP_DATASETS,
)
from scripts.qa_bench_pipeline.run_benchmark import run_benchmark_batch
from scripts.qa_bench_pipeline.evaluator import evaluate_runs
from scripts.qa_bench_pipeline.summary_agent import QABenchSummaryAgent

DEFAULT_SUMMARY_BASE_URL = "http://123.57.228.132:8285/api"
DEFAULT_SUMMARY_MODEL = "deepseek-v3.2-20251201-160k-local"


def build_experience_data(
    evaluated_results: list,
    summaries: list,
) -> dict:
    """
    Build final experience data, classified into factual and SOP experiences.

    Args:
        evaluated_results: List of evaluated benchmark results
        summaries: List of experience summaries

    Returns:
        Dict with "factual_experiences" and "sop_experiences" lists
    """
    summary_map = {}
    for s in summaries:
        key = s["sample_id"]
        summary_map[key] = s

    factual_experiences = []
    sop_experiences = []

    for result in evaluated_results:
        sample_id = result["sample_id"]
        dataset = result["dataset"]
        experience_type = result.get("experience_type", "factual")

        summary_info = summary_map.get(sample_id, {})

        entry = {
            "sample_id": sample_id,
            "dataset": dataset,
            "experience_type": experience_type,
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "ground_truth_aliases": result.get("ground_truth_aliases", []),
            "runs": [],
            "experience_summary": summary_info.get("summary", ""),
            "success_count": summary_info.get("success_count", 0),
            "failure_count": summary_info.get("failure_count", 0),
            "metadata": result.get("metadata", {}),
        }

        for run in result.get("runs", []):
            entry["runs"].append(
                {
                    "run_id": run["run_id"],
                    "prediction": run.get("prediction", ""),
                    "score": run.get("score", 0.0),
                    "success": run.get("success", False),
                    "memory_stack_log": run.get("memory_stack_log", []),
                    "elapsed_seconds": run.get("elapsed_seconds", 0),
                }
            )

        if experience_type == "factual":
            factual_experiences.append(entry)
        else:
            sop_experiences.append(entry)

    return {
        "factual_experiences": factual_experiences,
        "sop_experiences": sop_experiences,
    }


def print_statistics(experience_data: dict):
    """Print summary statistics for the experience data."""
    factual = experience_data["factual_experiences"]
    sop = experience_data["sop_experiences"]

    print("\n" + "=" * 60)
    print("Experience Data Statistics")
    print("=" * 60)

    print(f"\n[Factual Experience (事实性经验)]")
    print(f"  Total samples: {len(factual)}")
    if factual:
        datasets = {}
        for e in factual:
            ds = e["dataset"]
            datasets.setdefault(ds, {"total": 0, "success": 0, "failure": 0})
            datasets[ds]["total"] += 1
            datasets[ds]["success"] += e.get("success_count", 0)
            datasets[ds]["failure"] += e.get("failure_count", 0)
        for ds, stats in datasets.items():
            total_runs = stats["success"] + stats["failure"]
            success_rate = stats["success"] / max(total_runs, 1) * 100
            print(
                f"  {ds}: {stats['total']} samples, "
                f"{stats['success']}/{total_runs} success runs ({success_rate:.1f}%)"
            )

    print(f"\n[SOP System Experience (SOP系统层经验)]")
    print(f"  Total samples: {len(sop)}")
    if sop:
        datasets = {}
        for e in sop:
            ds = e["dataset"]
            datasets.setdefault(ds, {"total": 0, "success": 0, "failure": 0})
            datasets[ds]["total"] += 1
            datasets[ds]["success"] += e.get("success_count", 0)
            datasets[ds]["failure"] += e.get("failure_count", 0)
        for ds, stats in datasets.items():
            total_runs = stats["success"] + stats["failure"]
            success_rate = stats["success"] / max(total_runs, 1) * 100
            print(
                f"  {ds}: {stats['total']} samples, "
                f"{stats['success']}/{total_runs} success runs ({success_rate:.1f}%)"
            )

    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="QA Bench Experience Data Pipeline")
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=f"Datasets to process (default: all). Options: {ALL_DATASETS}",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/qa_bench",
        help="Base output directory. A timestamped subdirectory will be created.",
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of runs per QA sample"
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per dataset (None for all)",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of samples to process in parallel",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark run, use existing results",
    )
    parser.add_argument(
        "--skip-summary", action="store_true", help="Skip summary generation"
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=os.getenv("SUMMARY_BASE_URL", DEFAULT_SUMMARY_BASE_URL),
    )
    parser.add_argument(
        "--summary-api-key",
        type=str,
        default=os.getenv("SUMMARY_API_KEY", ""),
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=os.getenv("SUMMARY_MODEL", DEFAULT_SUMMARY_MODEL),
    )
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{run_timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    selected_datasets = args.datasets or ALL_DATASETS

    print("=" * 60)
    print("QA Bench Experience Data Pipeline")
    print("=" * 60)
    print(f"Datasets: {selected_datasets}")
    print(f"Split: {args.split}")
    print(f"Output dir: {run_dir}")
    print(f"Runs per QA: {args.num_runs}")
    print(f"Temperature: {args.temperature}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Concurrency: {args.concurrency}")

    factual_selected = [d for d in selected_datasets if d in FACTUAL_DATASETS]
    sop_selected = [d for d in selected_datasets if d in SOP_DATASETS]
    print(f"Factual datasets: {factual_selected}")
    print(f"SOP datasets: {sop_selected}")
    print("=" * 60)

    # ─── Step 1: Load data ────────────────────────────────────────────
    print("\n[Step 1] Loading QA benchmark data...")
    samples = load_qa_samples(
        datasets=selected_datasets,
        split=args.split,
        max_samples_per_dataset=args.max_samples,
    )
    print(f"Loaded {len(samples)} total QA samples")

    samples_file = os.path.join(run_dir, "prepared_samples.json")
    save_samples(samples, samples_file)

    # ─── Step 2: Run benchmark ────────────────────────────────────────
    benchmark_results_file = os.path.join(run_dir, "benchmark_results.json")

    if args.skip_benchmark and os.path.exists(benchmark_results_file):
        print("\n[Step 2] Loading existing benchmark results (--skip-benchmark)...")
        with open(benchmark_results_file, "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)
    else:
        print(f"\n[Step 2] Running benchmark ({args.num_runs} runs per QA)...")
        benchmark_results = asyncio.run(
            run_benchmark_batch(
                samples,
                num_runs=args.num_runs,
                temperature=args.temperature,
                output_dir=run_dir,
                resume=not args.no_resume,
                concurrency=args.concurrency,
            )
        )
        with open(benchmark_results_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print(f"Benchmark complete: {len(benchmark_results)} QA samples processed")

    # ─── Step 3: Evaluate results ─────────────────────────────────────
    print("\n[Step 3] Evaluating results with per-dataset metrics...")
    for result in benchmark_results:
        ground_truth = result["ground_truth"]
        dataset = result["dataset"]
        ground_truth_aliases = result.get("ground_truth_aliases", [])
        result["runs"] = evaluate_runs(
            result["runs"], ground_truth, dataset, ground_truth_aliases
        )

        success_count = sum(1 for r in result["runs"] if r.get("success", False))
        total = len(result["runs"])
        avg_score = sum(r.get("score", 0) for r in result["runs"]) / max(total, 1)
        print(
            f"  {result['sample_id']} ({dataset}): "
            f"{success_count}/{total} success, avg_score={avg_score:.4f}"
        )

    evaluated_file = os.path.join(run_dir, "evaluated_results.json")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # ─── Step 4: Generate summaries ───────────────────────────────────
    summaries = []
    if not args.skip_summary:
        print("\n[Step 4] Generating experience summaries...")
        print("  - Factual Experience summaries for TriviaQA/PopQA...")
        print("  - SOP System Experience summaries for GPQA/GSM8K/MATH...")
        summary_agent = QABenchSummaryAgent(
            base_url=args.summary_base_url,
            api_key=args.summary_api_key,
            model=args.summary_model,
        )
        summaries = summary_agent.summarize_batch(
            benchmark_results, concurrency=args.concurrency
        )

        summaries_file = os.path.join(run_dir, "summaries.json")
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    else:
        print("\n[Step 4] Skipping summary generation (--skip-summary)")
        summaries_file = os.path.join(run_dir, "summaries.json")
        if os.path.exists(summaries_file):
            with open(summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)

    # ─── Step 5: Build final experience data ──────────────────────────
    print("\n[Step 5] Building final experience data...")
    experience_data = build_experience_data(benchmark_results, summaries)

    # Save combined experience data
    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)

    # Save factual and SOP experiences separately for convenience
    factual_file = os.path.join(run_dir, "factual_experiences.json")
    with open(factual_file, "w", encoding="utf-8") as f:
        json.dump(
            experience_data["factual_experiences"], f, ensure_ascii=False, indent=2
        )

    sop_file = os.path.join(run_dir, "sop_experiences.json")
    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(experience_data["sop_experiences"], f, ensure_ascii=False, indent=2)

    # ─── Print statistics ─────────────────────────────────────────────
    print_statistics(experience_data)

    print(f"\nPipeline Complete!")
    print(f"All experience data saved to: {run_dir}")
    print(f"  - Combined: {output_file}")
    print(f"  - Factual experiences: {factual_file}")
    print(f"  - SOP experiences: {sop_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
