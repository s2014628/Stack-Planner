"""
QA Bench Experience Data Pipeline

End-to-end pipeline for generating experience data from QA benchmarks:
1. Load QA samples from specified datasets
2. Run benchmark through central agent + search agent
3. Evaluate results with per-dataset metrics
4. Generate experience summaries (factual + SOP)
5. Build final experience data (classified by experience type)

Cross-dataset isolation:
  Each dataset is processed as a **fully independent batch**.  Before each
  batch the pipeline resets all module-level shared state (global_statistics,
  LLM cache, search semaphore) so that datasets never interfere with each
  other -- even when running in parallel with ``--dataset-parallel``.

Experience Types:
- 事实性经验 (Factual Experience): TriviaQA, PopQA
  - Knowledge retrieval, search strategy, fact verification patterns
- SOP系统层经验 (SOP System Experience): GPQA, GSM8K, MATH
  - Step-by-step reasoning workflows, tool usage SOPs, error recovery

Usage:
    python -m scripts.qa_bench_pipeline.run_pipeline \\
        --datasets triviaqa popqa gpqa gsm8k math \\
        --max-samples 50 --num-runs 5

    # Run datasets in parallel (share search rate-limiter):
    python -m scripts.qa_bench_pipeline.run_pipeline \\
        --datasets triviaqa popqa --dataset-parallel \\
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
from scripts.qa_bench_pipeline.run_benchmark import (
    run_benchmark_batch,
    reset_global_state,
)
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


# ─────────────────────────────────────────────────────────────
#  Per-dataset pipeline: each dataset is a fully isolated batch
# ─────────────────────────────────────────────────────────────


async def _run_dataset_pipeline(
    dataset_name: str,
    args,
    run_dir: str,
) -> dict:
    """Run the complete pipeline for a single dataset.

    This function is the **unit of isolation**: before doing any work it
    resets all module-level shared state so that nothing leaks from a
    previous (or concurrent) dataset batch.

    Returns:
        Dict with keys ``benchmark_results``, ``summaries``,
        ``experience_data``, and ``dataset_dir``.
    """
    # 0. Isolate: reset shared state
    reset_global_state()

    dataset_dir = os.path.join(run_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    sep = "-" * 60
    print(f"\n{sep}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Output:  {dataset_dir}")
    print(f"{sep}")

    # 1. Load data
    samples = load_qa_samples(
        datasets=[dataset_name],
        split=args.split,
        max_samples_per_dataset=args.max_samples,
    )
    print(f"  [{dataset_name}] Loaded {len(samples)} samples")
    save_samples(samples, os.path.join(dataset_dir, "prepared_samples.json"))

    # 2. Run benchmark
    benchmark_results_file = os.path.join(dataset_dir, "benchmark_results.json")

    if args.skip_benchmark and os.path.exists(benchmark_results_file):
        print(
            f"  [{dataset_name}] Loading existing benchmark results "
            f"(--skip-benchmark)"
        )
        with open(benchmark_results_file, "r", encoding="utf-8") as f:
            benchmark_results = json.load(f)
    else:
        print(
            f"  [{dataset_name}] Running benchmark " f"({args.num_runs} runs per QA)..."
        )
        benchmark_results = await run_benchmark_batch(
            samples,
            num_runs=args.num_runs,
            temperature=args.temperature,
            output_dir=dataset_dir,
            resume=not args.no_resume,
            concurrency=args.concurrency,
            run_concurrency=args.run_concurrency,
            search_concurrency=args.search_concurrency,
        )
        with open(benchmark_results_file, "w", encoding="utf-8") as f:
            json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print(
        f"  [{dataset_name}] Benchmark complete: " f"{len(benchmark_results)} samples"
    )

    # 3. Evaluate
    print(f"  [{dataset_name}] Evaluating results...")
    for result in benchmark_results:
        ground_truth = result["ground_truth"]
        ds = result["dataset"]
        ground_truth_aliases = result.get("ground_truth_aliases", [])
        result["runs"] = evaluate_runs(
            result["runs"], ground_truth, ds, ground_truth_aliases
        )

        success_count = sum(1 for r in result["runs"] if r.get("success", False))
        total = len(result["runs"])
        avg_score = sum(r.get("score", 0) for r in result["runs"]) / max(total, 1)
        sid = result["sample_id"]
        print(
            f"    {sid}: " f"{success_count}/{total} success, avg_score={avg_score:.4f}"
        )

    evaluated_file = os.path.join(dataset_dir, "evaluated_results.json")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # 4. Summarise
    summaries = []
    if not args.skip_summary:
        print(f"  [{dataset_name}] Generating experience summaries...")
        summary_agent = QABenchSummaryAgent(
            base_url=args.summary_base_url,
            api_key=args.summary_api_key,
            model=args.summary_model,
        )
        summaries = summary_agent.summarize_batch(
            benchmark_results, concurrency=args.concurrency
        )
        summaries_file = os.path.join(dataset_dir, "summaries.json")
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    else:
        summaries_file = os.path.join(dataset_dir, "summaries.json")
        if os.path.exists(summaries_file):
            with open(summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)

    # 5. Build experience data
    experience_data = build_experience_data(benchmark_results, summaries)

    # Save per-dataset experience data
    exp_file = os.path.join(dataset_dir, "experience_data.json")
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)
    fact_file = os.path.join(dataset_dir, "factual_experiences.json")
    with open(fact_file, "w", encoding="utf-8") as f:
        json.dump(
            experience_data["factual_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )
    sop_file = os.path.join(dataset_dir, "sop_experiences.json")
    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(
            experience_data["sop_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"  [{dataset_name}] Pipeline complete")

    return {
        "benchmark_results": benchmark_results,
        "summaries": summaries,
        "experience_data": experience_data,
        "dataset_dir": dataset_dir,
    }


def _merge_experience_data(per_dataset_results: list) -> dict:
    """Merge per-dataset experience_data dicts into a single combined dict."""
    merged = {"factual_experiences": [], "sop_experiences": []}
    for r in per_dataset_results:
        ed = r["experience_data"]
        merged["factual_experiences"].extend(ed.get("factual_experiences", []))
        merged["sop_experiences"].extend(ed.get("sop_experiences", []))
    return merged


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
        default=3,
        help="Number of samples to process in parallel (default: 3)",
    )
    parser.add_argument(
        "--run-concurrency",
        type=int,
        default=5,
        help="Number of runs per sample in parallel (default: 5)",
    )
    parser.add_argument(
        "--search-concurrency",
        type=int,
        default=5,
        help=(
            "Max concurrent researcher (search) calls globally. "
            "Lower this if you hit search-API rate limits (default: 5)"
        ),
    )
    parser.add_argument(
        "--dataset-parallel",
        action="store_true",
        help=(
            "Process datasets in parallel instead of sequentially. "
            "When enabled, all datasets run concurrently but share the "
            "global search rate-limiter to avoid API overload."
        ),
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
        default=os.getenv("SUMMARY_API_KEY", "sk-7374e2abda1141ffa4fe8eb01ae582f7"),
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

    factual_selected = [d for d in selected_datasets if d in FACTUAL_DATASETS]
    sop_selected = [d for d in selected_datasets if d in SOP_DATASETS]

    print("=" * 60)
    print("QA Bench Experience Data Pipeline")
    print("=" * 60)
    print(f"Datasets: {selected_datasets}")
    print(f"Split: {args.split}")
    print(f"Output dir: {run_dir}")
    print(f"Runs per QA: {args.num_runs}")
    print(f"Temperature: {args.temperature}")
    print(f"Max samples per dataset: {args.max_samples}")
    print(f"Sample concurrency: {args.concurrency}")
    print(f"Run concurrency: {args.run_concurrency}")
    print(f"Search concurrency: {args.search_concurrency}")
    print(f"Dataset parallel: {args.dataset_parallel}")
    print(f"Factual datasets: {factual_selected}")
    print(f"SOP datasets: {sop_selected}")
    print("=" * 60)

    # ─── Per-dataset pipeline execution ───────────────────────────
    if args.dataset_parallel:
        # Parallel: all datasets run concurrently via asyncio.gather.
        # They share the global search semaphore so API rate limits are
        # respected, but each dataset resets its own statistics / caches.

        async def _run_all_parallel():
            tasks = [
                _run_dataset_pipeline(ds, args, run_dir) for ds in selected_datasets
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        raw_results = asyncio.run(_run_all_parallel())
        per_dataset_results = []
        for i, r in enumerate(raw_results):
            if isinstance(r, Exception):
                ds_name = selected_datasets[i]
                print(f"\n  [ERROR] Dataset {ds_name} failed: {r}")
            else:
                per_dataset_results.append(r)
    else:
        # Sequential (default): one dataset at a time, fully isolated.
        per_dataset_results = []
        for ds in selected_datasets:
            try:
                result = asyncio.run(_run_dataset_pipeline(ds, args, run_dir))
                per_dataset_results.append(result)
            except Exception as e:
                print(f"\n  [ERROR] Dataset {ds} failed: {e}")

    if not per_dataset_results:
        print("\nNo datasets completed successfully.")
        return

    # ─── Merge results across datasets ────────────────────────────
    print("\n[Merging] Combining results from all datasets...")
    merged_experience = _merge_experience_data(per_dataset_results)

    # Save combined experience data at the top level
    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_experience, f, ensure_ascii=False, indent=2)

    factual_file = os.path.join(run_dir, "factual_experiences.json")
    with open(factual_file, "w", encoding="utf-8") as f:
        json.dump(
            merged_experience["factual_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    sop_file = os.path.join(run_dir, "sop_experiences.json")
    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(
            merged_experience["sop_experiences"],
            f,
            ensure_ascii=False,
            indent=2,
        )

    # ─── Print statistics ─────────────────────────────────────────
    print_statistics(merged_experience)

    print(f"\nPipeline Complete!")
    print(f"All experience data saved to: {run_dir}")
    print(f"  - Combined: {output_file}")
    print(f"  - Factual experiences: {factual_file}")
    print(f"  - SOP experiences: {sop_file}")
    for r in per_dataset_results:
        dd = r["dataset_dir"]
        print(f"  - {dd}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
