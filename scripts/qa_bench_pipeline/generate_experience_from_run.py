"""
Generate experience data and summaries from an existing QA bench run directory.

Use when a pipeline run did not complete (e.g. benchmark was interrupted), so
benchmark_results.json / evaluated_results / summaries / experience_data were
never written. This script:

1. For each dataset subdir (e.g. math/, gsm8k/): load benchmark results from
   benchmark_results.json if present, else checkpoint.json, else aggregate
   from *_result.json files.
2. Evaluate runs (add score/success), save evaluated_results.json.
3. Generate experience summaries via QABenchSummaryAgent, save summaries.json.
4. Build experience_data (factual/sop), save per-dataset.
5. Merge all datasets and save run-level experience_data.json,
   factual_experiences.json, sop_experiences.json; print statistics.

Usage:
    python -m scripts.qa_bench_pipeline.generate_experience_from_run \\
        --run-dir results/qa_bench/run_20260311_083526

    # Skip summary generation (e.g. no API key):
    python -m scripts.qa_bench_pipeline.generate_experience_from_run \\
        --run-dir results/qa_bench/run_20260311_083526 --skip-summary
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.qa_bench_pipeline.evaluator import evaluate_runs
from scripts.qa_bench_pipeline.run_pipeline import (
    build_experience_data,
    build_experience_data_per_run,
    print_statistics,
    _merge_experience_data,
)
from scripts.qa_bench_pipeline.summary_agent import QABenchSummaryAgent


def _normalize_run_predictions(runs: List[Dict[str, Any]]) -> None:
    """Ensure each run's prediction is a string for evaluator."""
    for run in runs:
        pred = run.get("prediction", "")
        if not isinstance(pred, str):
            run["prediction"] = str(pred) if pred is not None else ""


def load_benchmark_results(dataset_dir: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from dataset_dir.
    Prefer: benchmark_results.json > checkpoint.json > aggregate *_result.json.
    """
    bench_file = os.path.join(dataset_dir, "benchmark_results.json")
    if os.path.exists(bench_file):
        with open(bench_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"    Loaded {len(results)} samples from benchmark_results.json")
        return results

    checkpoint_file = os.path.join(dataset_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"    Loaded {len(results)} samples from checkpoint.json")
        return results

    # Aggregate from *_result.json
    import glob
    pattern = os.path.join(dataset_dir, "*_result.json")
    files = sorted(glob.glob(pattern))
    results = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    # Sort by sample_id for consistency
    results.sort(key=lambda x: (x.get("dataset", ""), x.get("sample_id", "")))
    print(f"    Loaded {len(results)} samples from {len(files)} *_result.json files")
    return results


def process_dataset(
    dataset_name: str,
    dataset_dir: str,
    *,
    skip_summary: bool = False,
    per_run: bool = False,
    per_run_summary: bool = False,
    summary_base_url: str = "",
    summary_api_key: str = "",
    summary_model: str = "",
    concurrency: int = 3,
) -> Dict[str, Any]:
    """
    Run evaluate -> summarize -> build_experience_data for one dataset.
    Returns dict with benchmark_results, summaries, experience_data, dataset_dir.
    """
    print(f"\n  Dataset: {dataset_name} -> {dataset_dir}")

    benchmark_results = load_benchmark_results(dataset_dir)
    if not benchmark_results:
        print(f"    No benchmark results found, skipping.")
        return {
            "benchmark_results": [],
            "summaries": [],
            "experience_data": {"factual_experiences": [], "sop_experiences": []},
            "dataset_dir": dataset_dir,
        }

    # Normalize prediction to string for evaluator
    for result in benchmark_results:
        _normalize_run_predictions(result.get("runs", []))

    # Evaluate
    print(f"    Evaluating {len(benchmark_results)} samples...")
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
        print(f"      {sid}: {success_count}/{total} success, avg_score={avg_score:.4f}")

    evaluated_file = os.path.join(dataset_dir, "evaluated_results.json")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    print(f"    Saved {evaluated_file}")

    # Also write benchmark_results.json so future runs have it
    bench_file = os.path.join(dataset_dir, "benchmark_results.json")
    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # Summarize
    summaries = []
    if not skip_summary:
        print(f"    Generating experience summaries...")
        summary_agent = QABenchSummaryAgent(
            base_url=summary_base_url,
            api_key=summary_api_key,
            model=summary_model,
        )
        summaries = summary_agent.summarize_batch(
            benchmark_results, concurrency=concurrency
        )
        summaries_file = os.path.join(dataset_dir, "summaries.json")
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"    Saved {summaries_file}")
    else:
        summaries_file = os.path.join(dataset_dir, "summaries.json")
        if os.path.exists(summaries_file):
            with open(summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            print(f"    Loaded existing {summaries_file} (--skip-summary)")

    # Build experience data
    experience_data = build_experience_data(benchmark_results, summaries)

    # When skip_summary, summary has no success_count/failure_count; fill from runs
    if skip_summary:
        for entry in experience_data["factual_experiences"] + experience_data["sop_experiences"]:
            runs = entry.get("runs", [])
            entry["success_count"] = sum(1 for r in runs if r.get("success", False))
            entry["failure_count"] = len(runs) - entry["success_count"]

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
    print(f"    Saved experience_data.json, factual_experiences.json, sop_experiences.json")

    # Build per-run flat experience data
    per_run_experiences = []
    run_summaries_map = None
    if per_run:
        # Generate per-run summaries if requested
        if per_run_summary and not skip_summary:
            print(f"    Generating per-run summaries...")
            summary_agent = QABenchSummaryAgent(
                base_url=summary_base_url,
                api_key=summary_api_key,
                model=summary_model,
            )
            run_summaries_map = summary_agent.summarize_all_runs(
                benchmark_results, concurrency=concurrency
            )
            run_summaries_file = os.path.join(dataset_dir, "per_run_summaries.json")
            with open(run_summaries_file, "w", encoding="utf-8") as f:
                json.dump(run_summaries_map, f, ensure_ascii=False, indent=2)
            print(
                f"    Saved per_run_summaries.json "
                f"({len(run_summaries_map)} run-level summaries)"
            )
        elif per_run_summary and skip_summary:
            # Try to load existing per-run summaries
            run_summaries_file = os.path.join(dataset_dir, "per_run_summaries.json")
            if os.path.exists(run_summaries_file):
                with open(run_summaries_file, "r", encoding="utf-8") as f:
                    run_summaries_map = json.load(f)
                print(
                    f"    Loaded existing {run_summaries_file} "
                    f"({len(run_summaries_map)} entries, --skip-summary)"
                )

        per_run_experiences = build_experience_data_per_run(
            benchmark_results,
            summaries or None,
            per_run_summaries=run_summaries_map,
        )
        per_run_file = os.path.join(dataset_dir, "per_run_experiences.json")
        with open(per_run_file, "w", encoding="utf-8") as f:
            json.dump(per_run_experiences, f, ensure_ascii=False, indent=2)
        print(
            f"    Saved per_run_experiences.json "
            f"({len(per_run_experiences)} run-level entries)"
        )

    return {
        "benchmark_results": benchmark_results,
        "summaries": summaries,
        "experience_data": experience_data,
        "per_run_experiences": per_run_experiences,
        "dataset_dir": dataset_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate experience data and summaries from an existing QA bench run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g. results/qa_bench/run_20260311_083526)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip summary generation (only evaluate and build experience from empty summaries)",
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=os.getenv("SUMMARY_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    parser.add_argument(
        "--summary-api-key",
        type=str,
        default=os.getenv(
            "SUMMARY_API_KEY",
            "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",
        ),
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2"),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=3,
        help="Summary generation concurrency (default: 3)",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help=(
            "Also generate per-run flat experience data where each run is "
            "an independent entry (saves per_run_experiences.json). "
            "This is useful for RQ-KMeans codebook training."
        ),
    )
    parser.add_argument(
        "--per-run-summary",
        action="store_true",
        help=(
            "Generate a dedicated summary for each individual run "
            "(instead of sharing the sample-level summary). "
            "Implies --per-run. Saves per_run_summaries.json."
        ),
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        print(f"Error: not a directory: {run_dir}")
        sys.exit(1)

    # Find dataset dirs: either (1) subdirs of run_dir, or (2) run_dir itself if it looks like a dataset dir
    dataset_dirs = []
    for name in sorted(os.listdir(run_dir)):
        path = os.path.join(run_dir, name)
        if not os.path.isdir(path):
            continue
        if (
            os.path.exists(os.path.join(path, "checkpoint.json"))
            or os.path.exists(os.path.join(path, "benchmark_results.json"))
            or any(f.endswith("_result.json") for f in os.listdir(path))
        ):
            dataset_dirs.append((name, path))

    # If no subdirs found, treat run_dir as a single dataset dir (e.g. .../run_xxx/popqa)
    if not dataset_dirs:
        if (
            os.path.exists(os.path.join(run_dir, "checkpoint.json"))
            or os.path.exists(os.path.join(run_dir, "benchmark_results.json"))
            or any(f.endswith("_result.json") for f in os.listdir(run_dir))
        ):
            dataset_name = os.path.basename(run_dir.rstrip(os.sep))
            dataset_dirs = [(dataset_name, run_dir)]
        else:
            print(f"No dataset subdirs found in {run_dir}")
            print("  Tip: use run root (e.g. .../run_20260309_192448) or a dataset dir (e.g. .../run_xxx/popqa)")
            sys.exit(1)

    print("=" * 60)
    print("Generate experience data from existing run")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Datasets: {[d[0] for d in dataset_dirs]}")
    print(f"Skip summary: {args.skip_summary}")
    print("=" * 60)

    # --per-run-summary implies --per-run
    if args.per_run_summary:
        args.per_run = True

    per_dataset_results = []
    for dataset_name, dataset_dir in dataset_dirs:
        try:
            result = process_dataset(
                dataset_name,
                dataset_dir,
                skip_summary=args.skip_summary,
                per_run=args.per_run,
                per_run_summary=args.per_run_summary,
                summary_base_url=args.summary_base_url,
                summary_api_key=args.summary_api_key,
                summary_model=args.summary_model,
                concurrency=args.concurrency,
            )
            per_dataset_results.append(result)
        except Exception as e:
            print(f"\n  [ERROR] {dataset_name}: {e}")
            import traceback
            traceback.print_exc()

    if not per_dataset_results:
        print("\nNo datasets processed successfully.")
        sys.exit(1)

    # Merge and save run-level experience data
    merged = _merge_experience_data(per_dataset_results)

    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    factual_file = os.path.join(run_dir, "factual_experiences.json")
    with open(factual_file, "w", encoding="utf-8") as f:
        json.dump(merged["factual_experiences"], f, ensure_ascii=False, indent=2)

    sop_file = os.path.join(run_dir, "sop_experiences.json")
    with open(sop_file, "w", encoding="utf-8") as f:
        json.dump(merged["sop_experiences"], f, ensure_ascii=False, indent=2)

    print_statistics(merged)

    # Merge and save run-level per-run experience data
    if args.per_run:
        all_per_run = []
        for r in per_dataset_results:
            all_per_run.extend(r.get("per_run_experiences", []))
        per_run_file = os.path.join(run_dir, "per_run_experiences.json")
        with open(per_run_file, "w", encoding="utf-8") as f:
            json.dump(all_per_run, f, ensure_ascii=False, indent=2)
        print(f"\nPer-run experience data: {len(all_per_run)} entries")
        print(f"  - {per_run_file}")

    print(f"\nDone. Experience data saved to: {run_dir}")
    print(f"  - {output_file}")
    print(f"  - {factual_file}")
    print(f"  - {sop_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
