"""
Generate experience data and summaries from an existing LoCoMo benchmark run directory.

Use when a pipeline run did not complete (e.g. benchmark was interrupted), so
experience_data / summaries / per_run_experiences were never written. This script:

1. Load benchmark results from the run directory (benchmark_results.json or
   checkpoint.json or aggregate *_result.json files).
2. Evaluate runs (add f1_score/success), save evaluated_results.json.
3. Generate experience summaries via SummaryAgent, save summaries.json.
4. Build experience_data (per-sample), save experience_data.json.
5. Optionally build per-run flat experience data with per-run summaries
   from real execution traces (saves per_run_experiences.json).

Usage:
    python -m scripts.locomo_pipeline.generate_experience_from_run \\
        --run-dir results/locomo/run_20260311_083526

    # Skip summary generation (e.g. no API key):
    python -m scripts.locomo_pipeline.generate_experience_from_run \\
        --run-dir results/locomo/run_20260311_083526 --skip-summary

    # Generate per-run experience data for RQ-KMeans pipeline:
    python -m scripts.locomo_pipeline.generate_experience_from_run \\
        --run-dir results/locomo/run_20260311_083526 --per-run --per-run-summary
"""

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.locomo_pipeline.evaluator import evaluate_runs
from scripts.locomo_pipeline.run_pipeline import (
    build_experience_data,
    build_experience_data_per_run,
)
from scripts.locomo_pipeline.summary_agent import (
    SummaryAgent,
    SUMMARY_API_KEY,
    SUMMARY_BASE_URL,
    SUMMARY_MODEL,
)


def _normalize_run_predictions(runs: List[Dict[str, Any]]) -> None:
    """Ensure each run's prediction is a string for evaluator."""
    for run in runs:
        pred = run.get("prediction", "")
        if not isinstance(pred, str):
            run["prediction"] = str(pred) if pred is not None else ""


def load_benchmark_results(run_dir: str) -> List[Dict[str, Any]]:
    """
    Load benchmark results from run_dir.
    Prefer: benchmark_results.json > checkpoint.json > aggregate *_result.json.
    """
    bench_file = os.path.join(run_dir, "benchmark_results.json")
    if os.path.exists(bench_file):
        with open(bench_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} samples from benchmark_results.json")
        return results

    checkpoint_file = os.path.join(run_dir, "checkpoint.json")
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} samples from checkpoint.json")
        return results

    # Aggregate from *_result.json
    pattern = os.path.join(run_dir, "*_result.json")
    files = sorted(glob.glob(pattern))
    results = []
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            results.append(json.load(f))
    results.sort(key=lambda x: (x.get("sample_id", ""), x.get("qa_index", 0)))
    print(f"  Loaded {len(results)} samples from {len(files)} *_result.json files")
    return results


def process_run(
    run_dir: str,
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
    Run evaluate -> summarize -> build_experience_data for a run directory.
    Returns dict with benchmark_results, summaries, experience_data.
    """
    print(f"\n  Run dir: {run_dir}")

    benchmark_results = load_benchmark_results(run_dir)
    if not benchmark_results:
        print("  No benchmark results found.")
        return {
            "benchmark_results": [],
            "summaries": [],
            "experience_data": [],
            "per_run_experiences": [],
        }

    # Normalize prediction to string for evaluator
    for result in benchmark_results:
        _normalize_run_predictions(result.get("runs", []))

    # Evaluate
    print(f"  Evaluating {len(benchmark_results)} samples...")
    for result in benchmark_results:
        ground_truth = result["ground_truth"]
        category = result["category"]
        result["runs"] = evaluate_runs(result["runs"], ground_truth, category)
        success_count = sum(1 for r in result["runs"] if r.get("success", False))
        total = len(result["runs"])
        avg_f1 = sum(r.get("f1_score", 0) for r in result["runs"]) / max(total, 1)
        sid = result["sample_id"]
        qi = result["qa_index"]
        print(
            f"    {sid} qa={qi}: "
            f"{success_count}/{total} success, avg_f1={avg_f1:.4f}"
        )

    evaluated_file = os.path.join(run_dir, "evaluated_results.json")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
    print(f"  Saved {evaluated_file}")

    # Also write benchmark_results.json so future runs have it
    bench_file = os.path.join(run_dir, "benchmark_results.json")
    with open(bench_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    # Summarize
    summaries = []
    summary_agent = None
    if not skip_summary:
        print("  Generating experience summaries...")
        summary_agent = SummaryAgent(
            base_url=summary_base_url,
            api_key=summary_api_key,
            model=summary_model,
        )
        summaries = summary_agent.summarize_batch(
            benchmark_results, concurrency=concurrency
        )
        summaries_file = os.path.join(run_dir, "summaries.json")
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"  Saved {summaries_file}")
    else:
        summaries_file = os.path.join(run_dir, "summaries.json")
        if os.path.exists(summaries_file):
            with open(summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)
            print(f"  Loaded existing {summaries_file} (--skip-summary)")

    # Try to load prepared samples for evidence info
    samples = None
    samples_file = os.path.join(run_dir, "prepared_samples.json")
    if os.path.exists(samples_file):
        with open(samples_file, "r", encoding="utf-8") as f:
            samples = json.load(f)
        print(f"  Loaded {len(samples)} prepared samples for evidence lookup")

    # Build experience data
    experience_data = build_experience_data(
        benchmark_results, summaries, samples=samples
    )

    # When skip_summary, summary has no success_count/failure_count; fill from runs
    if skip_summary:
        for entry in experience_data:
            runs = entry.get("runs", [])
            entry["success_count"] = sum(1 for r in runs if r.get("success", False))
            entry["failure_count"] = len(runs) - entry["success_count"]

    exp_file = os.path.join(run_dir, "experience_data.json")
    with open(exp_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)
    print(f"  Saved experience_data.json ({len(experience_data)} entries)")

    # Build per-run flat experience data
    per_run_experiences = []
    if per_run:
        run_summaries_map = None

        # Generate per-run summaries from execution traces
        if per_run_summary and not skip_summary:
            print("  Generating per-run summaries from execution traces...")
            if summary_agent is None:
                summary_agent = SummaryAgent(
                    base_url=summary_base_url,
                    api_key=summary_api_key,
                    model=summary_model,
                )
            run_summaries_map = summary_agent.summarize_all_runs(
                benchmark_results, concurrency=concurrency
            )
            run_summaries_file = os.path.join(run_dir, "per_run_summaries.json")
            with open(run_summaries_file, "w", encoding="utf-8") as f:
                json.dump(run_summaries_map, f, ensure_ascii=False, indent=2)
            print(
                f"  Saved per_run_summaries.json "
                f"({len(run_summaries_map)} run-level summaries)"
            )
        elif per_run_summary and skip_summary:
            run_summaries_file = os.path.join(run_dir, "per_run_summaries.json")
            if os.path.exists(run_summaries_file):
                with open(run_summaries_file, "r", encoding="utf-8") as f:
                    run_summaries_map = json.load(f)
                print(
                    f"  Loaded existing {run_summaries_file} "
                    f"({len(run_summaries_map)} entries, --skip-summary)"
                )

        per_run_experiences = build_experience_data_per_run(
            benchmark_results,
            summaries or None,
            samples=samples,
            per_run_summaries=run_summaries_map,
        )
        per_run_file = os.path.join(run_dir, "per_run_experiences.json")
        with open(per_run_file, "w", encoding="utf-8") as f:
            json.dump(per_run_experiences, f, ensure_ascii=False, indent=2)
        print(
            f"  Saved per_run_experiences.json "
            f"({len(per_run_experiences)} run-level entries)"
        )

    return {
        "benchmark_results": benchmark_results,
        "summaries": summaries,
        "experience_data": experience_data,
        "per_run_experiences": per_run_experiences,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate experience data from an existing LoCoMo benchmark run"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to run directory (e.g. results/locomo/run_20260311_083526)",
    )
    parser.add_argument(
        "--skip-summary",
        action="store_true",
        help="Skip summary generation (only evaluate and build experience from empty summaries)",
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=SUMMARY_BASE_URL,
        help="OpenRouter-compatible API base URL",
    )
    parser.add_argument(
        "--summary-api-key",
        type=str,
        default=SUMMARY_API_KEY,
        help="API key",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=SUMMARY_MODEL,
        help="Model id on the summary provider",
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

    # --per-run-summary implies --per-run
    if args.per_run_summary:
        args.per_run = True

    print("=" * 60)
    print("Generate experience data from existing LoCoMo run")
    print("=" * 60)
    print(f"Run dir: {run_dir}")
    print(f"Skip summary: {args.skip_summary}")
    print(f"Per-run: {args.per_run}")
    print(f"Per-run summary: {args.per_run_summary}")
    print("=" * 60)

    result = process_run(
        run_dir,
        skip_summary=args.skip_summary,
        per_run=args.per_run,
        per_run_summary=args.per_run_summary,
        summary_base_url=args.summary_base_url,
        summary_api_key=args.summary_api_key,
        summary_model=args.summary_model,
        concurrency=args.concurrency,
    )

    experience_data = result["experience_data"]
    per_run_experiences = result["per_run_experiences"]

    print("\n" + "=" * 60)
    print("Done!")
    print(f"Experience data: {len(experience_data)} per-sample entries")
    total_success = sum(e.get("success_count", 0) for e in experience_data)
    total_failure = sum(e.get("failure_count", 0) for e in experience_data)
    print(f"Total success runs: {total_success}")
    print(f"Total failure runs: {total_failure}")
    if per_run_experiences:
        print(f"Per-run experience entries: {len(per_run_experiences)}")
    print(f"Output: {run_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
