import asyncio
import json
import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from scripts.locomo_pipeline.data_loader import (
    load_locomo_data,
    extract_qa_samples,
    save_samples,
)
from scripts.locomo_pipeline.run_benchmark import run_benchmark_batch
from scripts.locomo_pipeline.evaluator import evaluate_runs
from scripts.locomo_pipeline.summary_agent import SummaryAgent


DEFAULT_LOCOMO_DATA = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "locomo", "data", "locomo10.json"
)

DEFAULT_SUMMARY_BASE_URL = "http://123.57.228.132:8285/api"
DEFAULT_SUMMARY_MODEL = "deepseek-v3.2-20251201-160k-local"
# Old OpenRouter config:
# DEFAULT_SUMMARY_BASE_URL = "https://openrouter.ai/api/v1"
# DEFAULT_SUMMARY_MODEL = "deepseek/deepseek-v3.2"


def build_experience_data(
    evaluated_results: list,
    summaries: list,
) -> list:
    summary_map = {}
    for s in summaries:
        key = f"{s['sample_id']}_{s['qa_index']}"
        summary_map[key] = s

    experience_data = []
    for result in evaluated_results:
        sample_id = result["sample_id"]
        qa_index = result["qa_index"]
        key = f"{sample_id}_{qa_index}"

        summary_info = summary_map.get(key, {})

        entry = {
            "sample_id": sample_id,
            "qa_index": qa_index,
            "category": result["category"],
            "history": result.get("history", ""),
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            "runs": [],
            "success_summary": summary_info.get("summary", ""),
            "success_count": summary_info.get("success_count", 0),
            "failure_count": summary_info.get("failure_count", 0),
        }

        for run in result.get("runs", []):
            entry["runs"].append({
                "run_id": run["run_id"],
                "prediction": run.get("prediction", ""),
                "f1_score": run.get("f1_score", 0.0),
                "success": run.get("success", False),
                "memory_stack_log": run.get("memory_stack_log", []),
                "elapsed_seconds": run.get("elapsed_seconds", 0),
            })

        experience_data.append(entry)

    return experience_data


def main():
    parser = argparse.ArgumentParser(description="LoCoMo Experience Data Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_LOCOMO_DATA,
        help="Path to locomo10.json",
    )
    parser.add_argument("--output-dir", type=str, default="./results/locomo",
                        help="Base output directory. A timestamped subdirectory will be created for each run.")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-per-conv", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of samples to process in parallel (default: 1). Uses multiprocessing.")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmark run, use existing results")
    parser.add_argument("--skip-summary", action="store_true", help="Skip summary generation")
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

    print("=" * 60)
    print("LoCoMo Experience Data Pipeline")
    print("=" * 60)
    print(f"Data path: {args.data_path}")
    print(f"Output dir: {run_dir}")
    print(f"Runs per QA: {args.num_runs}")
    print(f"Temperature: {args.temperature}")
    print(f"Categories filter: {args.categories}")
    print(f"Concurrency: {args.concurrency}")
    print("=" * 60)

    print("\n[Step 1] Loading LoCoMo data...")
    data = load_locomo_data(args.data_path)
    samples = extract_qa_samples(
        data,
        categories=args.categories,
        max_samples_per_conversation=args.max_per_conv,
    )
    print(f"Loaded {len(samples)} QA samples from {len(data)} conversations")

    samples_file = os.path.join(run_dir, "prepared_samples.json")
    save_samples(samples, samples_file)

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

    print("\n[Step 3] Evaluating results with F1 score...")
    for result in benchmark_results:
        ground_truth = result["ground_truth"]
        category = result["category"]
        result["runs"] = evaluate_runs(result["runs"], ground_truth, category)

        success_count = sum(1 for r in result["runs"] if r.get("success", False))
        total = len(result["runs"])
        avg_f1 = sum(r.get("f1_score", 0) for r in result["runs"]) / max(total, 1)
        print(
            f"  {result['sample_id']} qa={result['qa_index']}: "
            f"{success_count}/{total} success, avg_f1={avg_f1:.4f}"
        )

    evaluated_file = os.path.join(run_dir, "evaluated_results.json")
    with open(evaluated_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    summaries = []
    if not args.skip_summary:
        print("\n[Step 4] Generating experience summaries...")
        summary_agent = SummaryAgent(
            base_url=args.summary_base_url,
            api_key=args.summary_api_key,
            model=args.summary_model,
        )
        summaries = summary_agent.summarize_batch(benchmark_results, concurrency=args.concurrency)

        summaries_file = os.path.join(run_dir, "summaries.json")
        with open(summaries_file, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
    else:
        print("\n[Step 4] Skipping summary generation (--skip-summary)")
        summaries_file = os.path.join(run_dir, "summaries.json")
        if os.path.exists(summaries_file):
            with open(summaries_file, "r", encoding="utf-8") as f:
                summaries = json.load(f)

    print("\n[Step 5] Building final experience data...")
    experience_data = build_experience_data(benchmark_results, summaries)

    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Experience data saved to: {output_file}")
    print(f"Total QA samples: {len(experience_data)}")
    total_success = sum(e.get("success_count", 0) for e in experience_data)
    total_failure = sum(e.get("failure_count", 0) for e in experience_data)
    print(f"Total success runs: {total_success}")
    print(f"Total failure runs: {total_failure}")
    print("=" * 60)


if __name__ == "__main__":
    main()
