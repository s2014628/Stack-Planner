import asyncio
import json
import os
import re
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


def _extract_run_reasoning(memory_stack_log: list) -> str:
    """Extract reasoning text from a single run's memory_stack_log.

    Parses each entry's content (JSON or plain text) and collects
    reasoning fields into a single joined string.
    """
    parts = []
    for entry in memory_stack_log:
        content = entry.get("content", "")
        if not content:
            continue
        raw = content.strip()
        raw = re.sub(r"^```json\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            obj = json.loads(raw)
            reasoning = obj.get("reasoning", "")
            if reasoning:
                parts.append(reasoning.strip())
        except (json.JSONDecodeError, AttributeError):
            parts.append(raw[:500])
    return " ".join(parts)


def build_experience_data_per_run(
    evaluated_results: list,
    summaries: list | None = None,
    samples: list | None = None,
    per_run_summaries: dict | None = None,
) -> list:
    """Build flat experience data where each run is an independent entry.

    Instead of grouping runs under their parent sample, this function
    produces one experience entry per run.  This increases the number of
    data points (N samples * M runs) and gives each entry a unique
    reasoning chain, making it better suited for embedding / codebook
    training (RQ-KMeans -> LlamaFactory pipeline).

    Args:
        evaluated_results: List of evaluated benchmark results (each
            contains ``runs`` with ``memory_stack_log``).
        summaries: Optional list of per-sample summaries.  When provided
            and ``per_run_summaries`` is not available, the sample-level
            summary is used as fallback.
        samples: Optional list of prepared samples for evidence lookup.
        per_run_summaries: Optional dict mapping
            ``"{sample_id}_{qa_index}_run_{run_id}"`` to a run-specific
            summary string.  When provided, each run entry gets its own
            dedicated summary instead of sharing the sample-level one.

    Returns:
        Flat list of experience dicts, each with:
        - sample_id: ``{orig_sample_id}_{qa_index}_run_{run_id}``
        - original_sample_id, qa_index, run_id
        - category, question, ground_truth
        - reasoning: extracted reasoning chain for *this* run only
        - prediction, f1_score, success
        - memory_stack_log: the raw log for this run
        - experience_summary: per-run summary (preferred) or per-sample
          summary (fallback)
    """
    summary_map = {}
    if summaries:
        for s in summaries:
            key = f"{s['sample_id']}_{s['qa_index']}"
            summary_map[key] = s

    sample_map = {}
    if samples:
        for s in samples:
            key = f"{s['sample_id']}_{s['qa_index']}"
            sample_map[key] = s

    per_run_entries: list = []

    for result in evaluated_results:
        sample_id = result["sample_id"]
        qa_index = result["qa_index"]
        category = result["category"]
        question = result["question"]
        ground_truth = result["ground_truth"]
        key = f"{sample_id}_{qa_index}"
        sample_summary = summary_map.get(key, {}).get("summary", "")
        sample_info = sample_map.get(key, {})

        for run in result.get("runs", []):
            run_id = run["run_id"]
            run_key = f"{sample_id}_{qa_index}_run_{run_id}"
            stack_log = run.get("memory_stack_log", [])
            reasoning = _extract_run_reasoning(stack_log)

            # Prefer per-run summary; fall back to sample-level summary
            if per_run_summaries and run_key in per_run_summaries:
                run_summary = per_run_summaries[run_key]
            else:
                run_summary = sample_summary

            entry = {
                "sample_id": run_key,
                "original_sample_id": sample_id,
                "qa_index": qa_index,
                "run_id": run_id,
                "category": category,
                "question": question,
                "ground_truth": ground_truth,
                "reasoning": reasoning,
                "prediction": run.get("prediction", ""),
                "f1_score": run.get("f1_score", 0.0),
                "success": run.get("success", False),
                "memory_stack_log": stack_log,
                "elapsed_seconds": run.get("elapsed_seconds", 0),
                "experience_summary": run_summary,
                # Evidence from sample
                "evidence_refs": sample_info.get("evidence", []),
                "evidence_snippets": sample_info.get("evidence_snippets", []),
            }
            per_run_entries.append(entry)

    return per_run_entries


def build_experience_data(
    evaluated_results: list,
    summaries: list,
    samples: list = None,
    experiences: list = None,
) -> list:
    summary_map = {}
    for s in summaries:
        key = f"{s['sample_id']}_{s['qa_index']}"
        summary_map[key] = s

    experience_map = {}
    if experiences:
        for exp in experiences:
            key = f"{exp['sample_id']}_{exp['qa_index']}"
            experience_map[key] = exp

    sample_map = {}
    if samples:
        for s in samples:
            key = f"{s['sample_id']}_{s['qa_index']}"
            sample_map[key] = s

    experience_data = []
    for result in evaluated_results:
        sample_id = result["sample_id"]
        qa_index = result["qa_index"]
        key = f"{sample_id}_{qa_index}"

        summary_info = summary_map.get(key, {})
        experience_info = experience_map.get(key, {})
        sample_info = sample_map.get(key, {})

        entry = {
            "sample_id": sample_id,
            "qa_index": qa_index,
            "category": result["category"],
            "question": result["question"],
            "ground_truth": result["ground_truth"],
            # Evidence: the raw references and extracted dialogue snippets
            "evidence_refs": sample_info.get("evidence", []),
            "evidence_snippets": sample_info.get("evidence_snippets", []),
            "evidence_session_context": sample_info.get("evidence_session_context", {}),
            # Experience: structured analysis from the LLM
            "evidence_analysis": experience_info.get("evidence_analysis", ""),
            "effective_strategy": experience_info.get("effective_strategy", ""),
            "common_mistakes": experience_info.get("common_mistakes", ""),
            "experience_note": experience_info.get("experience_note", ""),
            # Run statistics
            "success_count": summary_info.get(
                "success_count", experience_info.get("success_count", 0)
            ),
            "failure_count": summary_info.get(
                "failure_count", experience_info.get("failure_count", 0)
            ),
            "success_summary": summary_info.get("summary", ""),
            "runs": [],
            # Keep full history for reference (can be dropped to save space)
            "history": result.get("history", ""),
        }

        for run in result.get("runs", []):
            entry["runs"].append(
                {
                    "run_id": run["run_id"],
                    "prediction": run.get("prediction", ""),
                    "f1_score": run.get("f1_score", 0.0),
                    "success": run.get("success", False),
                    "memory_stack_log": run.get("memory_stack_log", []),
                    "elapsed_seconds": run.get("elapsed_seconds", 0),
                }
            )

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
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/locomo",
        help="Base output directory. A timestamped subdirectory will be created for each run.",
    )
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-per-conv", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of samples to process in parallel (default: 1). Uses multiprocessing.",
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

    # Step 4.5: Generate structured per-question experience notes
    experiences = []
    if not args.skip_summary:
        print("\n[Step 4.5] Generating per-question experience notes...")
        if not summary_agent:
            summary_agent = SummaryAgent(
                base_url=args.summary_base_url,
                api_key=args.summary_api_key,
                model=args.summary_model,
            )
        experiences = summary_agent.generate_experience_batch(
            benchmark_results, samples, concurrency=args.concurrency
        )

        experiences_file = os.path.join(run_dir, "experiences.json")
        with open(experiences_file, "w", encoding="utf-8") as f:
            json.dump(experiences, f, ensure_ascii=False, indent=2)
    else:
        experiences_file = os.path.join(run_dir, "experiences.json")
        if os.path.exists(experiences_file):
            with open(experiences_file, "r", encoding="utf-8") as f:
                experiences = json.load(f)

    # --per-run-summary implies --per-run
    if args.per_run_summary:
        args.per_run = True

    print("\n[Step 5] Building final experience data...")
    experience_data = build_experience_data(
        benchmark_results, summaries, samples=samples, experiences=experiences
    )

    output_file = os.path.join(run_dir, "experience_data.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(experience_data, f, ensure_ascii=False, indent=2)

    # Step 6: Build per-run flat experience data (for RQ-KMeans pipeline)
    per_run_experiences = []
    if args.per_run:
        run_summaries_map = None

        # Generate per-run summaries from real execution traces
        if args.per_run_summary and not args.skip_summary:
            print(
                "\n[Step 6] Generating per-run experience summaries from execution traces..."
            )
            if not summary_agent:
                summary_agent = SummaryAgent(
                    base_url=args.summary_base_url,
                    api_key=args.summary_api_key,
                    model=args.summary_model,
                )
            run_summaries_map = summary_agent.summarize_all_runs(
                benchmark_results, concurrency=args.concurrency
            )
            run_summaries_file = os.path.join(run_dir, "per_run_summaries.json")
            with open(run_summaries_file, "w", encoding="utf-8") as f:
                json.dump(run_summaries_map, f, ensure_ascii=False, indent=2)
            print(
                f"    Saved per_run_summaries.json "
                f"({len(run_summaries_map)} run-level summaries)"
            )
        elif args.per_run_summary and args.skip_summary:
            # Try to load existing per-run summaries
            run_summaries_file = os.path.join(run_dir, "per_run_summaries.json")
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
            samples=samples,
            per_run_summaries=run_summaries_map,
        )
        per_run_file = os.path.join(run_dir, "per_run_experiences.json")
        with open(per_run_file, "w", encoding="utf-8") as f:
            json.dump(per_run_experiences, f, ensure_ascii=False, indent=2)
        print(
            f"    Saved per_run_experiences.json "
            f"({len(per_run_experiences)} run-level entries)"
        )

    print("\n" + "=" * 60)
    print("Pipeline Complete!")
    print(f"Experience data saved to: {output_file}")
    print(f"Total QA samples: {len(experience_data)}")
    total_success = sum(e.get("success_count", 0) for e in experience_data)
    total_failure = sum(e.get("failure_count", 0) for e in experience_data)
    print(f"Total success runs: {total_success}")
    print(f"Total failure runs: {total_failure}")
    if per_run_experiences:
        print(f"Per-run experience entries: {len(per_run_experiences)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
