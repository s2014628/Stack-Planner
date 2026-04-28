"""
Direct Generate Pipeline — Main Entry Point

Faster alternative to the full StackPlanner agent pipeline.
Calls an LLM directly to:
  Step 1. Solve each question (get reasoning + prediction) — N times per sample
  Step 2. Generate structured experience (problem_type, practice, lessons_learned)

Outputs are fully compatible with qa_bench_pipeline's per_run_experiences.json.

Usage
-----
uv run scripts/direct_generate_pipeline/run_pipeline.py \\
    --datasets math gpqa gsm8k triviaqa popqa \\
    --num-runs 2 \\
    --concurrency 10 \\
    --output-dir results/direct_generate \\
    --split train \\
    --max-samples 200

Resume a previous run (already-completed result files are skipped):
uv run scripts/direct_generate_pipeline/run_pipeline.py \\
    --output-dir results/direct_generate/run_20260401_120000 \\
    --datasets math \\
    --resume
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow imports from repo root and scripts folder
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "qa_bench_pipeline"))

from evaluator import evaluate_qa, is_success
from solver_agent import SolverAgent
from experience_agent import ExperienceAgent

# ─── Default model settings (same as conf.yaml BASIC_MODEL) ──────────────────

_DEFAULT_BASE_URL = os.getenv("SUMMARY_BASE_URL", "https://openrouter.ai/api/v1")
_DEFAULT_API_KEY = os.getenv(
    "SUMMARY_API_KEY",
    "",
)
_DEFAULT_MODEL = os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2")

AVAILABLE_DATASETS = ["math", "gpqa", "gsm8k", "triviaqa", "popqa"]


# ─── Per-sample result processing ─────────────────────────────────────────────

def _solve_sample_once(
    solver: SolverAgent,
    sample: Dict[str, Any],
    run_id: int,
) -> Dict[str, Any]:
    """Solve a single sample once and evaluate the result."""
    t0 = time.time()
    result = solver.solve(sample)
    elapsed = round(time.time() - t0, 2)

    prediction = result.get("prediction", "")
    reasoning = result.get("reasoning", "")

    ground_truth = sample.get("ground_truth", "")
    ground_truth_aliases = sample.get("ground_truth_aliases")
    dataset = sample.get("dataset", "")

    score = evaluate_qa(prediction, ground_truth, dataset, ground_truth_aliases)
    success = is_success(score, dataset)

    return {
        "run_id": run_id,
        "reasoning": reasoning,
        "prediction": prediction,
        "score": round(score, 4),
        "success": success,
        "elapsed_seconds": elapsed,
    }


def _process_sample(
    sample: Dict[str, Any],
    solver: SolverAgent,
    experience_agent: ExperienceAgent,
    num_runs: int,
    inner_concurrency: int,
    output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Full pipeline for one sample:
      1. Solve num_runs times (possibly in parallel)
      2. Generate experience from all runs
      3. Expand into one entry per run in the output format
      4. Save to {sample_id}_result.json and return list of per-run entries
    """
    sample_id = sample["sample_id"]
    result_path = output_dir / f"{sample_id}_result.json"

    # Resume: skip if already complete
    if result_path.exists():
        try:
            with open(result_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass  # Re-run if file is corrupted

    # ── Step 1: solve N times ──────────────────────────────────────────────
    runs: List[Dict[str, Any]] = []
    if inner_concurrency > 1 and num_runs > 1:
        with ThreadPoolExecutor(max_workers=min(inner_concurrency, num_runs)) as pool:
            futures = {
                pool.submit(_solve_sample_once, solver, sample, i + 1): i + 1
                for i in range(num_runs)
            }
            for fut in as_completed(futures):
                try:
                    runs.append(fut.result())
                except Exception as exc:
                    run_id = futures[fut]
                    runs.append({
                        "run_id": run_id,
                        "reasoning": f"(Error: {exc})",
                        "prediction": "",
                        "score": 0.0,
                        "success": False,
                        "elapsed_seconds": 0.0,
                    })
    else:
        for i in range(num_runs):
            try:
                runs.append(_solve_sample_once(solver, sample, i + 1))
            except Exception as exc:
                runs.append({
                    "run_id": i + 1,
                    "reasoning": f"(Error: {exc})",
                    "prediction": "",
                    "score": 0.0,
                    "success": False,
                    "elapsed_seconds": 0.0,
                })
    runs.sort(key=lambda r: r["run_id"])

    # ── Step 2: generate experience ────────────────────────────────────────
    try:
        exp = experience_agent.generate(sample, runs)
    except Exception as exc:
        exp = {
            "problem_type": "",
            "problem_addressed": "",
            "practice": f"(Error: {exc})",
            "lessons_learned": "",
            "experience_summary": "",
        }

    # ── Build per-run experience entries ───────────────────────────────────
    entries = []
    for run in runs:
        entry = {
            "sample_id": f"{sample_id}_run_{run['run_id']}",
            "original_sample_id": sample_id,
            "run_id": run["run_id"],
            "dataset": sample.get("dataset", ""),
            "experience_type": sample.get("experience_type", ""),
            "problem_type": exp.get("problem_type", ""),
            "question": sample.get("question", ""),
            "ground_truth": sample.get("ground_truth", ""),
            "ground_truth_aliases": sample.get("ground_truth_aliases", []),
            "reasoning": run.get("reasoning", ""),
            "prediction": run.get("prediction", ""),
            "score": run.get("score", 0.0),
            "success": run.get("success", False),
            "memory_stack_log": [],  # direct generate — no agent memory
            "elapsed_seconds": run.get("elapsed_seconds", 0.0),
            "problem_addressed": exp.get("problem_addressed", ""),
            "practice": exp.get("practice", ""),
            "lessons_learned": exp.get("lessons_learned", ""),
            "experience_summary": exp.get("experience_summary", ""),
            "metadata": sample.get("metadata", {}),
        }
        entries.append(entry)

    # Save individual result
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    return entries


# ─── Dataset loading ──────────────────────────────────────────────────────────

def _load_samples(
    datasets: List[str],
    split: str,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """Load QA samples using qa_bench_pipeline's data_loader."""
    from data_loader import load_qa_samples

    print(f"Loading datasets: {datasets}, split={split}, max_samples={max_samples}")
    samples = load_qa_samples(
        datasets=datasets,
        split=split,
        max_samples_per_dataset=max_samples,
    )
    print(f"Total samples loaded: {len(samples)}")
    return samples


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(
    datasets: List[str],
    split: str,
    max_samples: Optional[int],
    num_runs: int,
    concurrency: int,
    output_dir: Path,
    base_url: str,
    api_key: str,
    model: str,
    resume: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    samples = _load_samples(datasets, split, max_samples)

    solver = SolverAgent(base_url=base_url, api_key=api_key, model=model)
    experience_agent = ExperienceAgent(base_url=base_url, api_key=api_key, model=model)

    # Determine inner parallelism so that total concurrent LLM calls stay at concurrency.
    # Heuristic: solve inner first, experience second; cap inner at num_runs.
    inner_conc = min(num_runs, max(1, concurrency // max(1, len(samples))))

    all_entries: List[Dict[str, Any]] = []
    failed_samples: List[str] = []

    t_start = time.time()

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(
                _process_sample,
                sample,
                solver,
                experience_agent,
                num_runs,
                inner_conc,
                output_dir,
            ): sample["sample_id"]
            for sample in samples
        }

        done_count = 0
        for fut in as_completed(futures):
            sample_id = futures[fut]
            done_count += 1
            try:
                result = fut.result()
                if result:
                    if isinstance(result, list):
                        all_entries.extend(result)
                    else:
                        all_entries.append(result)
                elapsed = time.time() - t_start
                avg = elapsed / done_count
                eta = avg * (len(samples) - done_count)
                print(
                    f"[{done_count}/{len(samples)}] {sample_id} done  "
                    f"(elapsed {elapsed:.0f}s, ETA {eta:.0f}s)"
                )
            except Exception as exc:
                print(f"[ERROR] {sample_id}: {exc}")
                failed_samples.append(sample_id)

    # Sort by original_sample_id then run_id for deterministic output
    all_entries.sort(
        key=lambda e: (e.get("original_sample_id", ""), e.get("run_id", 0))
    )

    # ── Save combined output (compatible with per_run_experiences.json) ────
    experiences_path = output_dir / "per_run_experiences.json"
    with open(experiences_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_entries)} entries to {experiences_path}")

    # ── Basic statistics ──────────────────────────────────────────────────
    if all_entries:
        by_dataset: Dict[str, Dict[str, Any]] = {}
        for e in all_entries:
            ds = e.get("dataset", "unknown")
            if ds not in by_dataset:
                by_dataset[ds] = {"total": 0, "correct": 0}
            by_dataset[ds]["total"] += 1
            if e.get("success", False):
                by_dataset[ds]["correct"] += 1
        print("\n─── Accuracy ───────────────────────────────")
        for ds, stats in sorted(by_dataset.items()):
            acc = stats["correct"] / stats["total"] if stats["total"] else 0
            print(f"  {ds:12s}: {acc:.1%}  ({stats['correct']}/{stats['total']})")
        print("────────────────────────────────────────────")

    if failed_samples:
        print(f"\nFailed samples ({len(failed_samples)}): {failed_samples[:20]}")

    total_elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct LLM generation pipeline — faster alternative to StackPlanner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["math"],
        choices=AVAILABLE_DATASETS,
        metavar="DATASET",
        help=f"Datasets to process. Choices: {AVAILABLE_DATASETS}",
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "test", "validation"],
        help="Dataset split",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        metavar="N",
        help="Max samples per dataset (None = all)",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="Number of solve attempts per sample (multiple runs enable lessons_learned)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        metavar="N",
        help="Number of samples to process concurrently",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Output directory. Defaults to "
            "results/direct_generate/run_<TIMESTAMP>"
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous run: skip samples that already have result files",
    )
    parser.add_argument(
        "--base-url",
        default=_DEFAULT_BASE_URL,
        help="LLM API base URL",
    )
    parser.add_argument(
        "--api-key",
        default=_DEFAULT_API_KEY,
        help="LLM API key (falls back to SUMMARY_API_KEY env var)",
    )
    parser.add_argument(
        "--model",
        default=_DEFAULT_MODEL,
        help="LLM model name",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = _REPO_ROOT / "results" / "direct_generate" / f"run_{timestamp}"

    run_pipeline(
        datasets=args.datasets,
        split=args.split,
        max_samples=args.max_samples,
        num_runs=args.num_runs,
        concurrency=args.concurrency,
        output_dir=output_dir,
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
