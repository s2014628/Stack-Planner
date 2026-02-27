"""
QA Bench Pipeline - Run Benchmark

Execute QA benchmark samples through the Stack-Planner multi-agent system
using central agent + researcher (search agent).

Unlike the locomo pipeline which uses only CentralAgent's THINK/FINISH,
this pipeline leverages the researcher sub-agent for web search to answer
factual and reasoning questions.
"""

import asyncio
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import logger

RUNS_PER_QA = 10
TEMPERATURE = 0.8
DEFAULT_CONCURRENCY = 1


def set_temperature(temperature: float):
    """Set LLM temperature via environment variable."""
    os.environ["BASIC_MODEL__temperature"] = str(temperature)


def _clear_llm_cache():
    """Clear cached LLM instances to force re-initialization."""
    from src.llms.llm import _llm_cache

    _llm_cache.clear()


def _init_qa_bench_agents():
    """Initialize agents with qa_bench graph format (central + researcher)."""
    from src.graph.sp_nodes import init_agents

    init_agents("qa_bench")


def _get_qa_bench_graph():
    """Get the compiled qa_bench graph."""
    from src.graph.builder import qa_bench_graph

    return qa_bench_graph


async def _run_graph(
    question: str,
    dataset: str,
    experience_type: str,
    graph,
) -> Dict[str, Any]:
    """
    Run a single QA sample through the central agent + search agent graph.

    Unlike locomo which puts conversation history in observations,
    QA bench puts the question directly in the task message and relies
    on the researcher (search agent) for information retrieval.

    Args:
        question: The formatted question to answer
        dataset: Dataset name for context
        experience_type: "factual" or "sop"
        graph: Compiled state graph

    Returns:
        Dict with prediction and memory_stack_log
    """
    # Build task message based on experience type
    if experience_type == "factual":
        task_message = (
            f"You must respond with a Decision JSON object. "
            f"This is a factual question that may require searching for information. "
            f"Use the researcher agent to search for relevant facts if needed. "
            f"The question to answer is: {question}\n\n"
            f"IMPORTANT: Your response MUST be a JSON object with fields: "
            f"action, reasoning, params, instruction, locale. "
            f"Do NOT answer the question directly in your first response. "
            f"Choose an action (think/delegate/finish) and put your analysis "
            f"in the reasoning field. "
            f"You MUST respond in English. Set locale to 'en' in your JSON response."
        )
    else:
        # SOP type: GPQA, GSM8K, MATH
        task_message = (
            f"You must respond with a Decision JSON object. "
            f"This is a reasoning question from the {dataset} benchmark. "
            f"You may use the researcher agent to search for relevant knowledge "
            f"or formulas if needed, then reason through the answer step by step. "
            f"The question to answer is: {question}\n\n"
            f"IMPORTANT: Your response MUST be a JSON object with fields: "
            f"action, reasoning, params, instruction, locale. "
            f"Do NOT answer the question directly in your first response. "
            f"Choose an action (think/delegate/finish) and put your analysis "
            f"in the reasoning field. "
            f"You MUST respond in English. Set locale to 'en' in your JSON response."
        )

    initial_state = {
        "messages": [{"role": "user", "content": task_message}],
        "observations": [],
        "auto_accepted_plan": True,
        "enable_background_investigation": False,
        "user_query": task_message,
        "locale": "en",
    }

    config = {
        "configurable": {
            "thread_id": f"qa_bench_{dataset}_{datetime.now().timestamp()}",
            "graph_format": "qa_bench",
            "max_plan_iterations": 1,
            "max_step_num": 5,
            "mcp_settings": {},
        },
        "recursion_limit": 50,
    }

    final_state = None
    async for s in graph.astream(
        input=initial_state, config=config, stream_mode="values"
    ):
        if isinstance(s, dict):
            final_state = s

    prediction = ""
    memory_stack_log = []

    if final_state:
        prediction = final_state.get("final_report", "") or ""
        memory_stack_raw = final_state.get("memory_stack", None)
        if memory_stack_raw and isinstance(memory_stack_raw, str):
            try:
                memory_stack_log = json.loads(memory_stack_raw)
            except json.JSONDecodeError:
                memory_stack_log = []

    return {
        "prediction": prediction,
        "memory_stack_log": memory_stack_log,
    }


async def run_single_qa(
    sample: Dict[str, Any],
    run_id: int,
    graph=None,
) -> Dict[str, Any]:
    """
    Run a single QA benchmark sample once.

    Args:
        sample: Normalized QA sample from data_loader
        run_id: Run iteration number
        graph: Compiled state graph (optional, will be loaded if None)

    Returns:
        Run result dict
    """
    dataset = sample["dataset"]
    experience_type = sample["experience_type"]
    question = sample["question"]
    sample_id = sample["sample_id"]

    logger.info(f"Running sample={sample_id}, run={run_id}")

    start_time = time.time()
    try:
        _clear_llm_cache()
        _init_qa_bench_agents()
        if graph is None:
            graph = _get_qa_bench_graph()

        result = await _run_graph(question, dataset, experience_type, graph)
        prediction = result.get("prediction", "")
        memory_stack_log = result.get("memory_stack_log", [])
    except Exception as e:
        logger.error(f"Run failed: sample={sample_id}, run={run_id}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        prediction = ""
        memory_stack_log = []

    elapsed = time.time() - start_time

    return {
        "run_id": run_id,
        "prediction": prediction,
        "memory_stack_log": memory_stack_log,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }


async def run_benchmark_for_sample(
    sample: Dict[str, Any],
    num_runs: int = RUNS_PER_QA,
    temperature: float = TEMPERATURE,
) -> Dict[str, Any]:
    """
    Run benchmark for a single sample across multiple runs.

    Args:
        sample: Normalized QA sample
        num_runs: Number of runs per sample
        temperature: LLM sampling temperature

    Returns:
        Sample result with all runs
    """
    set_temperature(temperature)
    graph = _get_qa_bench_graph()

    runs = []
    for run_id in range(1, num_runs + 1):
        result = await run_single_qa(sample, run_id, graph=graph)
        runs.append(result)

    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "experience_type": sample["experience_type"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "ground_truth_aliases": sample.get("ground_truth_aliases", []),
        "metadata": sample.get("metadata", {}),
        "runs": runs,
    }


def _run_sample_worker(args):
    """Worker function for multiprocessing."""
    sample, num_runs, temperature = args
    return asyncio.run(run_benchmark_for_sample(sample, num_runs, temperature))


def _save_checkpoint(all_results: List[Dict], checkpoint_file: str):
    """Save checkpoint of current results."""
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


async def run_benchmark_batch(
    samples: List[Dict[str, Any]],
    num_runs: int = RUNS_PER_QA,
    temperature: float = TEMPERATURE,
    output_dir: str = "./results/qa_bench",
    resume: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """
    Run benchmark for a batch of QA samples.

    Args:
        samples: List of normalized QA samples
        num_runs: Number of runs per sample
        temperature: LLM sampling temperature
        output_dir: Directory for output files
        resume: Whether to resume from checkpoint
        concurrency: Number of parallel workers

    Returns:
        List of all benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")

    completed = {}
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        for item in checkpoint_data:
            key = item["sample_id"]
            completed[key] = item
        logger.info(
            f"Resuming from checkpoint: {len(completed)} samples already completed"
        )

    all_results = list(completed.values())

    pending_samples = []
    for sample in samples:
        key = sample["sample_id"]
        if key in completed:
            continue
        pending_samples.append(sample)

    if not pending_samples:
        logger.info("All samples already completed")
        return all_results

    total_pending = len(pending_samples)
    logger.info(f"{total_pending} samples to process, concurrency={concurrency}")

    if concurrency <= 1:
        for idx, sample in enumerate(pending_samples):
            key = sample["sample_id"]
            logger.info(f"Processing [{idx+1}/{total_pending}]: {key}")
            result = await run_benchmark_for_sample(sample, num_runs, temperature)
            all_results.append(result)

            _save_checkpoint(all_results, checkpoint_file)

            intermediate_file = os.path.join(output_dir, f"{key}_result.json")
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        logger.info(f"Using ProcessPoolExecutor with {concurrency} workers")
        work_items = [(s, num_runs, temperature) for s in pending_samples]
        done_count = 0

        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            future_to_sample = {}
            for item in work_items:
                future = executor.submit(_run_sample_worker, item)
                key = item[0]["sample_id"]
                future_to_sample[future] = (key, item[0])

            for future in as_completed(future_to_sample):
                key, sample = future_to_sample[future]
                done_count += 1
                try:
                    result = future.result()
                except Exception as e:
                    logger.error(f"Sample {key} failed: {e}")
                    import traceback

                    logger.error(traceback.format_exc())
                    continue

                logger.info(f"Completed [{done_count}/{total_pending}]: {key}")
                all_results.append(result)

                _save_checkpoint(all_results, checkpoint_file)

                intermediate_file = os.path.join(output_dir, f"{key}_result.json")
                with open(intermediate_file, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

    return all_results


if __name__ == "__main__":
    import argparse
    from scripts.qa_bench_pipeline.data_loader import load_qa_samples

    parser = argparse.ArgumentParser(
        description="Run QA Bench benchmark on Stack-Planner"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help="Datasets to benchmark (default: all)",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-dir", type=str, default="./results/qa_bench")
    parser.add_argument("--num-runs", type=int, default=RUNS_PER_QA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_CONCURRENCY,
        help="Number of samples to process in parallel (default: 1)",
    )
    args = parser.parse_args()

    samples = load_qa_samples(
        datasets=args.datasets,
        split=args.split,
        max_samples_per_dataset=args.max_samples,
    )
    logger.info(f"Loaded {len(samples)} QA samples")

    results = asyncio.run(
        run_benchmark_batch(
            samples,
            num_runs=args.num_runs,
            temperature=args.temperature,
            output_dir=args.output_dir,
            resume=not args.no_resume,
            concurrency=args.concurrency,
        )
    )

    output_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Benchmark complete. Results saved to {output_file}")
