import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import logger
from scripts.locomo_pipeline.data_loader import build_user_query


RUNS_PER_QA = 10
TEMPERATURE = 0.8


def set_temperature(temperature: float):
    os.environ["BASIC_MODEL__temperature"] = str(temperature)


def _clear_llm_cache():
    from src.llms.llm import _llm_cache
    _llm_cache.clear()


def _init_locomo_agents():
    from src.graph.sp_nodes import init_agents
    init_agents("locomo")


def _get_locomo_graph():
    from src.graph.builder import locomo_graph
    return locomo_graph


async def _run_graph(user_query: str, graph) -> Dict[str, Any]:
    initial_state = {
        "messages": [{"role": "user", "content": user_query}],
        "auto_accepted_plan": True,
        "enable_background_investigation": False,
        "user_query": user_query,
    }

    config = {
        "configurable": {
            "thread_id": f"locomo_{datetime.now().timestamp()}",
            "max_plan_iterations": 1,
            "max_step_num": 3,
            "mcp_settings": {},
        },
        "recursion_limit": 100,
    }

    final_state = None
    async for s in graph.astream(input=initial_state, config=config, stream_mode="values"):
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
    user_query = build_user_query(sample)
    sample_id = sample["sample_id"]
    qa_index = sample["qa_index"]

    logger.info(
        f"Running sample={sample_id}, qa={qa_index}, run={run_id}"
    )

    start_time = time.time()
    try:
        _clear_llm_cache()
        _init_locomo_agents()
        if graph is None:
            graph = _get_locomo_graph()

        result = await _run_graph(user_query, graph)
        prediction = result.get("prediction", "")
        memory_stack_log = result.get("memory_stack_log", [])
    except Exception as e:
        logger.error(f"Run failed: sample={sample_id}, qa={qa_index}, run={run_id}: {e}")
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
    set_temperature(temperature)
    graph = _get_locomo_graph()

    runs = []
    for run_id in range(1, num_runs + 1):
        result = await run_single_qa(sample, run_id, graph=graph)
        runs.append(result)

    return {
        "sample_id": sample["sample_id"],
        "qa_index": sample["qa_index"],
        "category": sample["category"],
        "history": sample["history"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "runs": runs,
    }


async def run_benchmark_batch(
    samples: List[Dict[str, Any]],
    num_runs: int = RUNS_PER_QA,
    temperature: float = TEMPERATURE,
    output_dir: str = "./results/locomo",
    resume: bool = True,
) -> List[Dict[str, Any]]:
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")

    completed = {}
    if resume and os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r", encoding="utf-8") as f:
            checkpoint_data = json.load(f)
        for item in checkpoint_data:
            key = f"{item['sample_id']}_{item['qa_index']}"
            completed[key] = item
        logger.info(f"Resuming from checkpoint: {len(completed)} samples already completed")

    all_results = list(completed.values())

    for idx, sample in enumerate(samples):
        key = f"{sample['sample_id']}_{sample['qa_index']}"
        if key in completed:
            logger.info(f"Skipping {key} (already completed)")
            continue

        logger.info(f"Processing [{idx+1}/{len(samples)}]: {key}")
        result = await run_benchmark_for_sample(sample, num_runs, temperature)
        all_results.append(result)

        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        intermediate_file = os.path.join(output_dir, f"{key}_result.json")
        with open(intermediate_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

    return all_results


if __name__ == "__main__":
    import argparse
    from scripts.locomo_pipeline.data_loader import load_locomo_data, extract_qa_samples

    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark on Stack-Planner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--output-dir", type=str, default="./results/locomo")
    parser.add_argument("--num-runs", type=int, default=RUNS_PER_QA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    data = load_locomo_data(args.data_path)
    samples = extract_qa_samples(data, categories=args.categories, max_samples_per_conversation=args.max_samples)
    logger.info(f"Loaded {len(samples)} QA samples")

    results = asyncio.run(
        run_benchmark_batch(
            samples,
            num_runs=args.num_runs,
            temperature=args.temperature,
            output_dir=args.output_dir,
            resume=not args.no_resume,
        )
    )

    output_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Benchmark complete. Results saved to {output_file}")
