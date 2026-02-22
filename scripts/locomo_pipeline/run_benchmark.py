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


async def _run_graph(
    conversation_history: str, question: str, graph
) -> Dict[str, Any]:
    # NOTE: 对话历史放在 observations 中（通过系统 prompt 注入），不放在 messages 中。
    # 原因：如果把长对话放进 messages，LLM 会把它当作直接问答任务，
    # 返回自然语言答案而非 Decision JSON，导致 make_decision 解析失败。
    # observations 在 decision 阶段通过模板 "Task Context Information" 展示，
    # 在 THINK 阶段通过 "Current Progress" 展示。
    task_message = (
        f"You must respond with a Decision JSON object. "
        f"Analyze the conversation context in the system information and determine your action. "
        f"The question to answer is: {question}\n\n"
        f"IMPORTANT: Your response MUST be a JSON object with fields: action, reasoning, params, instruction, locale. "
        f"Do NOT answer the question directly. Choose an action (think/finish) and put your analysis in the reasoning field. "
        f"You MUST respond in English. Set locale to 'en' in your JSON response."
    )

    initial_state = {
        "messages": [{"role": "user", "content": task_message}],
        "observations": [conversation_history],
        "auto_accepted_plan": True,
        "enable_background_investigation": False,
        "user_query": task_message,
        "locale": "en",
    }

    config = {
        "configurable": {
            "thread_id": f"locomo_{datetime.now().timestamp()}",
            "graph_format": "locomo",
            "max_plan_iterations": 1,
            "max_step_num": 3,
            "mcp_settings": {},
        },
        "recursion_limit": 30,
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
    conversation_history = sample["history"]
    question = sample["question"]
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

        result = await _run_graph(conversation_history, question, graph)
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


def _run_sample_worker(args):
    sample, num_runs, temperature = args
    return asyncio.run(run_benchmark_for_sample(sample, num_runs, temperature))


def _save_checkpoint(all_results, checkpoint_file):
    with open(checkpoint_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)


async def run_benchmark_batch(
    samples: List[Dict[str, Any]],
    num_runs: int = RUNS_PER_QA,
    temperature: float = TEMPERATURE,
    output_dir: str = "./results/locomo",
    resume: bool = True,
    concurrency: int = DEFAULT_CONCURRENCY,
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

    pending_samples = []
    for sample in samples:
        key = f"{sample['sample_id']}_{sample['qa_index']}"
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
            key = f"{sample['sample_id']}_{sample['qa_index']}"
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
                key = f"{item[0]['sample_id']}_{item[0]['qa_index']}"
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
    from scripts.locomo_pipeline.data_loader import load_locomo_data, extract_qa_samples

    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark on Stack-Planner")
    parser.add_argument("--data-path", type=str, required=True, help="Path to locomo10.json")
    parser.add_argument("--output-dir", type=str, default="./results/locomo")
    parser.add_argument("--num-runs", type=int, default=RUNS_PER_QA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--categories", type=int, nargs="*", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                        help="Number of samples to process in parallel (default: 1)")
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
            concurrency=args.concurrency,
        )
    )

    output_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Benchmark complete. Results saved to {output_file}")
