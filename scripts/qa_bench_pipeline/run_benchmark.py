"""
QA Bench Pipeline - Run Benchmark (Optimized)

Execute QA benchmark samples through the Stack-Planner multi-agent system
using central agent + researcher (search agent).

Concurrency model:
- Level 1 (sample concurrency): multiple samples processed in parallel
- Level 2 (run concurrency): multiple runs within one sample in parallel

State isolation:
  Each concurrent run gets its own isolated graph with a fresh CentralAgent
  and SubAgentManager, so memory_stack / _decision_count never leak between
  concurrent invocations.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import logger

RUNS_PER_QA = 10
TEMPERATURE = 0.8
DEFAULT_SAMPLE_CONCURRENCY = 3
DEFAULT_RUN_CONCURRENCY = 5


def set_temperature(temperature: float):
    """Set LLM temperature via environment variable."""
    os.environ["BASIC_MODEL__temperature"] = str(temperature)


def _clear_llm_cache():
    """Clear cached LLM instances to force re-initialization."""
    from src.llms.llm import _llm_cache

    _llm_cache.clear()


def _create_isolated_qa_bench_graph():
    """
    Create a fully isolated qa_bench graph with its own CentralAgent instance.

    Each invocation creates:
    - A fresh CentralAgent with its own memory_stack and _decision_count
    - A fresh SubAgentManager bound to that CentralAgent
    - Node functions that close over these isolated instances
    - A compiled StateGraph

    This ensures concurrent invocations never share mutable state
    (memory_stack, _decision_count, etc.).

    Returns:
        A compiled LangGraph StateGraph with isolated agent state.
    """
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from src.agents.CentralAgent import CentralAgent
    from src.agents.SubAgentManager import SubAgentManager
    from src.graph.types import State

    # Create isolated agent instances (each has its own memory_stack)
    central_agent = CentralAgent(graph_format="qa_bench")
    agent_manager = SubAgentManager(central_agent)

    # --- Closure-based node functions that reference isolated instances ---

    async def isolated_central_agent_node(
        state: State, config: RunnableConfig
    ) -> Command:
        """Central agent node bound to an isolated CentralAgent instance."""
        decision = central_agent.make_decision(state, config)
        return central_agent.execute_action(decision, state, config)

    async def isolated_researcher_node(state: State, config: RunnableConfig) -> Command:
        """Researcher node bound to an isolated SubAgentManager instance.

        Wraps execute_researcher with error handling so that tool
        initialization failures (e.g. missing TAVILY_API_KEY) don't crash
        the entire graph.  Instead, returns an error message back to the
        central agent so it can fall back to its own reasoning.
        """
        try:
            return await agent_manager.execute_researcher(state, config)
        except Exception as e:
            from langchain_core.messages import HumanMessage

            logger.error(f"Researcher node failed: {type(e).__name__}: {e}")
            return Command(
                update={
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Research task failed: {type(e).__name__}: {e}. "
                                "Please answer based on your own knowledge."
                            ),
                            name="researcher",
                        )
                    ],
                    "current_node": "central_agent",
                },
                goto="central_agent",
            )

    # --- Build the graph ---
    builder = StateGraph(State)
    builder.add_node("central_agent", isolated_central_agent_node)
    builder.add_node("researcher", isolated_researcher_node)
    builder.add_edge(START, "central_agent")
    builder.add_edge("central_agent", END)

    return builder.compile()


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
        "recursion_limit": 100,
    }

    final_state = None
    graph_error = None
    try:
        async for s in graph.astream(
            input=initial_state, config=config, stream_mode="values"
        ):
            if isinstance(s, dict):
                final_state = s
    except Exception as e:
        # Catch GraphRecursionError and any other graph execution errors.
        # Use whatever partial state was accumulated before the error.
        graph_error = e
        logger.warning(
            f"Graph execution error ({type(e).__name__}): {e}. "
            f"Using last captured state (exists={final_state is not None})."
        )

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

    result = {
        "prediction": prediction,
        "memory_stack_log": memory_stack_log,
    }
    if graph_error is not None:
        result["graph_error"] = f"{type(graph_error).__name__}: {graph_error}"
    return result


async def run_single_qa(
    sample: Dict[str, Any],
    run_id: int,
) -> Dict[str, Any]:
    """
    Run a single QA benchmark sample once.

    Creates an isolated graph per invocation to guarantee no shared mutable
    state (memory_stack, _decision_count) between concurrent runs.

    Args:
        sample: Normalized QA sample from data_loader
        run_id: Run iteration number

    Returns:
        Run result dict
    """
    dataset = sample["dataset"]
    experience_type = sample["experience_type"]
    question = sample["question"]
    sample_id = sample["sample_id"]

    logger.info(f"Running sample={sample_id}, run={run_id}")

    start_time = time.time()
    error_msg = None
    try:
        # Each run gets a fresh isolated graph (own CentralAgent + memory_stack)
        graph = _create_isolated_qa_bench_graph()
        result = await _run_graph(question, dataset, experience_type, graph)
        prediction = result.get("prediction", "")
        memory_stack_log = result.get("memory_stack_log", [])
        # Propagate graph-level errors (e.g. GraphRecursionError)
        if "graph_error" in result:
            error_msg = result["graph_error"]
    except Exception as e:
        logger.error(f"Run failed: sample={sample_id}, run={run_id}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        prediction = ""
        memory_stack_log = []
        error_msg = f"{type(e).__name__}: {e}"

    elapsed = time.time() - start_time

    run_result: Dict[str, Any] = {
        "run_id": run_id,
        "prediction": prediction,
        "memory_stack_log": memory_stack_log,
        "elapsed_seconds": round(elapsed, 2),
        "timestamp": datetime.now().isoformat(),
    }
    if error_msg:
        run_result["error"] = error_msg
    return run_result


async def run_benchmark_for_sample(
    sample: Dict[str, Any],
    num_runs: int = RUNS_PER_QA,
    temperature: float = TEMPERATURE,
    run_concurrency: int = DEFAULT_RUN_CONCURRENCY,
) -> Dict[str, Any]:
    """
    Run benchmark for a single sample across multiple runs.

    Uses asyncio.gather with a semaphore to run multiple runs concurrently
    within a single sample.  Each run creates its own isolated graph
    (CentralAgent + SubAgentManager) so memory_stack never leaks between
    concurrent runs.

    Args:
        sample: Normalized QA sample
        num_runs: Number of runs per sample
        temperature: LLM sampling temperature
        run_concurrency: Max concurrent runs within this sample

    Returns:
        Sample result with all runs
    """
    set_temperature(temperature)

    run_sem = asyncio.Semaphore(run_concurrency)

    async def _guarded_run(rid: int) -> Dict[str, Any]:
        async with run_sem:
            return await run_single_qa(sample, rid)

    # Launch all runs concurrently, bounded by run_sem
    tasks = [_guarded_run(run_id) for run_id in range(1, num_runs + 1)]
    runs = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and log them
    valid_runs: List[Dict[str, Any]] = []
    for i, run in enumerate(runs):
        if isinstance(run, Exception):
            logger.error(f"Run {i+1} for {sample['sample_id']} raised exception: {run}")
            valid_runs.append(
                {
                    "run_id": i + 1,
                    "prediction": "",
                    "memory_stack_log": [],
                    "elapsed_seconds": 0,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(run),
                }
            )
        else:
            valid_runs.append(run)

    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["dataset"],
        "experience_type": sample["experience_type"],
        "question": sample["question"],
        "ground_truth": sample["ground_truth"],
        "ground_truth_aliases": sample.get("ground_truth_aliases", []),
        "metadata": sample.get("metadata", {}),
        "runs": valid_runs,
    }


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
    concurrency: int = DEFAULT_SAMPLE_CONCURRENCY,
    run_concurrency: int = DEFAULT_RUN_CONCURRENCY,
) -> List[Dict[str, Any]]:
    """
    Run benchmark for a batch of QA samples with two-level concurrency.

    Level 1 (sample concurrency): Multiple samples processed in parallel
             via asyncio.Semaphore + asyncio.gather.
    Level 2 (run concurrency): Multiple runs within one sample processed
             in parallel via asyncio.Semaphore + asyncio.gather.

    This replaces the previous ProcessPoolExecutor approach, which had
    heavy process-spawning overhead and required re-importing everything.

    Args:
        samples: List of normalized QA samples
        num_runs: Number of runs per sample
        temperature: LLM sampling temperature
        output_dir: Directory for output files
        resume: Whether to resume from checkpoint
        concurrency: Number of samples to process in parallel
        run_concurrency: Number of runs per sample to process in parallel

    Returns:
        List of all benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")

    completed: Dict[str, Dict] = {}
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
    logger.info(
        f"{total_pending} samples to process, "
        f"sample_concurrency={concurrency}, run_concurrency={run_concurrency}"
    )

    # Clear LLM cache once and set temperature for the batch.
    # Each concurrent run creates its own isolated graph (with fresh
    # CentralAgent + SubAgentManager), so no shared mutable state.
    _clear_llm_cache()
    set_temperature(temperature)

    # Use asyncio.Semaphore for sample-level concurrency control
    sample_sem = asyncio.Semaphore(concurrency)
    # Lock for thread-safe checkpoint writing
    checkpoint_lock = asyncio.Lock()

    async def _process_sample(
        idx: int, sample: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        key = sample["sample_id"]
        async with sample_sem:
            logger.info(f"Processing [{idx+1}/{total_pending}]: {key}")
            try:
                result = await run_benchmark_for_sample(
                    sample,
                    num_runs,
                    temperature,
                    run_concurrency=run_concurrency,
                )
            except Exception as e:
                logger.error(f"Sample {key} failed: {e}")
                import traceback

                logger.error(traceback.format_exc())
                return None

            # Save checkpoint and intermediate result
            async with checkpoint_lock:
                all_results.append(result)
                _save_checkpoint(all_results, checkpoint_file)

            intermediate_file = os.path.join(output_dir, f"{key}_result.json")
            with open(intermediate_file, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)

            logger.info(f"Completed [{idx+1}/{total_pending}]: {key}")
            return result

    # Launch all samples concurrently, bounded by sample_sem
    tasks = [_process_sample(idx, sample) for idx, sample in enumerate(pending_samples)]
    await asyncio.gather(*tasks, return_exceptions=True)

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
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output-dir", type=str, default="./results/qa_bench")
    parser.add_argument("--num-runs", type=int, default=RUNS_PER_QA)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=DEFAULT_SAMPLE_CONCURRENCY,
        help=(
            "Number of samples to process in parallel "
            f"(default: {DEFAULT_SAMPLE_CONCURRENCY})"
        ),
    )
    parser.add_argument(
        "--run-concurrency",
        type=int,
        default=DEFAULT_RUN_CONCURRENCY,
        help=(
            "Number of runs per sample in parallel "
            f"(default: {DEFAULT_RUN_CONCURRENCY})"
        ),
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
            run_concurrency=args.run_concurrency,
        )
    )

    output_file = os.path.join(args.output_dir, "benchmark_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Benchmark complete. Results saved to {output_file}")
