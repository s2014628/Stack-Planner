"""
Standalone QA Interface for Stack-Planner

Provides a simple callable interface to the multi-agent QA system.
Input: a question string.
Output: answer (prediction) + execution trajectory (memory_stack_log).

This module does NOT modify any existing code. It reuses the same
isolated-graph pattern from qa_bench_pipeline to create fresh agent
instances per invocation.

Usage as a Python module:
    from scripts.qa_api.qa_interface import ask, ask_sync

    # async
    result = await ask("What is the capital of France?")
    print(result["answer"])
    print(result["trajectory"])

    # sync wrapper
    result = ask_sync("What is the capital of France?")

Usage from CLI:
    python -m scripts.qa_api.qa_interface "What is the capital of France?"
    python -m scripts.qa_api.qa_interface --no-search "What is 2+2?"
    python -m scripts.qa_api.qa_interface --json "Some question"
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.utils.logger import logger


# ---------------------------------------------------------------------------
# Graph factories (isolated per invocation, mirrors qa_bench_pipeline)
# ---------------------------------------------------------------------------

def _create_qa_graph_with_search():
    """
    Create an isolated QA graph with CentralAgent + Researcher.

    Each call returns a brand-new compiled StateGraph with its own
    CentralAgent and SubAgentManager so that memory_stack / _decision_count
    never leak between invocations.
    """
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from src.agents.CentralAgent import CentralAgent
    from src.agents.SubAgentManager import SubAgentManager
    from src.graph.types import State

    central_agent = CentralAgent(graph_format="qa_bench")
    agent_manager = SubAgentManager(central_agent)

    async def central_agent_node(state: State, config: RunnableConfig) -> Command:
        decision = central_agent.make_decision(state, config)
        return central_agent.execute_action(decision, state, config)

    async def researcher_node(state: State, config: RunnableConfig) -> Command:
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

    builder = StateGraph(State)
    builder.add_node("central_agent", central_agent_node)
    builder.add_node("researcher", researcher_node)
    builder.add_edge(START, "central_agent")
    builder.add_edge("central_agent", END)

    return builder.compile()


def _create_qa_graph_reasoning_only():
    """
    Create an isolated reasoning-only graph (no search).

    Suitable for math/logic problems or when search is not needed.
    """
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    from src.agents.CentralAgent import CentralAgent
    from src.graph.types import State

    central_agent = CentralAgent(graph_format="qa_bench_reasoning")

    async def central_agent_node(state: State, config: RunnableConfig) -> Command:
        decision = central_agent.make_decision(state, config)
        return central_agent.execute_action(decision, state, config)

    builder = StateGraph(State)
    builder.add_node("central_agent", central_agent_node)
    builder.add_edge(START, "central_agent")
    builder.add_edge("central_agent", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Core ask function
# ---------------------------------------------------------------------------

async def ask(
    question: str,
    *,
    use_search: bool = True,
    temperature: float = 0.0,
    max_step_num: int = 5,
    locale: str = "en",
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with execution trajectory.

    Args:
        question:     The question to answer.
        use_search:   Whether to enable the researcher/search agent.
                      Set to False for pure reasoning (math, logic).
        temperature:  LLM sampling temperature. Default 0.0 for
                      deterministic answers in API usage.
        max_step_num: Maximum reasoning steps for the central agent.
        locale:       Response language ("en" or "zh").

    Returns:
        Dict with keys:
            - answer (str): The final answer text.
            - trajectory (list): List of memory stack entries showing
              the agent's decision-making process.
            - elapsed_seconds (float): Wall-clock time in seconds.
            - error (str | None): Error message if something went wrong.
    """
    # Set temperature
    os.environ["BASIC_MODEL__temperature"] = str(temperature)

    # Clear LLM cache to pick up temperature change
    from src.llms.llm import _llm_cache
    _llm_cache.clear()

    # Build the appropriate graph
    if use_search:
        graph = _create_qa_graph_with_search()
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
            f"You MUST respond in {'English' if locale == 'en' else 'Chinese'}. "
            f"Set locale to '{locale}' in your JSON response."
        )
    else:
        graph = _create_qa_graph_reasoning_only()
        task_message = (
            f"You must respond with a Decision JSON object. "
            f"This is a reasoning problem. "
            f"You do NOT have access to any search tools. "
            f"Solve this problem step by step using your own reasoning ability. "
            f"Use 'think' actions to work through the problem, then 'finish' "
            f"with your final answer.\n\n"
            f"The problem to solve is: {question}\n\n"
            f"IMPORTANT: Your response MUST be a JSON object with fields: "
            f"action, reasoning, params, instruction, locale. "
            f"Do NOT answer the question directly in your first response. "
            f"Choose an action (think/finish) and put your step-by-step "
            f"reasoning in the reasoning field. "
            f"You MUST respond in {'English' if locale == 'en' else 'Chinese'}. "
            f"Set locale to '{locale}' in your JSON response."
        )

    initial_state = {
        "messages": [{"role": "user", "content": task_message}],
        "observations": [],
        "auto_accepted_plan": True,
        "enable_background_investigation": False,
        "user_query": task_message,
        "locale": locale,
    }

    config = {
        "configurable": {
            "thread_id": f"qa_api_{datetime.now().timestamp()}",
            "graph_format": "qa_bench" if use_search else "qa_bench_reasoning",
            "max_plan_iterations": 1,
            "max_step_num": max_step_num,
            "mcp_settings": {},
        },
        "recursion_limit": max(25, max_step_num * 2 + 10),
    }

    start_time = time.time()
    final_state = None
    error_msg = None

    try:
        async for s in graph.astream(
            input=initial_state, config=config, stream_mode="values"
        ):
            if isinstance(s, dict):
                final_state = s
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        logger.warning(
            f"Graph execution error: {error_msg}. "
            f"Using last captured state (exists={final_state is not None})."
        )

    elapsed = round(time.time() - start_time, 2)

    # Extract results
    answer = ""
    trajectory = []

    if final_state:
        answer = final_state.get("final_report", "") or ""
        memory_stack_raw = final_state.get("memory_stack", None)
        if memory_stack_raw and isinstance(memory_stack_raw, str):
            try:
                trajectory = json.loads(memory_stack_raw)
            except json.JSONDecodeError:
                trajectory = []

    return {
        "answer": answer,
        "trajectory": trajectory,
        "elapsed_seconds": elapsed,
        "error": error_msg,
    }


def ask_sync(
    question: str,
    *,
    use_search: bool = True,
    temperature: float = 0.0,
    max_step_num: int = 5,
    locale: str = "en",
) -> Dict[str, Any]:
    """
    Synchronous wrapper around ``ask()``.

    Convenient for non-async callers. Creates a new event loop
    if none is running.

    Args:
        Same as ``ask()``.

    Returns:
        Same as ``ask()``.
    """
    return asyncio.run(
        ask(
            question,
            use_search=use_search,
            temperature=temperature,
            max_step_num=max_step_num,
            locale=locale,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Ask a question via the Stack-Planner multi-agent system"
    )
    parser.add_argument("question", type=str, help="The question to answer")
    parser.add_argument(
        "--no-search",
        action="store_true",
        help="Disable researcher/search agent (pure reasoning mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="LLM sampling temperature (default: 0.0)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum reasoning steps (default: 5)",
    )
    parser.add_argument(
        "--locale",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Response language (default: en)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full result as JSON (including trajectory)",
    )
    args = parser.parse_args()

    result = ask_sync(
        args.question,
        use_search=not args.no_search,
        temperature=args.temperature,
        max_step_num=args.max_steps,
        locale=args.locale,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"\n{'=' * 60}")
        print(f"Question: {args.question}")
        print(f"{'=' * 60}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nElapsed: {result['elapsed_seconds']}s")
        if result["error"]:
            print(f"Error: {result['error']}")
        if result["trajectory"]:
            print(f"\nTrajectory ({len(result['trajectory'])} steps):")
            for i, step in enumerate(result["trajectory"]):
                action = step.get("action", "unknown")
                reasoning = step.get("reasoning", "")
                preview = reasoning[:120] + "..." if len(reasoning) > 120 else reasoning
                print(f"  [{i+1}] {action}: {preview}")


if __name__ == "__main__":
    main()
