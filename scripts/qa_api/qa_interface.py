"""
Standalone QA Interface for Stack-Planner

Provides a simple callable interface to the multi-agent QA system.
Input: a question string.
Output: answer (prediction) + experience data (memory_stack_log + experience_summary).

The output format mirrors the experience data produced by the benchmark
pipelines (qa_bench_pipeline / locomo_pipeline), making it easy to feed
single-question results into the same downstream analysis tools.

This module does NOT modify any existing code. It reuses the same
isolated-graph pattern from qa_bench_pipeline to create fresh agent
instances per invocation.

Usage as a Python module:
    from scripts.qa_api.qa_interface import ask, ask_sync

    # async
    result = await ask("What is the capital of France?")
    print(result["answer"])
    print(result["experience_data"])

    # sync wrapper
    result = ask_sync("What is the capital of France?")

Usage from CLI:
    python -m scripts.qa_api.qa_interface "What is the capital of France?"
    python -m scripts.qa_api.qa_interface --no-search "What is 2+2?"
    python -m scripts.qa_api.qa_interface --json "Some question"
    python -m scripts.qa_api.qa_interface --summarize "Some question"
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
# Experience summary generation (single-run variant)
# ---------------------------------------------------------------------------

EXPERIENCE_SUMMARY_SYSTEM_PROMPT = """You are an expert at analyzing multi-agent system execution traces.

Your task is to analyze a single execution run and produce a structured experience summary
that captures the agent's decision-making process.

Provide your analysis in a structured format:
- **Decision Chain**: The sequence of actions the agent took and why
- **Search Strategy** (if applicable): How the agent formulated search queries
- **Reasoning Quality**: How effectively the agent reasoned through the problem
- **Key Observations**: Notable patterns, strengths, or weaknesses
- **Experience Summary**: Concise, reusable experience patterns

Keep your analysis concise and actionable.
"""

EXPERIENCE_SUMMARY_USER_TEMPLATE = """Analyze the following single execution run for a QA task.

Question: {question}
Answer/Prediction: {prediction}
Mode: {mode}
Elapsed Time: {elapsed}s

## Execution Trace (memory_stack_log):
{memory_stack_text}

Please provide a structured experience analysis of this execution run.
Focus on:
1. What was the agent's decision-making chain?
2. How did the agent arrive at the answer?
3. What reusable experience patterns can be extracted?
"""


def _format_memory_stack_for_summary(memory_stack_log: List[Dict[str, Any]]) -> str:
    """Format memory_stack_log entries into readable text for the summary prompt."""
    if not memory_stack_log:
        return "No execution trace recorded."

    parts = []
    for i, entry in enumerate(memory_stack_log):
        action = entry.get("action", "unknown")
        agent_type = entry.get("agent_type", "")
        content = entry.get("content", "")
        timestamp = entry.get("timestamp", "")
        result = entry.get("result")

        # Handle content that might be a dict
        if isinstance(content, dict):
            content = content.get("content", json.dumps(content, ensure_ascii=False))
        elif not isinstance(content, str):
            content = str(content)

        prefix = f"[{action}]"
        if agent_type:
            prefix = f"[{action} -> {agent_type}]"

        entry_text = f"Step {i+1} ({timestamp[:19] if timestamp else '?'}): {prefix} {content}"

        if result and isinstance(result, dict):
            observations = result.get("observations", [])
            if observations:
                obs_text = []
                for obs in observations:
                    if isinstance(obs, str):
                        obs_preview = obs[:200] + "..." if len(obs) > 200 else obs
                        obs_text.append(f"    - {obs_preview}")
                    elif isinstance(obs, dict):
                        obs_content = obs.get("content", str(obs))
                        if isinstance(obs_content, str):
                            obs_preview = obs_content[:200] + "..." if len(obs_content) > 200 else obs_content
                        else:
                            obs_preview = str(obs_content)[:200]
                        obs_text.append(f"    - {obs_preview}")
                if obs_text:
                    entry_text += "\n  Results:\n" + "\n".join(obs_text)

        parts.append(entry_text)

    return "\n".join(parts)


def generate_experience_summary(
    question: str,
    prediction: str,
    memory_stack_log: List[Dict[str, Any]],
    use_search: bool,
    elapsed_seconds: float,
    *,
    summary_base_url: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    summary_model: Optional[str] = None,
) -> str:
    """
    Generate an experience summary by analyzing the execution trace.

    Uses the same summary LLM endpoint as the benchmark pipelines.

    Args:
        question:         The original question.
        prediction:       The agent's answer.
        memory_stack_log: Raw execution trace (list of MemoryStackEntry dicts).
        use_search:       Whether search mode was used.
        elapsed_seconds:  Execution time.
        summary_base_url: Override for SUMMARY_BASE_URL env var.
        summary_api_key:  Override for SUMMARY_API_KEY env var.
        summary_model:    Override for SUMMARY_MODEL env var.

    Returns:
        Experience summary text.
    """
    from openai import OpenAI

    # Align with qa_bench_pipeline/summary_agent.py OpenRouter defaults
    base_url = summary_base_url or os.getenv(
        "SUMMARY_BASE_URL", "https://openrouter.ai/api/v1"
    )
    api_key = summary_api_key or os.getenv(
        "SUMMARY_API_KEY", "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"
    )
    model = summary_model or os.getenv(
        "SUMMARY_MODEL", "deepseek/deepseek-v3.2"
    )
    client = OpenAI(base_url=base_url, api_key=api_key)

    memory_stack_text = _format_memory_stack_for_summary(memory_stack_log)
    mode = "search-augmented (factual)" if use_search else "reasoning-only"

    user_message = EXPERIENCE_SUMMARY_USER_TEMPLATE.format(
        question=question[:2000],
        prediction=prediction[:1000],
        mode=mode,
        elapsed=elapsed_seconds,
        memory_stack_text=memory_stack_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXPERIENCE_SUMMARY_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Experience summary generation failed: {e}")
        return f"Experience summary generation failed: {e}"


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
    summarize: bool = False,
    summary_base_url: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    summary_model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Ask a question and get an answer with experience data.

    The output format mirrors the experience data produced by
    qa_bench_pipeline / locomo_pipeline.

    Args:
        question:         The question to answer.
        use_search:       Whether to enable the researcher/search agent.
                          Set to False for pure reasoning (math, logic).
        temperature:      LLM sampling temperature. Default 0.0 for
                          deterministic answers in API usage.
        max_step_num:     Maximum reasoning steps for the central agent.
        locale:           Response language ("en" or "zh").
        summarize:        Whether to generate an experience summary via
                          the summary LLM. Requires an additional LLM call.
        summary_base_url: Override for summary LLM base URL.
        summary_api_key:  Override for summary LLM API key.
        summary_model:    Override for summary LLM model name.

    Returns:
        Dict with keys:
            - answer (str): The final answer text (prediction).
            - experience_data (dict): Experience data in the same format
              as the benchmark pipelines, containing:
                - question (str)
                - prediction (str)
                - memory_stack_log (list): Raw execution trace entries.
                - experience_summary (str): LLM-generated experience analysis
                  (empty string if ``summarize=False``).
                - elapsed_seconds (float)
                - mode (str): "search" or "reasoning"
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
    memory_stack_log: List[Dict[str, Any]] = []

    if final_state:
        answer = final_state.get("final_report", "") or ""
        memory_stack_raw = final_state.get("memory_stack", None)
        if memory_stack_raw and isinstance(memory_stack_raw, str):
            try:
                memory_stack_log = json.loads(memory_stack_raw)
            except json.JSONDecodeError:
                memory_stack_log = []

    # Generate experience summary if requested
    experience_summary = ""
    if summarize and memory_stack_log:
        experience_summary = generate_experience_summary(
            question=question,
            prediction=answer,
            memory_stack_log=memory_stack_log,
            use_search=use_search,
            elapsed_seconds=elapsed,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
        )

    # Build experience data in the same format as benchmark pipelines
    experience_data = {
        "question": question,
        "prediction": answer,
        "memory_stack_log": memory_stack_log,
        "experience_summary": experience_summary,
        "elapsed_seconds": elapsed,
        "mode": "search" if use_search else "reasoning",
        "timestamp": datetime.now().isoformat(),
    }

    return {
        "answer": answer,
        "experience_data": experience_data,
        # Keep backward-compatible fields
        "trajectory": memory_stack_log,
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
    summarize: bool = False,
    summary_base_url: Optional[str] = None,
    summary_api_key: Optional[str] = None,
    summary_model: Optional[str] = None,
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
            summarize=summarize,
            summary_base_url=summary_base_url,
            summary_api_key=summary_api_key,
            summary_model=summary_model,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _format_memory_stack_cli(memory_stack_log: List[Dict[str, Any]]) -> str:
    """Format memory_stack_log for human-readable CLI display."""
    if not memory_stack_log:
        return "  (no execution trace recorded)"

    lines = []
    for i, entry in enumerate(memory_stack_log):
        action = entry.get("action", "unknown")
        agent_type = entry.get("agent_type", "")
        content = entry.get("content", "")
        timestamp = entry.get("timestamp", "")
        result = entry.get("result")

        # Handle content that might be a dict
        if isinstance(content, dict):
            content = content.get("content", str(content))
        elif not isinstance(content, str):
            content = str(content)

        # Format action label
        prefix = f"[{action}]"
        if agent_type:
            prefix = f"[{action} -> {agent_type}]"

        ts_label = f" @{timestamp[:19]}" if timestamp else ""
        lines.append(f"  Step {i+1}{ts_label}")
        lines.append(f"    Action: {prefix}")

        # Show content (truncated)
        content_preview = content[:300] + "..." if len(content) > 300 else content
        lines.append(f"    Content: {content_preview}")

        # Show result observations if present
        if result and isinstance(result, dict):
            observations = result.get("observations", [])
            if observations:
                lines.append(f"    Results ({len(observations)} observations):")
                for obs in observations[:3]:  # Show at most 3 observations
                    if isinstance(obs, str):
                        obs_preview = obs[:200] + "..." if len(obs) > 200 else obs
                    elif isinstance(obs, dict):
                        obs_content = obs.get("content", str(obs))
                        obs_preview = str(obs_content)[:200]
                    else:
                        obs_preview = str(obs)[:200]
                    lines.append(f"      - {obs_preview}")
                if len(observations) > 3:
                    lines.append(f"      ... and {len(observations) - 3} more")

        lines.append("")  # blank line between steps

    return "\n".join(lines)


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
        help="Output full result as JSON (including experience data)",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help=(
            "Generate an LLM-based experience summary "
            "analyzing the execution trace. Requires an additional LLM call "
            "to the summary endpoint."
        ),
    )
    parser.add_argument(
        "--summary-base-url",
        type=str,
        default=None,
        help="Override SUMMARY_BASE_URL for experience summary generation",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default=None,
        help="Override SUMMARY_MODEL for experience summary generation",
    )
    args = parser.parse_args()

    result = ask_sync(
        args.question,
        use_search=not args.no_search,
        temperature=args.temperature,
        max_step_num=args.max_steps,
        locale=args.locale,
        summarize=args.summarize,
        summary_base_url=args.summary_base_url,
        summary_model=args.summary_model,
    )

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        exp = result["experience_data"]

        print(f"\n{'=' * 60}")
        print(f"Question: {args.question}")
        print(f"Mode: {exp['mode']}")
        print(f"{'=' * 60}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nElapsed: {result['elapsed_seconds']}s")
        if result["error"]:
            print(f"Error: {result['error']}")

        # Display execution trace (memory_stack_log)
        memory_stack_log = exp["memory_stack_log"]
        print(f"\n{'=' * 60}")
        print(f"Execution Trace / Memory Stack ({len(memory_stack_log)} steps):")
        print(f"{'=' * 60}")
        print(_format_memory_stack_cli(memory_stack_log))

        # Display experience summary if generated
        if exp["experience_summary"]:
            print(f"{'=' * 60}")
            print("Experience Summary:")
            print(f"{'=' * 60}")
            print(exp["experience_summary"])
            print()


if __name__ == "__main__":
    main()