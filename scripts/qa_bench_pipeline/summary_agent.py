"""
QA Bench Pipeline - Summary Agent

Generate experience summaries from benchmark runs, categorized into:
- 事实性经验 (Factual Experience): Patterns from TriviaQA/PopQA
  - How the agent retrieves factual knowledge
  - Search query formulation strategies
  - Fact verification patterns
- SOP系统层经验 (SOP System Experience): Patterns from GPQA/GSM8K/MATH
  - Step-by-step reasoning workflows
  - Tool usage patterns (search for formulas, concepts)
  - Error recovery and self-correction strategies
"""

import os
from typing import Any, Dict, List

from openai import OpenAI

SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "http://123.57.228.132:8285/api")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "sk-7374e2abda1141ffa4fe8eb01ae582f7")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-v3.2-20251201-160k-local")


# ─── Factual Experience Summary Prompt ───────────────────────────────

FACTUAL_SYSTEM_PROMPT = """You are an expert at analyzing multi-agent system execution traces for factual question-answering tasks.

Your task is to analyze successful and failed runs to extract **factual experience patterns (事实性经验)**.

Factual experience focuses on:
1. **Search Strategy Patterns**: How the agent formulates search queries, what keywords it uses
2. **Knowledge Retrieval Patterns**: How effectively the agent retrieves and filters relevant information
3. **Fact Verification Patterns**: How the agent verifies retrieved facts against the question
4. **Answer Extraction Patterns**: How the agent extracts concise answers from retrieved content

Provide your analysis in a structured format:
- **Successful Search Patterns**: What search strategies worked well
- **Failed Search Patterns**: What search approaches led to failure
- **Knowledge Gap Indicators**: When the agent correctly identified vs. missed knowledge gaps
- **Factual Experience Summary**: Concise, reusable factual knowledge patterns
"""

FACTUAL_USER_TEMPLATE = """Analyze the following execution runs for a factual QA task.

Dataset: {dataset}
Question: {question}
Ground Truth Answer: {ground_truth}

## Successful Runs (score >= threshold):
{success_runs}

## Failed Runs (score < threshold):
{failure_runs}

Please extract factual experience patterns. Focus on:
1. What search queries and strategies led to correct answers?
2. What information retrieval patterns were effective vs. ineffective?
3. What reusable factual knowledge patterns can be derived?

Keep your analysis concise and actionable.
"""


# ─── SOP System Experience Summary Prompt ────────────────────────────

SOP_SYSTEM_PROMPT = """You are an expert at analyzing multi-agent system execution traces for reasoning and problem-solving tasks.

Your task is to analyze successful and failed runs to extract **SOP system-level experience (SOP系统层经验)**.

SOP system experience focuses on:
1. **Reasoning Workflow Patterns**: Step-by-step reasoning strategies that work for different problem types
2. **Tool Usage SOPs**: When and how to use search/research tools during reasoning
3. **Error Detection & Recovery**: How the agent identifies errors and self-corrects
4. **Decision Tree Patterns**: Key decision points in the solving process

Provide your analysis in a structured format:
- **Effective Reasoning Workflows**: Step-by-step strategies that led to correct answers
- **Failed Reasoning Patterns**: Reasoning approaches that led to wrong answers
- **Tool Usage Recommendations**: When search/research should be invoked during reasoning
- **SOP Experience Summary**: Concise, reusable standard operating procedures
"""

SOP_USER_TEMPLATE = """Analyze the following execution runs for a reasoning/problem-solving task.

Dataset: {dataset}
Question: {question}
Ground Truth Answer: {ground_truth}

## Successful Runs (score >= threshold):
{success_runs}

## Failed Runs (score < threshold):
{failure_runs}

Please extract SOP system-level experience patterns. Focus on:
1. What step-by-step reasoning workflows led to correct solutions?
2. How did the agent use tools (search/research) during reasoning?
3. What reusable standard operating procedures can be derived?

Keep your analysis concise and actionable.
"""


class QABenchSummaryAgent:
    """
    Summary agent that generates experience summaries from QA benchmark runs.

    Produces two types of experience:
    - Factual Experience (事实性经验): From TriviaQA/PopQA runs
    - SOP System Experience (SOP系统层经验): From GPQA/GSM8K/MATH runs
    """

    def __init__(
        self,
        base_url: str = SUMMARY_BASE_URL,
        api_key: str = SUMMARY_API_KEY,
        model: str = SUMMARY_MODEL,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def _format_runs(
        self, runs: List[Dict[str, Any]], include_stack: bool = True
    ) -> str:
        """Format run results for prompt inclusion."""
        if not runs:
            return "No runs in this category."

        parts = []
        for run in runs:
            run_text = (
                f"### Run {run.get('run_id', '?')} (Score: {run.get('score', 'N/A')})\n"
            )
            run_text += f"Prediction: {run.get('prediction', 'N/A')}\n"

            if include_stack and run.get("memory_stack_log"):
                stack_summary = []
                for entry in run["memory_stack_log"]:
                    action = entry.get("action", "unknown")
                    content = entry.get("content", "")
                    agent_type = entry.get("agent_type", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    prefix = f"[{action}]"
                    if agent_type:
                        prefix = f"[{action} -> {agent_type}]"
                    stack_summary.append(f"  - {prefix} {content}")
                run_text += "Memory Stack:\n" + "\n".join(stack_summary) + "\n"

            parts.append(run_text)

        return "\n".join(parts)

    def _get_prompts_for_type(self, experience_type: str):
        """Get the appropriate system and user prompt templates for the experience type."""
        if experience_type == "factual":
            return FACTUAL_SYSTEM_PROMPT, FACTUAL_USER_TEMPLATE
        else:
            return SOP_SYSTEM_PROMPT, SOP_USER_TEMPLATE

    def summarize_qa_runs(
        self,
        qa_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Summarize runs for a single QA sample.

        Args:
            qa_result: Benchmark result with runs, including "score" and "success" fields

        Returns:
            Summary dict with experience type classification
        """
        runs = qa_result.get("runs", [])
        question = qa_result.get("question", "")
        ground_truth = qa_result.get("ground_truth", "")
        dataset = qa_result.get("dataset", "")
        experience_type = qa_result.get("experience_type", "factual")

        success_runs = [r for r in runs if r.get("success", False)]
        failure_runs = [r for r in runs if not r.get("success", False)]

        success_text = self._format_runs(success_runs)
        failure_text = self._format_runs(failure_runs)

        system_prompt, user_template = self._get_prompts_for_type(experience_type)

        user_message = user_template.format(
            dataset=dataset,
            question=question[:2000],  # Truncate very long questions
            ground_truth=ground_truth,
            success_runs=success_text,
            failure_runs=failure_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=2000,
            )
            summary = response.choices[0].message.content.strip()
        except Exception as e:
            summary = f"Summary generation failed: {e}"

        return {
            "success_count": len(success_runs),
            "failure_count": len(failure_runs),
            "experience_type": experience_type,
            "summary": summary,
        }

    def _summarize_one(self, idx_and_result):
        """Worker function for summarizing a single result."""
        idx, total, qa_result = idx_and_result
        sample_id = qa_result.get("sample_id", "?")
        dataset = qa_result.get("dataset", "?")
        experience_type = qa_result.get("experience_type", "?")
        print(
            f"Summarizing [{idx+1}/{total}]: {sample_id} "
            f"(dataset={dataset}, type={experience_type})"
        )
        summary = self.summarize_qa_runs(qa_result)
        return {
            "sample_id": qa_result.get("sample_id", ""),
            "dataset": qa_result.get("dataset", ""),
            "experience_type": qa_result.get("experience_type", ""),
            "question": qa_result.get("question", ""),
            "ground_truth": qa_result.get("ground_truth", ""),
            **summary,
        }

    def summarize_batch(
        self,
        results: List[Dict[str, Any]],
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Summarize a batch of QA benchmark results.

        Args:
            results: List of benchmark results with evaluated runs
            concurrency: Number of parallel threads for summary generation

        Returns:
            List of summary dicts, each tagged with experience_type
        """
        total = len(results)
        work_items = [(idx, total, r) for idx, r in enumerate(results)]

        if concurrency <= 1:
            return [self._summarize_one(item) for item in work_items]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        summaries = []
        print(f"Summarizing with {concurrency} threads...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for item in work_items:
                future = executor.submit(self._summarize_one, item)
                future_to_idx[future] = item[0]

            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    summaries.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"Summary failed for index {idx}: {e}")

        return summaries
