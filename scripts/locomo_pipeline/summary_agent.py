import json
import os
from typing import Any, Dict, List

from openai import OpenAI

DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "http://123.57.228.132:8285/api")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v3.2-20251201-160k-local")

SUMMARY_SYSTEM_PROMPT = """You are an expert at analyzing multi-agent system execution traces.
Your task is to compare successful and failed execution runs and identify patterns.

For each group of runs (success/failure), analyze the memory_stack execution logs and identify:
1. Common patterns in how the agent approached the problem
2. Key decision points that led to success or failure
3. Specific actions or reasoning strategies that were effective or ineffective
4. Recommendations for improvement

Provide your analysis in a structured format with clear sections for:
- Success Patterns: What strategies worked well
- Failure Patterns: What went wrong and why
- Key Differences: Critical differences between successful and failed runs
- Recommendations: Actionable suggestions for improvement
"""

SUMMARY_USER_TEMPLATE = """Analyze the following execution runs for a question-answering task.

Question: {question}
Ground Truth Answer: {ground_truth}
Category: {category}

## Successful Runs (F1 >= threshold):
{success_runs}

## Failed Runs (F1 < threshold):
{failure_runs}

Please provide a structured analysis of the success and failure patterns.
Keep your analysis concise but insightful. Focus on actionable patterns.
"""


class SummaryAgent:

    def __init__(
        self,
        base_url: str = DEEPSEEK_BASE_URL,
        api_key: str = DEEPSEEK_API_KEY,
        model: str = DEEPSEEK_MODEL,
    ):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def _format_runs(self, runs: List[Dict[str, Any]], include_stack: bool = True) -> str:
        if not runs:
            return "No runs in this category."

        parts = []
        for run in runs:
            run_text = f"### Run {run.get('run_id', '?')} (F1: {run.get('f1_score', 'N/A')})\n"
            run_text += f"Prediction: {run.get('prediction', 'N/A')}\n"

            if include_stack and run.get("memory_stack_log"):
                stack_summary = []
                for entry in run["memory_stack_log"]:
                    action = entry.get("action", "unknown")
                    content = entry.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    stack_summary.append(f"  - [{action}] {content}")
                run_text += "Memory Stack:\n" + "\n".join(stack_summary) + "\n"

            parts.append(run_text)

        return "\n".join(parts)

    def summarize_qa_runs(
        self,
        qa_result: Dict[str, Any],
    ) -> Dict[str, str]:
        runs = qa_result.get("runs", [])
        question = qa_result.get("question", "")
        ground_truth = qa_result.get("ground_truth", "")
        category = qa_result.get("category", 0)

        success_runs = [r for r in runs if r.get("success", False)]
        failure_runs = [r for r in runs if not r.get("success", False)]

        success_text = self._format_runs(success_runs)
        failure_text = self._format_runs(failure_runs)

        user_message = SUMMARY_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            category=category,
            success_runs=success_text,
            failure_runs=failure_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
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
            "summary": summary,
        }

    def summarize_batch(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        summaries = []
        for idx, qa_result in enumerate(results):
            print(f"Summarizing [{idx+1}/{len(results)}]: {qa_result.get('sample_id', '?')} qa={qa_result.get('qa_index', '?')}")
            summary = self.summarize_qa_runs(qa_result)
            summaries.append({
                "sample_id": qa_result.get("sample_id", ""),
                "qa_index": qa_result.get("qa_index", 0),
                "category": qa_result.get("category", 0),
                "question": qa_result.get("question", ""),
                "ground_truth": qa_result.get("ground_truth", ""),
                **summary,
            })
        return summaries
