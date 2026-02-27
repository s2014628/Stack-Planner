import os
from typing import Any, Dict, List

from openai import OpenAI

SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "http://123.57.228.132:8285/api")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "")
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek-v3.2-20251201-160k-local")

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
        base_url: str = SUMMARY_BASE_URL,
        api_key: str = SUMMARY_API_KEY,
        model: str = SUMMARY_MODEL,
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

    def _summarize_one(self, idx_and_result):
        idx, total, qa_result = idx_and_result
        sample_id = qa_result.get("sample_id", "?")
        qa_index = qa_result.get("qa_index", "?")
        print(f"Summarizing [{idx+1}/{total}]: {sample_id} qa={qa_index}")
        summary = self.summarize_qa_runs(qa_result)
        return {
            "sample_id": qa_result.get("sample_id", ""),
            "qa_index": qa_result.get("qa_index", 0),
            "category": qa_result.get("category", 0),
            "question": qa_result.get("question", ""),
            "ground_truth": qa_result.get("ground_truth", ""),
            **summary,
        }

    def summarize_batch(
        self,
        results: List[Dict[str, Any]],
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
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
