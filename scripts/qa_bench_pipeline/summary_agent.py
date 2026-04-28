"""
QA Bench Pipeline - Summary Agent

Two-stage experience extraction pipeline:

  Stage 1 (Trajectory Summary): Compress each run's memory_stack_log into a
    concise step-by-step summary, flagging detours and errors.

  Stage 2 (Semantic Advantage): Compare correct vs. wrong trajectory summaries
    to extract structured, generalizable experience:
      - problem_type      short label for clustering / codebook
      - problem_addressed class of challenge this experience solves
      - practice          actionable steps an agent can follow
      - lessons_learned   key differences between correct and wrong attempts

Experience types:
  - factual : TriviaQA / PopQA  (retrieval, search, verification)
  - sop     : GPQA / GSM8K / MATH  (reasoning workflows, tools, recovery)
"""

import json
import os
import re
from typing import Any, Dict, List, Tuple

from openai import OpenAI

SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "https://openrouter.ai/api/v1")
SUMMARY_API_KEY = os.getenv(
    "SUMMARY_API_KEY",
    "",
)
SUMMARY_MODEL = os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2")


# ─── Stage 1: Trajectory Summary ───────────────────────────────────────────

TRAJECTORY_SYSTEM_PROMPT = (
    "You are an expert at concisely summarizing agent execution trajectories "
    "step-by-step."
)

TRAJECTORY_USER_TEMPLATE = """An agent produced the following trajectory while solving a problem.
Please summarize the trajectory step-by-step:

1. For each step, describe what action is being taken and what outcome it produced.
2. Given the evaluation result and the correct answer, identify any steps that \
represent detours, errors, or backtracking.
3. Maintain the core outcome of each step, even if it was part of a flawed process.

<question>{question}</question>
<trajectory>{trajectory}</trajectory>
<evaluation>{evaluation}</evaluation>
<groundtruth>{ground_truth}</groundtruth>

Return ONLY a numbered list. Each line: "Step N [action_type]: description."
"""


# ─── Stage 2: Semantic Advantage Extraction ────────────────────────────────

_STAGE2_JSON_RULE = """
Your output MUST be a single JSON object with exactly these keys (no markdown fences, \
no extra text):
{
  "problem_type": "<short English label, e.g. 'entity attribute lookup', \
'multi-step arithmetic reasoning'>",
  "problem_addressed": "<1-2 sentences: the class of challenge this experience \
solves, generalized beyond this specific question>",
  "practice": "<2-5 actionable steps an agent should follow for this type of problem>",
  "lessons_learned": "<key differences between correct and wrong attempts; \
write 'N/A' if only successful runs exist>"
}
"""

FACTUAL_SYSTEM_PROMPT = (
    "You are an expert at extracting generalizable experience from agent "
    "trajectory comparisons for factual question-answering tasks.\n"
    + _STAGE2_JSON_RULE
)

FACTUAL_USER_TEMPLATE = """Analyze the following trajectory summaries for a factual QA task.

Dataset: {dataset}
Question: {question}
Ground Truth: {ground_truth}

{attempts}

Extract generalizable factual experience. Focus on:
- Search / retrieval strategy that distinguished correct from wrong attempts
- How facts were verified and answers extracted
- Reusable patterns for similar factual questions
"""

SOP_SYSTEM_PROMPT = (
    "You are an expert at extracting generalizable experience from agent "
    "trajectory comparisons for reasoning and problem-solving tasks.\n"
    + _STAGE2_JSON_RULE
)

SOP_USER_TEMPLATE = """Analyze the following trajectory summaries for a reasoning/problem-solving task.

Dataset: {dataset}
Question: {question}
Ground Truth: {ground_truth}

{attempts}

Extract generalizable SOP experience. Focus on:
- Reasoning workflow that distinguished correct from wrong attempts
- How and when tools (search/research) should be invoked
- Reusable step-by-step procedure for similar problems
"""


# ─── Helpers ───────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> Dict[str, str]:
    """Parse JSON from LLM output, stripping markdown fences if present."""
    text = raw.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
    except (json.JSONDecodeError, AttributeError):
        pass
    return {}


def _compose_experience_summary(
    problem_addressed: str, practice: str, lessons_learned: str
) -> str:
    """Compose a human-readable summary from structured fields."""
    parts = []
    if problem_addressed:
        parts.append(f"**Problem addressed:** {problem_addressed}")
    if practice:
        parts.append(f"**Practice:** {practice}")
    if lessons_learned and lessons_learned.strip().upper() != "N/A":
        parts.append(f"**Lessons learned:** {lessons_learned}")
    return "\n\n".join(parts)


# ─── Main Agent Class ───────────────────────────────────────────────────────

class QABenchSummaryAgent:
    """Two-stage experience extraction from QA benchmark runs.

    Stage 1 (``_summarize_trajectory``): per-run step-by-step compression.
    Stage 2 (``_extract_semantic_advantage``): cross-run structured extraction.
    """

    def __init__(
        self,
        base_url: str = SUMMARY_BASE_URL,
        api_key: str = SUMMARY_API_KEY,
        model: str = SUMMARY_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    # ── Formatting helpers ───────────────────────────────────────────

    @staticmethod
    def _format_trajectory(run: Dict[str, Any]) -> str:
        """Serialize memory_stack_log into readable trajectory text."""
        steps = []
        for entry in run.get("memory_stack_log", []):
            action = entry.get("action", "unknown")
            agent_type = entry.get("agent_type", "")
            content = entry.get("content", "")
            if len(content) > 400:
                content = content[:400] + "..."
            label = (
                f"[{action} -> {agent_type}]" if agent_type else f"[{action}]"
            )
            steps.append(f"{label} {content}")
        return "\n".join(steps) if steps else "(empty trajectory)"

    @staticmethod
    def _format_runs_legacy(
        runs: List[Dict[str, Any]], include_stack: bool = True
    ) -> str:
        """Legacy run formatter (kept for backward-compat callers)."""
        if not runs:
            return "No runs in this category."
        parts = []
        for run in runs:
            text = f"### Run {run.get('run_id', '?')} (Score: {run.get('score', 'N/A')})\n"
            text += f"Prediction: {run.get('prediction', 'N/A')}\n"
            if include_stack and run.get("memory_stack_log"):
                stack = []
                for entry in run["memory_stack_log"]:
                    action = entry.get("action", "unknown")
                    agent_type = entry.get("agent_type", "")
                    content = entry.get("content", "")
                    if len(content) > 300:
                        content = content[:300] + "..."
                    prefix = (
                        f"[{action} -> {agent_type}]" if agent_type else f"[{action}]"
                    )
                    stack.append(f"  - {prefix} {content}")
                text += "Memory Stack:\n" + "\n".join(stack) + "\n"
            parts.append(text)
        return "\n".join(parts)

    # ── Stage 1 ─────────────────────────────────────────────────────

    def _summarize_trajectory(
        self, qa_result: Dict[str, Any], run: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Stage 1: compress one run's trajectory into a step-by-step summary.

        Returns a dict with run metadata and a ``steps_summary`` string.
        """
        question = qa_result.get("question", "")
        ground_truth = qa_result.get("ground_truth", "")
        trajectory = self._format_trajectory(run)
        evaluation = "correct" if run.get("success", False) else "wrong"

        user_message = TRAJECTORY_USER_TEMPLATE.format(
            question=question[:1500],
            trajectory=trajectory,
            evaluation=evaluation,
            ground_truth=ground_truth,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": TRAJECTORY_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=800,
            )
            steps_summary = response.choices[0].message.content.strip()
        except Exception as e:
            steps_summary = f"(Trajectory summarization failed: {e})"

        return {
            "run_id": run.get("run_id", "?"),
            "evaluation": evaluation,
            "prediction": run.get("prediction", ""),
            "score": run.get("score", 0.0),
            "steps_summary": steps_summary,
        }

    # ── Stage 2 ─────────────────────────────────────────────────────

    def _extract_semantic_advantage(
        self,
        qa_result: Dict[str, Any],
        trajectory_summaries: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Stage 2: extract structured experience from trajectory summaries.

        Returns a dict with problem_type, problem_addressed, practice,
        lessons_learned.
        """
        question = qa_result.get("question", "")
        ground_truth = qa_result.get("ground_truth", "")
        dataset = qa_result.get("dataset", "")
        experience_type = qa_result.get("experience_type", "factual")

        attempts_parts = []
        for ts in trajectory_summaries:
            label = ts["evaluation"].capitalize()
            attempts_parts.append(
                f"Attempt (Run {ts['run_id']}, Result: {label}):\n{ts['steps_summary']}"
            )
        attempts_text = "\n\n".join(attempts_parts) if attempts_parts else "(no attempts)"

        if experience_type == "factual":
            system_prompt, user_template = FACTUAL_SYSTEM_PROMPT, FACTUAL_USER_TEMPLATE
        else:
            system_prompt, user_template = SOP_SYSTEM_PROMPT, SOP_USER_TEMPLATE

        user_message = user_template.format(
            dataset=dataset,
            question=question[:2000],
            ground_truth=ground_truth,
            attempts=attempts_text,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.3,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return {
                "problem_type": "",
                "problem_addressed": "",
                "practice": f"(Extraction failed: {e})",
                "lessons_learned": "",
            }

        data = _parse_json_response(raw)
        return {
            "problem_type": data.get("problem_type", "").strip(),
            "problem_addressed": data.get("problem_addressed", "").strip(),
            "practice": data.get("practice", "").strip(),
            "lessons_learned": data.get("lessons_learned", "").strip(),
        }

    # ── Full two-stage pipeline for one sample ───────────────────────

    def summarize_qa_runs(self, qa_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run Stage 1 + Stage 2 for a single QA sample.

        Returns a summary dict containing structured experience fields
        and ``trajectory_summaries`` (Stage 1 output, to be saved separately
        by the caller as ``trajectory_summaries.json``).
        """
        runs = qa_result.get("runs", [])
        success_runs = [r for r in runs if r.get("success", False)]
        failure_runs = [r for r in runs if not r.get("success", False)]

        # Stage 1
        trajectory_summaries = [
            self._summarize_trajectory(qa_result, r) for r in runs
        ]

        # Stage 2
        experience = self._extract_semantic_advantage(qa_result, trajectory_summaries)

        summary = _compose_experience_summary(
            experience["problem_addressed"],
            experience["practice"],
            experience["lessons_learned"],
        )

        return {
            "success_count": len(success_runs),
            "failure_count": len(failure_runs),
            "experience_type": qa_result.get("experience_type", "factual"),
            "problem_type": experience["problem_type"],
            "problem_addressed": experience["problem_addressed"],
            "practice": experience["practice"],
            "lessons_learned": experience["lessons_learned"],
            "summary": summary,
            "trajectory_summaries": trajectory_summaries,
        }

    def _summarize_one(
        self, idx_and_result: Tuple[int, int, Dict[str, Any]]
    ) -> Dict[str, Any]:
        idx, total, qa_result = idx_and_result
        sample_id = qa_result.get("sample_id", "?")
        dataset = qa_result.get("dataset", "?")
        experience_type = qa_result.get("experience_type", "?")
        print(
            f"Summarizing [{idx+1}/{total}]: {sample_id} "
            f"(dataset={dataset}, type={experience_type})"
        )
        result = self.summarize_qa_runs(qa_result)
        return {
            "sample_id": qa_result.get("sample_id", ""),
            "dataset": qa_result.get("dataset", ""),
            "experience_type": qa_result.get("experience_type", ""),
            "question": qa_result.get("question", ""),
            "ground_truth": qa_result.get("ground_truth", ""),
            **result,
        }

    # ── Per-run mode (for summarize_all_runs / --per-run-summary) ────

    def summarize_single_run(
        self,
        qa_result: Dict[str, Any],
        run: Dict[str, Any],
    ) -> Dict[str, str]:
        """Stage 1 + lightweight Stage 2 for a single run.

        Used by ``summarize_all_runs`` (``--per-run-summary`` mode).
        Since there is only one run there is no cross-run comparison;
        ``lessons_learned`` will be "N/A".

        Returns dict with problem_type, problem_addressed, practice,
        lessons_learned, summary, steps_summary, evaluation.
        """
        ts = self._summarize_trajectory(qa_result, run)
        experience = self._extract_semantic_advantage(qa_result, [ts])
        return {
            "problem_type": experience["problem_type"],
            "problem_addressed": experience["problem_addressed"],
            "practice": experience["practice"],
            "lessons_learned": experience["lessons_learned"],
            "summary": _compose_experience_summary(
                experience["problem_addressed"],
                experience["practice"],
                experience["lessons_learned"],
            ),
            "steps_summary": ts["steps_summary"],
            "evaluation": ts["evaluation"],
        }

    def summarize_all_runs(
        self,
        results: List[Dict[str, Any]],
        concurrency: int = 1,
    ) -> Dict[str, Dict[str, str]]:
        """Generate per-run structured summaries for all runs.

        Returns dict mapping ``"{sample_id}_run_{run_id}"`` -> summary dict
        with problem_type, problem_addressed, practice, lessons_learned,
        steps_summary, evaluation.
        """
        work_items = []
        for qa_result in results:
            sample_id = qa_result.get("sample_id", "")
            for run in qa_result.get("runs", []):
                run_id = run.get("run_id", "?")
                key = f"{sample_id}_run_{run_id}"
                work_items.append((key, qa_result, run))

        total = len(work_items)
        print(f"Generating per-run summaries for {total} runs...")
        run_summaries: Dict[str, Dict[str, str]] = {}

        def _do_one(item: tuple) -> Tuple[str, Dict[str, str]]:
            key, qa_result, run = item
            return key, self.summarize_single_run(qa_result, run)

        if concurrency <= 1:
            for i, item in enumerate(work_items):
                key, summary = _do_one(item)
                run_summaries[key] = summary
                print(f"  [{i+1}/{total}] {key}")
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed

            print(f"  Using {concurrency} threads...")
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                future_to_key = {
                    executor.submit(_do_one, item): item[0] for item in work_items
                }
                done = 0
                for future in as_completed(future_to_key):
                    done += 1
                    key = future_to_key[future]
                    try:
                        _, summary = future.result()
                        run_summaries[key] = summary
                        print(f"  [{done}/{total}] {key}")
                    except Exception as e:
                        run_summaries[key] = {
                            "problem_type": "",
                            "problem_addressed": "",
                            "practice": "",
                            "lessons_learned": "",
                            "summary": f"Per-run summary failed: {e}",
                            "steps_summary": "",
                            "evaluation": "",
                        }
                        print(f"  [{done}/{total}] {key} FAILED: {e}")

        return run_summaries

    # ── Batch mode ───────────────────────────────────────────────────

    def summarize_batch(
        self,
        results: List[Dict[str, Any]],
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Summarize a batch of QA benchmark results (two-stage pipeline).

        Returns a list of summary dicts. Each dict contains
        ``trajectory_summaries`` (Stage 1 output) which the caller should
        extract and save as a separate ``trajectory_summaries.json`` file.
        """
        total = len(results)
        work_items = [(idx, total, r) for idx, r in enumerate(results)]

        if concurrency <= 1:
            return [self._summarize_one(item) for item in work_items]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        summaries = []
        print(f"Summarizing with {concurrency} threads...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(self._summarize_one, item): item[0]
                for item in work_items
            }
            for future in as_completed(future_to_idx):
                try:
                    summaries.append(future.result())
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"Summary failed for index {idx}: {e}")

        return summaries
