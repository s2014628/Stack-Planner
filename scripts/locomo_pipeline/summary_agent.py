import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

SUMMARY_BASE_URL = os.getenv("SUMMARY_BASE_URL", "http://123.57.228.132:8285/api")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "sk-7374e2abda1141ffa4fe8eb01ae582f7")
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

EXPERIENCE_SYSTEM_PROMPT = """You are an expert at distilling answering experience from question-answering evaluation runs.
Given a question, the ground-truth answer, the relevant evidence dialogue snippets, and multiple execution runs (both successful and failed), your task is to produce a concise **experience note** that future agents can use to answer similar questions correctly.

Your output MUST be a JSON object with exactly these fields:
{
  "evidence_analysis": "<Why these specific dialogue turns are the key evidence for answering this question. 1-3 sentences.>",
  "effective_strategy": "<The reasoning strategy that led to correct answers in the successful runs. 1-3 sentences.>",
  "common_mistakes": "<Typical errors observed in the failed runs and why they happened. 1-3 sentences. Write 'None' if all runs succeeded.>",
  "experience_note": "<A concise, reusable instruction (2-4 sentences) that tells a future agent how to correctly answer this type of question given similar evidence.>"
}

IMPORTANT: Respond ONLY with the JSON object, no markdown fences, no extra text.
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

EXPERIENCE_USER_TEMPLATE = """Generate an experience note for the following question-answering task.

Question: {question}
Ground Truth Answer: {ground_truth}
Category: {category}

## Evidence Dialogue Snippets:
{evidence_text}

## Session Context for Evidence:
{session_context_text}

## Successful Runs ({success_count} runs, F1 >= threshold):
{success_runs}

## Failed Runs ({failure_count} runs, F1 < threshold):
{failure_runs}

Respond with a JSON object containing: evidence_analysis, effective_strategy, common_mistakes, experience_note.
"""


class SummaryAgent:

    def __init__(
        self,
        base_url: str = SUMMARY_BASE_URL,
        api_key: str = SUMMARY_API_KEY,
        model: str = SUMMARY_MODEL,
    ):
        if not (api_key or "").strip():
            raise ValueError(
                "SUMMARY_API_KEY is missing. Set it in scripts/locomo_pipeline/.env "
                "(copy from .env.example) or pass api_key=... to SummaryAgent."
            )
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    @staticmethod
    def _format_evidence_snippets(snippets: List[Dict[str, Any]]) -> str:
        if not snippets:
            return "No evidence snippets available."
        parts = []
        for snip in snippets:
            line = f"[{snip.get('dia_id', '?')}] Session {snip.get('session_num', '?')} ({snip.get('session_date_time', '')}) "
            line += f"{snip.get('speaker', '?')}: {snip.get('text', '')}"
            if snip.get("blip_caption"):
                line += f" [image: {snip['blip_caption']}]"
            parts.append(line)
        return "\n".join(parts)

    @staticmethod
    def _format_session_context(session_context: Dict[str, Any]) -> str:
        if not session_context:
            return "No session context available."
        parts = []
        for snum, ctx in sorted(session_context.items(), key=lambda x: int(x[0])):
            part = f"Session {snum}:"
            if "summary" in ctx:
                summary = ctx["summary"]
                if len(summary) > 400:
                    summary = summary[:400] + "..."
                part += f"\n  Summary: {summary}"
            if "events" in ctx:
                events = ctx["events"]
                if isinstance(events, dict):
                    for speaker, evts in events.items():
                        if speaker == "date":
                            continue
                        if evts:
                            part += f"\n  Events ({speaker}): {'; '.join(str(e) for e in evts[:5])}"
            parts.append(part)
        return "\n".join(parts)

    def _format_runs(
        self, runs: List[Dict[str, Any]], include_stack: bool = True
    ) -> str:
        if not runs:
            return "No runs in this category."

        parts = []
        for run in runs:
            run_text = (
                f"### Run {run.get('run_id', '?')} (F1: {run.get('f1_score', 'N/A')})\n"
            )
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

    def generate_experience(
        self,
        qa_result: Dict[str, Any],
        evidence_snippets: List[Dict[str, Any]],
        evidence_session_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate a structured experience note for a single QA item.

        Unlike ``summarize_qa_runs`` which produces a free-form analysis,
        this method returns a structured JSON experience containing:
        evidence_analysis, effective_strategy, common_mistakes, experience_note.
        """
        runs = qa_result.get("runs", [])
        question = qa_result.get("question", "")
        ground_truth = qa_result.get("ground_truth", "")
        category = qa_result.get("category", 0)

        success_runs = [r for r in runs if r.get("success", False)]
        failure_runs = [r for r in runs if not r.get("success", False)]

        evidence_text = self._format_evidence_snippets(evidence_snippets)
        session_context_text = self._format_session_context(evidence_session_context)

        user_message = EXPERIENCE_USER_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            category=category,
            evidence_text=evidence_text,
            session_context_text=session_context_text,
            success_count=len(success_runs),
            success_runs=self._format_runs(success_runs, include_stack=False),
            failure_count=len(failure_runs),
            failure_runs=self._format_runs(failure_runs, include_stack=False),
        )

        import json as _json

        raw = ""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": EXPERIENCE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
                max_tokens=1500,
            )
            raw = response.choices[0].message.content.strip()
            experience = _json.loads(raw)
        except _json.JSONDecodeError:
            experience = {
                "evidence_analysis": "",
                "effective_strategy": "",
                "common_mistakes": "",
                "experience_note": raw,
            }
        except Exception as e:
            experience = {
                "evidence_analysis": "",
                "effective_strategy": "",
                "common_mistakes": "",
                "experience_note": f"Experience generation failed: {e}",
            }

        return {
            "success_count": len(success_runs),
            "failure_count": len(failure_runs),
            **experience,
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

    def _generate_experience_one(self, idx_and_args):
        idx, total, qa_result, evidence_snippets, evidence_session_context = (
            idx_and_args
        )
        sample_id = qa_result.get("sample_id", "?")
        qa_index = qa_result.get("qa_index", "?")
        print(f"Generating experience [{idx+1}/{total}]: {sample_id} qa={qa_index}")
        experience = self.generate_experience(
            qa_result, evidence_snippets, evidence_session_context
        )
        return {
            "sample_id": qa_result.get("sample_id", ""),
            "qa_index": qa_result.get("qa_index", 0),
            "category": qa_result.get("category", 0),
            "question": qa_result.get("question", ""),
            "ground_truth": qa_result.get("ground_truth", ""),
            **experience,
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

    def generate_experience_batch(
        self,
        results: List[Dict[str, Any]],
        samples: List[Dict[str, Any]],
        concurrency: int = 1,
    ) -> List[Dict[str, Any]]:
        """Generate experience notes for a batch of evaluated QA results.

        ``samples`` should be the original prepared samples (from
        ``extract_qa_samples``) so that evidence_snippets and
        evidence_session_context can be looked up.
        """
        # Build a lookup from (sample_id, qa_index) -> sample
        sample_map: Dict[str, Dict[str, Any]] = {}
        for s in samples:
            key = f"{s['sample_id']}_{s['qa_index']}"
            sample_map[key] = s

        total = len(results)
        work_items = []
        for idx, r in enumerate(results):
            key = f"{r['sample_id']}_{r['qa_index']}"
            sample = sample_map.get(key, {})
            work_items.append(
                (
                    idx,
                    total,
                    r,
                    sample.get("evidence_snippets", []),
                    sample.get("evidence_session_context", {}),
                )
            )

        if concurrency <= 1:
            return [self._generate_experience_one(item) for item in work_items]

        from concurrent.futures import ThreadPoolExecutor, as_completed

        experiences: List[Dict[str, Any]] = []
        print(f"Generating experiences with {concurrency} threads...")
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {}
            for item in work_items:
                future = executor.submit(self._generate_experience_one, item)
                future_to_idx[future] = item[0]

            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    experiences.append(result)
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"Experience generation failed for index {idx}: {e}")

        return experiences
