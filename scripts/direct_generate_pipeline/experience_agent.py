"""
Direct Generate Pipeline - Experience Agent

Step 2: Given reasoning + prediction (from SolverAgent) and the ground truth,
generate structured experience fields:
  - problem_type      : short label for clustering / codebook
  - problem_addressed : the class of challenge this experience solves
  - practice          : actionable steps an agent should follow
  - lessons_learned   : key differences between correct and wrong attempts
                        (N/A when all runs succeed or only one run)

One-shot examples are taken from existing per_run_summaries.json outputs.
Prompts mirror qa_bench_pipeline/summary_agent.py Stage-2 style.
"""

import json
import os
import re
from typing import Any, Dict, List

from openai import OpenAI

EXPERIENCE_BASE_URL = os.getenv("SUMMARY_BASE_URL", "https://openrouter.ai/api/v1")
EXPERIENCE_API_KEY = os.getenv(
    "SUMMARY_API_KEY",
    "",
)
EXPERIENCE_MODEL = os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2")

_FACTUAL_DATASETS = {"triviaqa", "popqa"}
_SOP_DATASETS = {"gpqa", "gsm8k", "math"}


# ─── JSON output rule (same as summary_agent.py Stage 2) ─────────────────────

_JSON_RULE = """\
Your output MUST be a single JSON object with exactly these keys \
(no markdown fences, no extra text):
{
  "problem_type": "<short English label, e.g. 'entity attribute lookup', \
'multi-step arithmetic reasoning'>",
  "problem_addressed": "<1-2 sentences: the class of challenge this experience \
solves, generalized beyond this specific question>",
  "practice": "<2-5 actionable steps an agent should follow for this type of problem>",
  "lessons_learned": "<key differences between correct and wrong attempts; \
write 'N/A' if only successful runs exist>"
}"""


# ─── System prompts ───────────────────────────────────────────────────────────

FACTUAL_SYSTEM_PROMPT = (
    "You are an expert at extracting generalizable experience from agent "
    "reasoning traces for factual question-answering tasks.\n" + _JSON_RULE
)

SOP_SYSTEM_PROMPT = (
    "You are an expert at extracting generalizable experience from agent "
    "reasoning traces for reasoning and problem-solving tasks.\n" + _JSON_RULE
)


# ─── One-shot examples (from existing per_run_summaries.json) ─────────────────

# SOP one-shot — math (correct attempt, lessons_learned = N/A)
_SOP_ONE_SHOT_USER = """\
Analyze the following reasoning attempts for a math problem.

Dataset: math
Question: How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?
Ground Truth: 2

Attempt (Run 1, Result: Correct):
Reasoning: I need to find vertical asymptotes of the rational function y = 2/(x\u00b2 + x - 6). Vertical asymptotes occur where the denominator equals zero (and numerator is non-zero). So I need to solve x\u00b2 + x - 6 = 0. This is a quadratic equation that can be factored: x\u00b2 + x - 6 = (x + 3)(x - 2) = 0. The solutions are x = -3 and x = 2. Both are valid vertical asymptotes because the numerator (2) is non-zero at these points. Therefore, the graph has two vertical asymptotes.
Prediction: 2

Extract generalizable SOP experience. Focus on:
- Reasoning workflow that distinguished correct from wrong attempts
- How and when tools (search/research) should be invoked
- Reusable step-by-step procedure for similar problems"""

_SOP_ONE_SHOT_ASSISTANT = """\
{
  "problem_type": "rational function asymptote analysis",
  "problem_addressed": "Determining the number of vertical asymptotes for a rational function by correctly identifying where the denominator is zero and the numerator is non-zero.",
  "practice": "1. Identify the rational function's denominator and set it equal to zero. 2. Solve the denominator equation to find all candidate x-values. 3. Verify the numerator is non-zero at each candidate value. 4. Count the verified values to determine the number of vertical asymptotes.",
  "lessons_learned": "N/A"
}"""

# SOP one-shot — math (mixed correct/wrong, has lessons_learned)
_SOP_MIXED_ONE_SHOT_USER = """\
Analyze the following reasoning attempts for a math problem.

Dataset: math
Question: Find the shortest angle bisector in a triangle with sides 10, 10, 12.
Ground Truth: \\frac{20\\sqrt{2}}{3}

Attempt (Run 1, Result: Correct):
Reasoning: The triangle is isosceles with sides 10, 10, 12. Using the angle bisector length formula: t_a = (2bc cos(A/2))/(b+c). First compute the angles using Law of Cosines. For the base angle: cos(B) = (100+144-100)/(2*10*12) = 144/240 = 0.6, so B = 53.13\u00b0. Bisector from B: t_b = 2*10*12*cos(26.57\u00b0)/(10+12) = 240*0.894/22 = 9.75. Repeat for all three bisectors, compare, and identify minimum.
Prediction: \\frac{20\\sqrt{2}}{3}

Attempt (Run 2, Result: Wrong):
Reasoning: Tried to use coordinate geometry instead of the bisector formula, placed vertices at coordinates, computed midpoints of angle bisectors incorrectly.
Prediction: 10

Extract generalizable SOP experience. Focus on:
- Reasoning workflow that distinguished correct from wrong attempts
- Reusable step-by-step procedure for similar problems"""

_SOP_MIXED_ONE_SHOT_ASSISTANT = """\
{
  "problem_type": "geometry with angle bisector optimization",
  "problem_addressed": "Finding the shortest among multiple possible geometric construct lengths (e.g., angle bisectors) in a given triangle, requiring systematic computation and comparison.",
  "practice": "1. Identify the triangle type and known lengths. 2. Compute all relevant angles using the Law of Cosines. 3. Apply the angle bisector length formula for each vertex. 4. Compare the computed lengths to determine the minimum. 5. Simplify the result to radical form if required.",
  "lessons_learned": "The wrong attempt switched to coordinate geometry unnecessarily, causing errors in bisector calculation. The correct approach applies the standard angle bisector formula directly after finding angles via Law of Cosines. Stick to the most direct analytical method rather than coordinate-based detours."
}"""

# Factual one-shot — triviaqa (correct, N/A)
_FACTUAL_ONE_SHOT_USER = """\
Analyze the following reasoning attempts for a triviaqa problem.

Dataset: triviaqa
Question: Which American-born Sinclair won the Nobel Prize for Literature in 1930?
Ground Truth: Sinclair Lewis

Attempt (Run 1, Result: Correct):
Reasoning: The Nobel Prize for Literature in 1930 was awarded to Sinclair Lewis, the American novelist known for works like 'Main Street' and 'Babbitt'. He was the first American to win the Nobel Prize in Literature.
Prediction: Sinclair Lewis

Extract generalizable factual experience. Focus on:
- Search / retrieval strategy that distinguished correct from wrong attempts
- How facts were verified and answers extracted
- Reusable patterns for similar factual questions"""

_FACTUAL_ONE_SHOT_ASSISTANT = """\
{
  "problem_type": "Nobel Prize laureate identification",
  "problem_addressed": "Identifying a specific historical figure associated with a named award, year, and category, requiring precise factual recall or targeted search.",
  "practice": "1. Extract the key identifying attributes from the question (award name, year, category, nationality). 2. Retrieve the specific award record for the given year. 3. Cross-check the retrieved name against any constraints (e.g., 'American-born', 'Sinclair'). 4. Return the full name of the matching person.",
  "lessons_learned": "N/A"
}"""


# ─── User prompt templates ────────────────────────────────────────────────────

_SOP_USER_TEMPLATE = """\
Analyze the following reasoning attempts for a {dataset} problem.

Dataset: {dataset}
Question: {question}
Ground Truth: {ground_truth}

{attempts}

Extract generalizable SOP experience. Focus on:
- Reasoning workflow that distinguished correct from wrong attempts
- How and when tools (search/research) should be invoked
- Reusable step-by-step procedure for similar problems"""

_FACTUAL_USER_TEMPLATE = """\
Analyze the following reasoning attempts for a {dataset} problem.

Dataset: {dataset}
Question: {question}
Ground Truth: {ground_truth}

{attempts}

Extract generalizable factual experience. Focus on:
- Search / retrieval strategy that distinguished correct from wrong attempts
- How facts were verified and answers extracted
- Reusable patterns for similar factual questions"""


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _parse_json_response(raw: str) -> Dict[str, str]:
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
    parts = []
    if problem_addressed:
        parts.append(f"**Problem addressed:** {problem_addressed}")
    if practice:
        parts.append(f"**Practice:** {practice}")
    if lessons_learned and lessons_learned.strip().upper() != "N/A":
        parts.append(f"**Lessons learned:** {lessons_learned}")
    return "\n\n".join(parts)


def _format_attempts(runs: List[Dict[str, Any]]) -> str:
    """Format a list of run dicts into the attempts text for the prompt."""
    parts = []
    for run in runs:
        run_id = run.get("run_id", "?")
        result = "Correct" if run.get("success", False) else "Wrong"
        reasoning = run.get("reasoning", "")
        prediction = run.get("prediction", "")
        parts.append(
            f"Attempt (Run {run_id}, Result: {result}):\n"
            f"Reasoning: {reasoning}\n"
            f"Prediction: {prediction}"
        )
    return "\n\n".join(parts) if parts else "(no attempts)"


# ─── Experience Agent ─────────────────────────────────────────────────────────

class ExperienceAgent:
    """Generate structured experience fields from a set of solved runs.

    Uses one-shot prompting with examples from existing pipeline outputs.
    Mirrors summary_agent.py Stage-2 logic but operates on direct
    reasoning traces instead of memory_stack_log trajectories.
    """

    def __init__(
        self,
        base_url: str = EXPERIENCE_BASE_URL,
        api_key: str = EXPERIENCE_API_KEY,
        model: str = EXPERIENCE_MODEL,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def _pick_one_shot(self, dataset: str, has_failures: bool):
        """Pick appropriate one-shot example based on dataset and outcome mix."""
        if dataset in _FACTUAL_DATASETS:
            return _FACTUAL_ONE_SHOT_USER, _FACTUAL_ONE_SHOT_ASSISTANT
        # SOP datasets
        if has_failures:
            return _SOP_MIXED_ONE_SHOT_USER, _SOP_MIXED_ONE_SHOT_ASSISTANT
        return _SOP_ONE_SHOT_USER, _SOP_ONE_SHOT_ASSISTANT

    def generate(
        self,
        sample: Dict[str, Any],
        runs: List[Dict[str, Any]],
    ) -> Dict[str, str]:
        """Generate structured experience for a sample given its solved runs.

        Args:
            sample: Original QA sample dict (question, ground_truth, dataset, …).
            runs: List of solved+evaluated run dicts, each with:
                  reasoning, prediction, success, run_id.

        Returns:
            Dict with problem_type, problem_addressed, practice,
            lessons_learned, experience_summary.
        """
        dataset = sample.get("dataset", "").lower()
        question = sample.get("question", "")
        ground_truth = sample.get("ground_truth", "")

        has_failures = any(not r.get("success", False) for r in runs)
        one_shot_user, one_shot_assistant = self._pick_one_shot(dataset, has_failures)

        attempts_text = _format_attempts(runs)

        if dataset in _FACTUAL_DATASETS:
            system_prompt = FACTUAL_SYSTEM_PROMPT
            user_template = _FACTUAL_USER_TEMPLATE
        else:
            system_prompt = SOP_SYSTEM_PROMPT
            user_template = _SOP_USER_TEMPLATE

        user_message = user_template.format(
            dataset=dataset,
            question=question[:2000],
            ground_truth=ground_truth,
            attempts=attempts_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": one_shot_user},
            {"role": "assistant", "content": one_shot_assistant},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.3,
                max_tokens=1000,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return {
                "problem_type": "",
                "problem_addressed": "",
                "practice": f"(Experience generation failed: {e})",
                "lessons_learned": "",
                "experience_summary": "",
            }

        data = _parse_json_response(raw)
        problem_type = data.get("problem_type", "").strip()
        problem_addressed = data.get("problem_addressed", "").strip()
        practice = data.get("practice", "").strip()
        lessons_learned = data.get("lessons_learned", "").strip()

        return {
            "problem_type": problem_type,
            "problem_addressed": problem_addressed,
            "practice": practice,
            "lessons_learned": lessons_learned,
            "experience_summary": _compose_experience_summary(
                problem_addressed, practice, lessons_learned
            ),
        }
