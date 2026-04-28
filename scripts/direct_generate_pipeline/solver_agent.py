"""
Direct Generate Pipeline - Solver Agent

Step 1: Directly call an LLM to solve a QA problem.
Returns reasoning + prediction WITHOUT revealing the ground truth.

One-shot examples are taken from existing pipeline outputs to ensure
consistent style with the rest of the experience data.
"""

import json
import os
import re
from typing import Any, Dict, Optional

from openai import OpenAI

SOLVER_BASE_URL = os.getenv("SUMMARY_BASE_URL", "https://openrouter.ai/api/v1")
SOLVER_API_KEY = os.getenv(
    "SUMMARY_API_KEY",
    "",
)
SOLVER_MODEL = os.getenv("SUMMARY_MODEL", "deepseek/deepseek-v3.2")


# ─── System prompt ────────────────────────────────────────────────────────────

SOLVER_SYSTEM_PROMPT = """\
You are an expert problem solver. Think through the problem step by step and \
give your final answer.

Your output MUST be a single JSON object with exactly these keys \
(no markdown fences, no extra text):
{
  "reasoning": "<step-by-step reasoning process>",
  "prediction": "<final answer only, concise>"
}"""


# ─── One-shot examples (from existing pipeline outputs) ───────────────────────

# SOP example — math (rational function vertical asymptote)
_SOP_ONE_SHOT_USER = """\
Solve the following math problem.

<question>How many vertical asymptotes does the graph of $y=\\frac{2}{x^2+x-6}$ have?</question>"""

_SOP_ONE_SHOT_ASSISTANT = """\
{
  "reasoning": "I need to find vertical asymptotes of the rational function y = 2/(x\u00b2 + x - 6). Vertical asymptotes occur where the denominator equals zero (and numerator is non-zero). So I need to solve x\u00b2 + x - 6 = 0. This is a quadratic equation that can be factored: x\u00b2 + x - 6 = (x + 3)(x - 2) = 0. The solutions are x = -3 and x = 2. Both are valid vertical asymptotes because the numerator (2) is non-zero at these points. Therefore, the graph has two vertical asymptotes.",
  "prediction": "2"
}"""

# SOP example — gpqa (multiple-choice physics)
_GPQA_ONE_SHOT_USER = """\
Solve the following gpqa problem.

<question>A new planet is discovered with a mass of 2 Earth masses and a radius of 2 Earth radii. What is the surface gravity of this planet compared to Earth?

Please choose one of the following options:
A. 0.25g
B. 0.5g
C. 1g
D. 2g
</question>"""

_GPQA_ONE_SHOT_ASSISTANT = """\
{
  "reasoning": "Surface gravity g = GM/R\u00b2. If mass M' = 2M_Earth and radius R' = 2R_Earth, then g' = G(2M)/(2R)\u00b2 = 2GM/4R\u00b2 = 0.5 \u00d7 GM/R\u00b2 = 0.5g_Earth. So the surface gravity is half of Earth's.",
  "prediction": "B"
}"""

# SOP example — gsm8k (arithmetic word problem)
_GSM8K_ONE_SHOT_USER = """\
Solve the following gsm8k problem.

<question>Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?</question>"""

_GSM8K_ONE_SHOT_ASSISTANT = """\
{
  "reasoning": "Janet's ducks lay 16 eggs per day. She uses 3 for breakfast and 4 for muffins, so she uses 3 + 4 = 7 eggs. Remaining eggs = 16 - 7 = 9 eggs. She sells each for $2, so daily income = 9 \u00d7 $2 = $18.",
  "prediction": "18"
}"""

# Factual example — triviaqa
_FACTUAL_ONE_SHOT_USER = """\
Solve the following triviaqa problem.

<question>Which American-born Sinclair won the Nobel Prize for Literature in 1930?</question>"""

_FACTUAL_ONE_SHOT_ASSISTANT = """\
{
  "reasoning": "I need to identify an American-born person named Sinclair who won the Nobel Prize for Literature in 1930. The Nobel Prize for Literature in 1930 was awarded to Sinclair Lewis, the American novelist best known for works like 'Main Street', 'Babbitt', and 'Elmer Gantry'. He was the first American to win the Nobel Prize in Literature.",
  "prediction": "Sinclair Lewis"
}"""

# Factual example — popqa
_POPQA_ONE_SHOT_USER = """\
Solve the following popqa problem.

<question>What is the capital of Australia?</question>"""

_POPQA_ONE_SHOT_ASSISTANT = """\
{
  "reasoning": "Australia's capital city is often confused with Sydney (its largest city), but the actual capital is Canberra. Canberra was purpose-built as a compromise between Sydney and Melbourne and became the capital in 1913.",
  "prediction": "Canberra"
}"""

# Map dataset -> (one_shot_user, one_shot_assistant)
_ONE_SHOT_MAP: Dict[str, tuple] = {
    "math": (_SOP_ONE_SHOT_USER, _SOP_ONE_SHOT_ASSISTANT),
    "gpqa": (_GPQA_ONE_SHOT_USER, _GPQA_ONE_SHOT_ASSISTANT),
    "gsm8k": (_GSM8K_ONE_SHOT_USER, _GSM8K_ONE_SHOT_ASSISTANT),
    "triviaqa": (_FACTUAL_ONE_SHOT_USER, _FACTUAL_ONE_SHOT_ASSISTANT),
    "popqa": (_POPQA_ONE_SHOT_USER, _POPQA_ONE_SHOT_ASSISTANT),
}

# Datasets that use SOP vs factual framing
_SOP_DATASETS = {"gpqa", "gsm8k", "math"}
_FACTUAL_DATASETS = {"triviaqa", "popqa"}


# ─── User prompt template ─────────────────────────────────────────────────────

_SOLVE_USER_TEMPLATE = "Solve the following {dataset} problem.\n\n<question>{question}</question>"


# ─── JSON parsing ─────────────────────────────────────────────────────────────

def _parse_solver_response(raw: str) -> Dict[str, str]:
    """Parse JSON from solver LLM output, stripping markdown fences if present."""
    text = raw.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return {
                "reasoning": str(data.get("reasoning", "")).strip(),
                "prediction": str(data.get("prediction", "")).strip(),
            }
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback: try to extract prediction from raw text
    pred_match = re.search(r'"prediction"\s*:\s*"([^"]*)"', raw)
    pred = pred_match.group(1).strip() if pred_match else ""
    reas_match = re.search(r'"reasoning"\s*:\s*"(.*?)(?:"\s*,|\s*"\s*})', raw, re.DOTALL)
    reas = reas_match.group(1).strip() if reas_match else raw[:500]
    return {"reasoning": reas, "prediction": pred}


# ─── Solver Agent ─────────────────────────────────────────────────────────────

class SolverAgent:
    """Directly call an LLM to solve a QA problem.

    Uses one-shot prompting with real examples from existing pipeline outputs.
    Does NOT reveal ground truth to the model.
    """

    def __init__(
        self,
        base_url: str = SOLVER_BASE_URL,
        api_key: str = SOLVER_API_KEY,
        model: str = SOLVER_MODEL,
        temperature: float = 0.7,
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def solve(self, sample: Dict[str, Any]) -> Dict[str, str]:
        """Solve a single QA sample.

        Args:
            sample: Dict with at least "question" and "dataset" keys.

        Returns:
            Dict with "reasoning" and "prediction" keys.
        """
        dataset = sample.get("dataset", "").lower()
        question = sample.get("question", "")

        one_shot_user, one_shot_assistant = _ONE_SHOT_MAP.get(
            dataset, (_SOP_ONE_SHOT_USER, _SOP_ONE_SHOT_ASSISTANT)
        )

        user_message = _SOLVE_USER_TEMPLATE.format(
            dataset=dataset,
            question=question,
        )

        messages = [
            {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
            {"role": "user", "content": one_shot_user},
            {"role": "assistant", "content": one_shot_assistant},
            {"role": "user", "content": user_message},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            return {
                "reasoning": f"(Solver failed: {e})",
                "prediction": "",
            }

        return _parse_solver_response(raw)
