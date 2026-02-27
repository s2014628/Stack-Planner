"""
QA Bench Pipeline - Evaluator

Evaluation metrics for different QA benchmark types:
- Factual QA (TriviaQA, PopQA): Alias matching + F1 score
- Multiple Choice (GPQA): Exact match on option letter
- Math (GSM8K, MATH): Numerical / boxed answer equivalence

Each evaluator returns a score in [0, 1] and a boolean success flag.
"""

import re
import string
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional

# ─── Success thresholds ──────────────────────────────────────────────
F1_SUCCESS_THRESHOLD = 0.5
MATH_SUCCESS_THRESHOLD = 1.0  # Exact match required for math


# ─── Text normalization utilities ────────────────────────────────────


def normalize_answer(s: str) -> str:
    """Normalize answer text for F1 scoring (lowercase, remove articles/punctuation)."""
    s = s.replace(",", "")

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


# ─── Factual QA evaluation (TriviaQA, PopQA) ────────────────────────


def evaluate_factual(
    prediction: str,
    ground_truth: str,
    ground_truth_aliases: Optional[List[str]] = None,
) -> float:
    """
    Evaluate factual QA prediction using alias matching + F1 score.
    Following MemGen's approach: check if any alias is contained in the prediction.

    Args:
        prediction: Model's prediction text
        ground_truth: Primary ground truth answer
        ground_truth_aliases: List of acceptable answer aliases

    Returns:
        Score in [0, 1]
    """
    if not prediction:
        return 0.0

    prediction_lower = prediction.lower().strip()

    # Check alias containment (MemGen approach)
    aliases = ground_truth_aliases or [ground_truth]
    for alias in aliases:
        if alias.lower() in prediction_lower:
            return 1.0

    # Fallback to F1 score against primary ground truth
    best_f1 = f1_score(prediction, ground_truth)

    # Also check F1 against all aliases
    for alias in aliases:
        alias_f1 = f1_score(prediction, alias)
        best_f1 = max(best_f1, alias_f1)

    return best_f1


# ─── Multiple Choice evaluation (GPQA) ──────────────────────────────


def evaluate_multiple_choice(prediction: str, ground_truth: str) -> float:
    """
    Evaluate multiple choice prediction by extracting the chosen option letter.

    Args:
        prediction: Model's prediction (may contain reasoning)
        ground_truth: Correct option letter (A, B, C, or D)

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if not prediction:
        return 0.0

    # Try to extract answer from \\boxed{X} format first
    boxed_match = re.findall(r"\\boxed\{([A-Da-d])\}", prediction)
    if boxed_match:
        extracted = boxed_match[-1].upper()
        return 1.0 if extracted == ground_truth.upper() else 0.0

    # Try to extract answer from <answer>X</answer> format
    answer_match = re.findall(r"<answer>\s*([A-Da-d])\s*</answer>", prediction)
    if answer_match:
        extracted = answer_match[-1].upper()
        return 1.0 if extracted == ground_truth.upper() else 0.0

    # Try to find standalone letter answer patterns
    # Patterns: "The answer is A", "answer: B", "choose C", "(D)"
    patterns = [
        r"(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Da-d])\)?",
        r"\b([A-Da-d])\b\s*$",  # Last single letter
        r"^\s*\(?([A-Da-d])\)?\s*$",  # Just a letter
    ]
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.MULTILINE)
        if match:
            extracted = match.group(1).upper()
            return 1.0 if extracted == ground_truth.upper() else 0.0

    # Last resort: check if the ground truth letter appears anywhere
    if ground_truth.upper() in prediction.upper():
        return 0.5  # Partial credit

    return 0.0


# ─── Math evaluation (GSM8K, MATH) ──────────────────────────────────


def _extract_number(text: str) -> Optional[str]:
    """Extract the final numerical answer from text."""
    if not text:
        return None

    # Try \\boxed{} first
    boxed_match = re.findall(r"\\boxed\{([^}]+)\}", text)
    if boxed_match:
        return boxed_match[-1].strip()

    # Try <answer>X</answer> format
    answer_match = re.findall(r"<answer>\s*(.*?)\s*</answer>", text)
    if answer_match:
        return answer_match[-1].strip()

    # Try "#### X" format (GSM8K style)
    hash_match = re.search(r"####\s*(.+)", text)
    if hash_match:
        return hash_match.group(1).strip()

    # Try "the answer is X" patterns
    answer_patterns = [
        r"(?:answer|result|solution)\s*(?:is|=|:)\s*\$?([+-]?\d[\d,]*\.?\d*)\$?",
        r"=\s*\$?([+-]?\d[\d,]*\.?\d*)\$?\s*$",
    ]
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "").strip()

    # Try to find the last number in the text
    numbers = re.findall(r"[+-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def _normalize_number(s: str) -> Optional[float]:
    """Try to parse a string as a number."""
    if not s:
        return None
    s = s.replace(",", "").replace(" ", "").strip()
    # Handle percentage
    if s.endswith("%"):
        s = s[:-1]
    try:
        return float(s)
    except ValueError:
        return None


def _is_math_equiv(str1: str, str2: str) -> bool:
    """
    Check if two math expressions are equivalent.
    Adapted from MemGen's math_utils.py approach.
    """
    if not str1 or not str2:
        return False

    # Direct string match after normalization
    s1 = str1.strip().lower().replace(" ", "")
    s2 = str2.strip().lower().replace(" ", "")
    if s1 == s2:
        return True

    # Numerical comparison
    n1 = _normalize_number(str1)
    n2 = _normalize_number(str2)
    if n1 is not None and n2 is not None:
        return abs(n1 - n2) < 1e-6

    return False


def evaluate_math(prediction: str, ground_truth: str) -> float:
    """
    Evaluate math prediction by comparing extracted numerical answers.

    Args:
        prediction: Model's prediction (may contain reasoning steps)
        ground_truth: Correct numerical answer

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    if not prediction:
        return 0.0

    predicted_answer = _extract_number(prediction)
    if predicted_answer is None:
        return 0.0

    if _is_math_equiv(predicted_answer, ground_truth):
        return 1.0

    return 0.0


# ─── Unified evaluation interface ────────────────────────────────────


def evaluate_qa(
    prediction: str,
    ground_truth: str,
    dataset: str,
    ground_truth_aliases: Optional[List[str]] = None,
) -> float:
    """
    Evaluate a QA prediction using the appropriate metric for the dataset.

    Args:
        prediction: Model's prediction text
        ground_truth: Ground truth answer
        dataset: Dataset name (triviaqa, popqa, gpqa, gsm8k, math)
        ground_truth_aliases: List of acceptable answer aliases (for factual QA)

    Returns:
        Score in [0, 1]
    """
    dataset = dataset.lower().strip()

    if dataset in ("triviaqa", "popqa"):
        return evaluate_factual(prediction, ground_truth, ground_truth_aliases)
    elif dataset == "gpqa":
        return evaluate_multiple_choice(prediction, ground_truth)
    elif dataset in ("gsm8k", "math"):
        return evaluate_math(prediction, ground_truth)
    else:
        # Default: F1 score
        return f1_score(prediction, ground_truth)


def is_success(score: float, dataset: str) -> bool:
    """
    Determine if a score indicates success for the given dataset type.

    Args:
        score: Evaluation score in [0, 1]
        dataset: Dataset name

    Returns:
        True if the score meets the success threshold
    """
    dataset = dataset.lower().strip()
    if dataset in ("gsm8k", "math", "gpqa"):
        return score >= MATH_SUCCESS_THRESHOLD
    else:
        return score >= F1_SUCCESS_THRESHOLD


def evaluate_runs(
    runs: List[Dict[str, Any]],
    ground_truth: str,
    dataset: str,
    ground_truth_aliases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluate a list of benchmark runs.

    Args:
        runs: List of run results with "prediction" field
        ground_truth: Ground truth answer
        dataset: Dataset name
        ground_truth_aliases: List of acceptable answer aliases

    Returns:
        List of runs with added "score" and "success" fields
    """
    evaluated_runs = []
    for run in runs:
        prediction = run.get("prediction", "")
        score = evaluate_qa(prediction, ground_truth, dataset, ground_truth_aliases)
        evaluated_run = {
            **run,
            "score": round(score, 4),
            "success": is_success(score, dataset),
        }
        evaluated_runs.append(evaluated_run)
    return evaluated_runs
