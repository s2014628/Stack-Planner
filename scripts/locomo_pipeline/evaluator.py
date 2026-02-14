import regex
import string
import unicodedata
from collections import Counter
from typing import List

import numpy as np
from nltk.stem import PorterStemmer

ps = PorterStemmer()

F1_SUCCESS_THRESHOLD = 0.5


def _normalize(text):
    return unicodedata.normalize("NFD", text)


def normalize_answer(s):
    s = s.replace(",", "")

    def remove_articles(text):
        return regex.sub(r"\b(a|an|the|and)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_multi_answer(prediction, ground_truth):
    predictions = [p.strip() for p in prediction.split(",")]
    ground_truths = [g.strip() for g in ground_truth.split(",")]
    return float(
        np.mean(
            [max([f1_score(pred, gt) for pred in predictions]) for gt in ground_truths]
        )
    )


def evaluate_qa(prediction: str, ground_truth: str, category: int) -> float:
    if category in [2, 3, 4]:
        if category == 3:
            ground_truth = ground_truth.split(";")[0].strip()
        return f1_score(prediction, ground_truth)
    elif category == 1:
        return f1_multi_answer(prediction, ground_truth)
    elif category == 5:
        if "no information available" in prediction.lower() or "not mentioned" in prediction.lower():
            return 1.0
        else:
            return 0.0
    else:
        return f1_score(prediction, ground_truth)


def is_success(f1: float, threshold: float = F1_SUCCESS_THRESHOLD) -> bool:
    return f1 >= threshold


def evaluate_runs(
    runs: List[dict], ground_truth: str, category: int
) -> List[dict]:
    evaluated_runs = []
    for run in runs:
        prediction = run.get("prediction", "")
        score = evaluate_qa(prediction, ground_truth, category)
        evaluated_run = {
            **run,
            "f1_score": round(score, 4),
            "success": is_success(score),
        }
        evaluated_runs.append(evaluated_run)
    return evaluated_runs
