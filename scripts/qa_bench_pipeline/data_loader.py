"""
QA Bench Pipeline - Data Loader

Load and preprocess QA benchmark datasets:
- TriviaQA (事实性经验 - Factual Experience)
- PopQA   (事实性经验 - Factual Experience)
- GPQA    (SOP系统层经验 - SOP System Experience)
- GSM8K   (SOP系统层经验 - SOP System Experience)
- MATH    (SOP系统层经验 - SOP System Experience)

Each dataset is loaded from HuggingFace and normalized into a unified sample format.
"""

import json
import os
import random
from typing import Any, Dict, List, Optional

# ─── Dataset type constants ───────────────────────────────────────────
FACTUAL_DATASETS = ["triviaqa", "popqa"]
SOP_DATASETS = ["gpqa", "gsm8k", "math"]
ALL_DATASETS = FACTUAL_DATASETS + SOP_DATASETS

# ─── Experience type classification ──────────────────────────────────
EXPERIENCE_TYPE_MAP = {
    "triviaqa": "factual",
    "popqa": "factual",
    "gpqa": "sop",
    "gsm8k": "sop",
    "math": "sop",
}


def load_triviaqa(
    split: str = "validation", max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load TriviaQA dataset.
    Reference: MemGen uses `mandarjoshi/trivia_qa` with `rc.wikipedia.nocontext` config.
    Uses search agent pattern: think -> search -> answer.

    Args:
        split: Dataset split to use ("train", "validation", "test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of normalized QA samples
    """
    from datasets import load_dataset

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia.nocontext", split=split)
    samples = []

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        question = example["question"].strip()
        # TriviaQA answers have `normalized_aliases` as a list of acceptable answers
        answer_obj = example["answer"]
        ground_truth_list = answer_obj.get("normalized_aliases", [])
        ground_truth = answer_obj.get("value", "")
        if not ground_truth and ground_truth_list:
            ground_truth = ground_truth_list[0]

        samples.append(
            {
                "sample_id": f"triviaqa_{idx}",
                "dataset": "triviaqa",
                "experience_type": "factual",
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_aliases": ground_truth_list,
                "metadata": {
                    "split": split,
                    "original_index": idx,
                },
            }
        )

    return samples


def load_popqa(
    split: str = "test", max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load PopQA dataset.
    PopQA contains factual questions about popular entities.
    Uses search agent pattern for knowledge retrieval.

    Args:
        split: Dataset split to use ("test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of normalized QA samples
    """
    from datasets import load_dataset

    ds = load_dataset("akariasai/PopQA", split=split)
    samples = []

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        question = example["question"].strip()
        # PopQA has `possible_answers` as a JSON-encoded list
        possible_answers_raw = example.get("possible_answers", "[]")
        if isinstance(possible_answers_raw, str):
            try:
                ground_truth_list = json.loads(possible_answers_raw)
            except json.JSONDecodeError:
                ground_truth_list = [possible_answers_raw]
        elif isinstance(possible_answers_raw, list):
            ground_truth_list = possible_answers_raw
        else:
            ground_truth_list = [str(possible_answers_raw)]

        ground_truth = ground_truth_list[0] if ground_truth_list else ""

        samples.append(
            {
                "sample_id": f"popqa_{idx}",
                "dataset": "popqa",
                "experience_type": "factual",
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_aliases": ground_truth_list,
                "metadata": {
                    "split": split,
                    "original_index": idx,
                    "subj": example.get("subj", ""),
                    "prop": example.get("prop", ""),
                    "s_popularity": example.get("s_pop", 0),
                },
            }
        )

    return samples


def load_gpqa(
    split: str = "train", max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load GPQA dataset (Graduate-Level Google-Proof QA).
    Reference: MemGen uses `Idavidrein/gpqa` with `gpqa_diamond` for test.
    Multiple choice format with randomized option order.

    Args:
        split: Dataset split to use ("train" for gpqa_main, "test" for gpqa_diamond)
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of normalized QA samples
    """
    from datasets import load_dataset

    if split == "test":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
    else:
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")

    samples = []

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        question_text = example["Question"].strip()
        correct_answer = example["Correct Answer"].strip()
        incorrect_1 = example["Incorrect Answer 1"].strip()
        incorrect_2 = example["Incorrect Answer 2"].strip()
        incorrect_3 = example["Incorrect Answer 3"].strip()

        # Randomize option order (following MemGen approach)
        candidates = [correct_answer, incorrect_1, incorrect_2, incorrect_3]
        indices = list(range(4))
        random.shuffle(indices)
        orders = [chr(ord("A") + i) for i in range(4)]

        correct_order = None
        options_text = ""
        for i, shuffled_idx in enumerate(indices):
            options_text += f"{orders[i]}. {candidates[shuffled_idx]}\n"
            if shuffled_idx == 0:  # correct answer
                correct_order = orders[i]

        formatted_question = (
            f"{question_text}\n\n"
            f"Please choose one of the following options:\n"
            f"{options_text}"
        )

        ground_truth = correct_order  # e.g., "A", "B", "C", or "D"

        samples.append(
            {
                "sample_id": f"gpqa_{idx}",
                "dataset": "gpqa",
                "experience_type": "sop",
                "question": formatted_question,
                "ground_truth": ground_truth,
                "ground_truth_aliases": [ground_truth],
                "metadata": {
                    "split": split,
                    "original_index": idx,
                    "correct_answer_text": correct_answer,
                    "explanation": example.get("Explanation", ""),
                    "domain": example.get("High-level domain", ""),
                },
            }
        )

    return samples


def load_gsm8k(
    split: str = "test", max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset (Grade School Math 8K).
    Reference: MemGen uses `gsm8k` with `main` config.
    Math word problems with numerical answers.

    Args:
        split: Dataset split to use ("train", "test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of normalized QA samples
    """
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split=split)
    samples = []

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        question = example["question"].strip()
        answer_raw = example["answer"].strip()

        # Extract the final numerical answer after "####"
        parts = answer_raw.split("####")
        if len(parts) > 1:
            numerical_answer = parts[-1].strip()
        else:
            numerical_answer = answer_raw

        samples.append(
            {
                "sample_id": f"gsm8k_{idx}",
                "dataset": "gsm8k",
                "experience_type": "sop",
                "question": question,
                "ground_truth": numerical_answer,
                "ground_truth_aliases": [numerical_answer],
                "metadata": {
                    "split": split,
                    "original_index": idx,
                    "full_solution": answer_raw,
                },
            }
        )

    return samples


def load_math(
    split: str = "test", max_samples: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Load MATH dataset (Competition Mathematics).
    Uses hendrycks_math / lighteval MATH dataset.
    Math competition problems with LaTeX boxed answers.

    Args:
        split: Dataset split to use ("train", "test")
        max_samples: Maximum number of samples to load (None for all)

    Returns:
        List of normalized QA samples
    """
    from datasets import load_dataset

    # Try lighteval/MATH first, fallback to hendrycks/competition_math
    try:
        ds = load_dataset("lighteval/MATH", split=split)
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split=split)

    samples = []

    for idx, example in enumerate(ds):
        if max_samples and idx >= max_samples:
            break

        question = example["problem"].strip()
        solution = example["solution"].strip()

        # Extract boxed answer from solution
        ground_truth = _extract_boxed_answer(solution)
        if not ground_truth:
            ground_truth = solution  # fallback to full solution

        samples.append(
            {
                "sample_id": f"math_{idx}",
                "dataset": "math",
                "experience_type": "sop",
                "question": question,
                "ground_truth": ground_truth,
                "ground_truth_aliases": [ground_truth],
                "metadata": {
                    "split": split,
                    "original_index": idx,
                    "full_solution": solution,
                    "level": example.get("level", ""),
                    "type": example.get("type", ""),
                },
            }
        )

    return samples


def _extract_boxed_answer(solution: str) -> str:
    """Extract the answer from \\boxed{...} in a LaTeX solution string."""
    idx = solution.rfind("\\boxed")
    if idx < 0:
        idx = solution.rfind("\\fbox")
        if idx < 0:
            return ""

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(solution):
        if solution[i] == "{":
            num_left_braces_open += 1
        if solution[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return ""

    boxed_str = solution[idx : right_brace_idx + 1]
    # Remove the \\boxed{ and trailing }
    if boxed_str.startswith("\\boxed{") and boxed_str.endswith("}"):
        return boxed_str[len("\\boxed{") : -1]
    elif boxed_str.startswith("\\boxed "):
        return boxed_str[len("\\boxed ") :]
    elif boxed_str.startswith("\\fbox{") and boxed_str.endswith("}"):
        return boxed_str[len("\\fbox{") : -1]

    return boxed_str


# ─── Unified loader ──────────────────────────────────────────────────

DATASET_LOADERS = {
    "triviaqa": load_triviaqa,
    "popqa": load_popqa,
    "gpqa": load_gpqa,
    "gsm8k": load_gsm8k,
    "math": load_math,
}


def load_qa_samples(
    datasets: Optional[List[str]] = None,
    split: str = "test",
    max_samples_per_dataset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Load QA samples from specified benchmark datasets.

    Args:
        datasets: List of dataset names to load (default: all datasets)
        split: Dataset split to use
        max_samples_per_dataset: Maximum samples per dataset (None for all)

    Returns:
        Combined list of normalized QA samples from all specified datasets
    """
    if datasets is None:
        datasets = ALL_DATASETS

    all_samples = []
    for dataset_name in datasets:
        dataset_name = dataset_name.lower().strip()
        if dataset_name not in DATASET_LOADERS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available: {list(DATASET_LOADERS.keys())}"
            )

        loader = DATASET_LOADERS[dataset_name]
        samples = loader(split=split, max_samples=max_samples_per_dataset)
        all_samples.extend(samples)
        print(f"  Loaded {len(samples)} samples from {dataset_name} ({split})")

    return all_samples


def build_user_query(sample: Dict[str, Any]) -> str:
    """
    Build a user query for the central agent based on the sample's dataset type.
    Different prompts for factual vs. SOP experience types.

    Args:
        sample: A normalized QA sample

    Returns:
        Formatted user query string for the central agent
    """
    dataset = sample["dataset"]
    question = sample["question"]
    experience_type = sample["experience_type"]

    if experience_type == "factual":
        # For TriviaQA/PopQA: search-oriented factual QA
        query = (
            f"Answer the following factual question. "
            f"You should search for relevant information to find the answer. "
            f"If you need to look up facts, use the search/research capability.\n\n"
            f"Question: {question}\n\n"
            f"Please provide a short, concise factual answer. "
            f"Answer with the most precise and specific information available."
        )
    elif dataset == "gpqa":
        # For GPQA: graduate-level multiple choice with reasoning
        query = (
            f"Answer the following graduate-level question. "
            f"You may search for relevant information to help reason through the answer. "
            f"Think step by step and choose the correct option.\n\n"
            f"{question}\n"
            f"Provide your final answer as a single letter (A, B, C, or D)."
        )
    elif dataset in ("gsm8k", "math"):
        # For GSM8K/MATH: math problem solving
        query = (
            f"Solve the following math problem step by step. "
            f"You may search for relevant formulas or mathematical concepts if needed. "
            f"Show your reasoning process.\n\n"
            f"Question: {question}\n\n"
            f"Provide the final numerical answer."
        )
    else:
        query = f"Question: {question}\nPlease provide a concise answer."

    return query


def save_samples(samples: List[Dict[str, Any]], output_path: str) -> None:
    """Save QA samples to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and preprocess QA bench datasets"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="*",
        default=None,
        help=f"Datasets to load (default: all). Options: {ALL_DATASETS}",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--output-path", type=str, default="./data/qa_bench_samples.json"
    )
    args = parser.parse_args()

    samples = load_qa_samples(
        datasets=args.datasets,
        split=args.split,
        max_samples_per_dataset=args.max_samples,
    )
    save_samples(samples, args.output_path)
    print(f"\nExtracted {len(samples)} total QA samples, saved to {args.output_path}")

    # Print distribution
    from collections import Counter

    dist = Counter(s["dataset"] for s in samples)
    type_dist = Counter(s["experience_type"] for s in samples)
    print(f"Dataset distribution: {dict(dist)}")
    print(f"Experience type distribution: {dict(type_dist)}")
