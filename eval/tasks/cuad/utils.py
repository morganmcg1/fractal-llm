"""
CUAD task utilities for lm-eval harness.

Provides doc_to_text, doc_to_target, and process_results functions
for evaluating models on the Contract Understanding Atticus Dataset.
"""

import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def compute_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """Check if prediction exactly matches any ground truth."""
    pred_normalized = normalize_answer(prediction)
    for gt in ground_truths:
        if normalize_answer(gt) == pred_normalized:
            return 1.0
    return 0.0


def compute_f1(prediction: str, ground_truths: list[str]) -> float:
    """Compute max F1 score against all ground truths."""
    pred_tokens = normalize_answer(prediction).split()

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = normalize_answer(gt).split()

        if not pred_tokens and not gt_tokens:
            max_f1 = max(max_f1, 1.0)
            continue
        if not pred_tokens or not gt_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1


def doc_to_text(doc: dict) -> str:
    """Convert document to input prompt."""
    context = doc["context"]
    question = doc["question"]

    # Truncate context if too long (keep first 4000 chars)
    if len(context) > 4000:
        context = context[:4000] + "..."

    return (
        f"You are a legal document analyst. Extract the relevant text span from the contract.\n\n"
        f"Contract:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def doc_to_target(doc: dict) -> str:
    """Get target answer from document."""
    answers = doc.get("answers", {})
    answer_texts = answers.get("text", [])

    if not answer_texts or all(not a.strip() for a in answer_texts):
        return "No relevant clause found."

    return answer_texts[0].strip()


def process_results(doc: dict, results: list[str]) -> dict:
    """Process model outputs and compute metrics."""
    prediction = results[0].strip() if results else ""

    # Get ground truth answers
    answers = doc.get("answers", {})
    ground_truths = answers.get("text", [])

    if not ground_truths or all(not gt.strip() for gt in ground_truths):
        ground_truths = ["No relevant clause found."]

    # Clean ground truths
    ground_truths = [gt.strip() for gt in ground_truths if gt.strip()]

    em = compute_exact_match(prediction, ground_truths)
    f1 = compute_f1(prediction, ground_truths)

    return {
        "exact_match": em,
        "f1": f1,
    }
