"""
Evaluate models on CUAD-QA using SQuAD-style metrics.

No LLM judge needed! Uses standard extractive QA metrics:
- Exact Match (EM): Prediction exactly matches a ground truth answer
- F1 Token Overlap: Token-level precision/recall vs best answer

Dataset: theatticusproject/cuad-qa (PR #6 parquet revision for datasets v4+)
Categories: 41 legal clause types

Usage:
    # Evaluate a model on CUAD test set
    uv run eval/eval_cuad.py --model nanochat-students/nanochat-d20 --max-samples 100

    # Evaluate predictions from a file
    uv run eval/eval_cuad.py --predictions-file results/cuad_preds.jsonl
"""

import json
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import simple_parsing as sp
from datasets import load_dataset
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


@dataclass
class Args:
    """Evaluate on CUAD-QA"""

    model: str | None = None  # HuggingFace model ID to evaluate
    predictions_file: str | None = None  # Path to predictions JSONL
    max_samples: int | None = None  # Limit samples (None = all)
    max_context_len: int = 2048  # Max context length in characters
    max_new_tokens: int = 256  # Max tokens to generate
    batch_size: int = 1  # Batch size for generation
    output_file: str | None = None  # Save predictions to JSONL
    split: str = "test"  # Dataset split: train or test


def normalize_answer(s: str) -> str:
    """Normalize answer for comparison (lowercase, remove punctuation/articles)."""
    # Lowercase
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove extra whitespace
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

        if not pred_tokens or not gt_tokens:
            # Handle empty case
            if not pred_tokens and not gt_tokens:
                max_f1 = max(max_f1, 1.0)
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


def format_prompt(context: str, question: str, max_context_len: int) -> str:
    """Format context and question as a prompt."""
    if len(context) > max_context_len:
        context = context[:max_context_len] + "..."

    return (
        "<|system|>\n"
        "You are a legal document analyst. Answer questions about contracts by extracting relevant text spans.\n"
        "<|user|>\n"
        f"Contract:\n{context}\n\n"
        f"Question: {question}\n"
        "<|assistant|>\n"
    )


def evaluate_model(args: Args):
    """Evaluate a model on CUAD-QA."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    console.rule(f"[bold blue]Evaluating {args.model} on CUAD-QA")

    # Load model and tokenizer
    console.print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    console.print(f"Loading CUAD-QA {args.split} split...")
    dataset = load_dataset("theatticusproject/cuad-qa", revision="refs/pr/6", split=args.split)

    if args.max_samples:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    # Evaluate
    results = []
    em_total = 0.0
    f1_total = 0.0

    with Progress() as progress:
        task = progress.add_task("Evaluating...", total=len(dataset))

        for sample in dataset:
            context = sample["context"]
            question = sample["question"]
            ground_truths = sample["answers"]["text"]

            # Skip samples with no answers (or treat as "no answer")
            if not ground_truths or all(not gt.strip() for gt in ground_truths):
                ground_truths = ["No relevant clause found."]

            # Generate prediction
            prompt = format_prompt(context, question, args.max_context_len)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)

            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode and extract answer
            full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the assistant's response
            if "<|assistant|>" in full_output:
                prediction = full_output.split("<|assistant|>")[-1].strip()
            else:
                prediction = full_output[len(prompt):].strip()

            # Compute metrics
            em = compute_exact_match(prediction, ground_truths)
            f1 = compute_f1(prediction, ground_truths)

            em_total += em
            f1_total += f1

            results.append({
                "id": sample["id"],
                "question": question,
                "ground_truths": ground_truths,
                "prediction": prediction,
                "exact_match": em,
                "f1": f1,
            })

            progress.update(task, advance=1)

    # Compute averages
    n = len(results)
    avg_em = em_total / n if n > 0 else 0
    avg_f1 = f1_total / n if n > 0 else 0

    return results, avg_em, avg_f1


def evaluate_predictions_file(args: Args):
    """Evaluate predictions from a file."""
    console.rule(f"[bold blue]Evaluating predictions from {args.predictions_file}")

    results = []
    em_total = 0.0
    f1_total = 0.0

    with open(args.predictions_file) as f:
        for line in f:
            pred = json.loads(line)
            ground_truths = pred.get("ground_truths", pred.get("all_answers", []))
            prediction = pred.get("prediction", pred.get("answer", ""))

            em = compute_exact_match(prediction, ground_truths)
            f1 = compute_f1(prediction, ground_truths)

            em_total += em
            f1_total += f1

            results.append({
                **pred,
                "exact_match": em,
                "f1": f1,
            })

    n = len(results)
    avg_em = em_total / n if n > 0 else 0
    avg_f1 = f1_total / n if n > 0 else 0

    return results, avg_em, avg_f1


def main():
    args = sp.parse(Args)

    if args.model:
        results, avg_em, avg_f1 = evaluate_model(args)
    elif args.predictions_file:
        results, avg_em, avg_f1 = evaluate_predictions_file(args)
    else:
        console.print("[red]Error: Specify --model or --predictions-file[/red]")
        return

    # Display results
    console.print()
    console.print(Panel(
        f"[bold green]Exact Match:[/bold green] {avg_em*100:.2f}%\n"
        f"[bold green]F1 Score:[/bold green] {avg_f1*100:.2f}%\n"
        f"[bold]Samples:[/bold] {len(results):,}",
        title="CUAD-QA Results",
    ))

    # Show some examples
    console.print("\n[bold]Sample predictions:[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Question", max_width=40)
    table.add_column("Prediction", max_width=40)
    table.add_column("EM", justify="center")
    table.add_column("F1", justify="center")

    for r in results[:5]:
        em_str = "[green]✓[/green]" if r["exact_match"] > 0 else "[red]✗[/red]"
        table.add_row(
            r["question"][:40] + "..." if len(r["question"]) > 40 else r["question"],
            r["prediction"][:40] + "..." if len(r["prediction"]) > 40 else r["prediction"],
            em_str,
            f"{r['f1']:.2f}",
        )

    console.print(table)

    # Save results if requested
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        console.print(f"\n[green]Saved predictions to {out_path}[/green]")

    # Return summary
    return {
        "exact_match": avg_em,
        "f1": avg_f1,
        "n_samples": len(results),
    }


if __name__ == "__main__":
    main()
