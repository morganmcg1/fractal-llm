"""
Run lm-evaluation-harness on a given model/checkpoint for quick OOD evaluation.

Example:
    uv run eval/run_lmeval.py --model nanochat-students/nanochat-d20 --tasks hellaswag,arc_challenge
"""

from dataclasses import dataclass
import simple_parsing as sp
from rich.console import Console
from rich.panel import Panel
from lm_eval import evaluator

console = Console()


@dataclass
class Args:
    """Evaluate a model with lm-evaluation-harness."""

    model: str  # HF model ID or local path
    tasks: str = "hellaswag,arc_challenge"  # Comma-separated task list
    batch_size: int = 8  # LM Eval batch size
    max_samples: int | None = 500  # Optional cap per task for speed
    device: str = "cuda"  # cuda or cpu


def main(args: Args):
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    console.rule("[bold blue]lm-eval-harness")
    console.print(Panel(f"Model: {args.model}\nTasks: {task_list}\nBatch: {args.batch_size}", title="Config"))

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.model},device={args.device}",
        tasks=task_list,
        batch_size=args.batch_size,
        limit=args.max_samples,
    )

    console.print(Panel(str(results["results"]), title="Scores"))
    console.print("[green]Complete.")


if __name__ == "__main__":
    main(sp.parse(Args))
