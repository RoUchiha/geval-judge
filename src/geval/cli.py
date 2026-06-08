"""CLI for G-Eval."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import box

app = typer.Typer(help="G-Eval: LLM-as-judge scoring with YAML rubrics.")
console = Console()


@app.command()
def evaluate(
    rubric: Path = typer.Option(..., help="Path to YAML rubric file."),
    input_text: str = typer.Option(..., "--input", help="Input/question text."),
    output_text: str = typer.Option(..., "--output", help="LLM output text to score."),
    context: Optional[str] = typer.Option(None, "--context", help="Optional reference context."),
    model: str = typer.Option("claude-haiku-4-5-20251001", help="Judge LLM model."),
) -> None:
    """Score a single input/output pair against a rubric."""
    from .rubric import load_rubric
    from .judge import GEvalJudge

    rb = load_rubric(rubric)
    judge = GEvalJudge(model=model)
    result = judge.evaluate_sync(rb, input_text, output_text, context)

    table = Table(title=f"G-Eval: {rb.name}", box=box.ROUNDED)
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Max", justify="center")
    table.add_column("Reasoning")

    for ds in result.dimension_scores:
        color = "green" if ds.score / ds.max_score >= 0.7 else "yellow" if ds.score / ds.max_score >= 0.4 else "red"
        table.add_row(ds.dimension, f"[{color}]{ds.score}[/{color}]", str(ds.max_score), ds.reasoning)

    console.print(table)
    console.print(f"\n[bold]Composite score:[/bold] {result.composite_score:.3f} / 1.0")


@app.command()
def batch(
    rubric: Path = typer.Option(..., help="Path to YAML rubric file."),
    dataset: Path = typer.Option(..., help="Path to JSONL file with {input, output, context} records."),
    output: Optional[Path] = typer.Option(None, help="Save results to CSV."),
    model: str = typer.Option("claude-haiku-4-5-20251001"),
) -> None:
    """Run batch evaluation over a JSONL dataset."""
    from .rubric import load_rubric
    from .batch import run_batch

    rb = load_rubric(rubric)
    df = run_batch(rb, dataset, model=model)
    console.print(df.to_string())
    if output:
        df.to_csv(output, index=False)
        console.print(f"\n[dim]Saved → {output}[/dim]")


if __name__ == "__main__":
    app()
