"""Batch evaluation over a JSONL dataset."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .judge import GEvalJudge
from .models import GEvalResult, Rubric


async def _evaluate_all(
    judge: GEvalJudge,
    rubric: Rubric,
    records: list,
) -> List[GEvalResult]:
    tasks = [
        judge.evaluate(
            rubric,
            r["input"],
            r["output"],
            r.get("context"),
        )
        for r in records
    ]
    return await asyncio.gather(*tasks)


def run_batch(
    rubric: Rubric,
    jsonl_path: Union[str, Path],
    model: str = "claude-haiku-4-5-20251001",
) -> pd.DataFrame:
    records = []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    judge = GEvalJudge(model=model)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn()) as progress:
        task = progress.add_task(f"Evaluating {len(records)} records…", total=len(records))
        results = asyncio.run(_evaluate_all(judge, rubric, records))
        progress.update(task, advance=len(records))

    rows = []
    for r in results:
        row = {
            "input": r.input_text[:80],
            "output": r.output_text[:80],
            "composite_score": r.composite_score,
        }
        for ds in r.dimension_scores:
            row[ds.dimension] = ds.score
            row[f"{ds.dimension}_reasoning"] = ds.reasoning
        rows.append(row)

    return pd.DataFrame(rows)
