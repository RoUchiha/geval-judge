"""Async LLM judge runner for G-Eval."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Optional

import anthropic

from .models import DimensionScore, GEvalResult, Rubric
from .prompt_builder import build_prompt


class GEvalJudge:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.model = model

    async def _score_dimension(
        self,
        rubric_name: str,
        dim,
        input_text: str,
        output_text: str,
        context: Optional[str],
    ) -> DimensionScore:
        prompt = build_prompt(dim, input_text, output_text, context)
        message = await self.client.messages.create(
            model=self.model,
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = message.content[0].text.strip()
        data = json.loads(raw)
        return DimensionScore(
            dimension=dim.name,
            score=int(data["score"]),
            max_score=dim.max_score,
            reasoning=data["reasoning"],
        )

    async def evaluate(
        self,
        rubric: Rubric,
        input_text: str,
        output_text: str,
        context: Optional[str] = None,
    ) -> GEvalResult:
        tasks = [
            self._score_dimension(rubric.name, dim, input_text, output_text, context)
            for dim in rubric.dimensions
        ]
        scores = await asyncio.gather(*tasks)

        # Weighted composite score, normalized 0-1
        total_weight = sum(d.weight for d in rubric.dimensions)
        weighted_sum = sum(
            (s.score / s.max_score) * d.weight
            for s, d in zip(scores, rubric.dimensions)
        )
        composite = weighted_sum / total_weight if total_weight > 0 else 0.0

        return GEvalResult(
            input_text=input_text,
            output_text=output_text,
            dimension_scores=list(scores),
            composite_score=round(composite, 4),
            evaluated_at=datetime.now(timezone.utc),
        )

    def evaluate_sync(self, rubric: Rubric, input_text: str, output_text: str, context: Optional[str] = None) -> GEvalResult:
        return asyncio.run(self.evaluate(rubric, input_text, output_text, context))
