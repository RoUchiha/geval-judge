"""Pydantic models for G-Eval."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, field_validator


class DimensionDef(BaseModel):
    name: str
    description: str
    scale: str  # e.g. "1-5"
    weight: float = 1.0

    @property
    def min_score(self) -> int:
        return int(self.scale.split("-")[0])

    @property
    def max_score(self) -> int:
        return int(self.scale.split("-")[1])


class Rubric(BaseModel):
    name: str
    dimensions: List[DimensionDef]


class DimensionScore(BaseModel):
    dimension: str
    score: int
    max_score: int
    reasoning: str

    @field_validator("score")
    @classmethod
    def score_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Score must be >= 1")
        return v


class GEvalResult(BaseModel):
    input_text: str
    output_text: str
    dimension_scores: List[DimensionScore]
    composite_score: float  # weighted mean, normalized 0-1
    evaluated_at: datetime
