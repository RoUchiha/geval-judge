"""Tests for rubric loading and prompt building."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from geval.models import DimensionDef, GEvalResult, Rubric
from geval.rubric import load_rubric
from geval.prompt_builder import build_prompt


SAMPLE_RUBRIC_YAML = """
name: "Test Rubric"
dimensions:
  - name: accuracy
    description: "Is the response accurate?"
    scale: 1-5
    weight: 1.0
  - name: clarity
    description: "Is the response clear?"
    scale: 1-3
    weight: 0.5
"""


def test_load_rubric(tmp_path):
    f = tmp_path / "rubric.yaml"
    f.write_text(SAMPLE_RUBRIC_YAML)
    rubric = load_rubric(f)
    assert rubric.name == "Test Rubric"
    assert len(rubric.dimensions) == 2
    assert rubric.dimensions[0].name == "accuracy"
    assert rubric.dimensions[1].max_score == 3


def test_dimension_min_max():
    dim = DimensionDef(name="test", description="test", scale="1-10")
    assert dim.min_score == 1
    assert dim.max_score == 10


def test_build_prompt_contains_dimension():
    dim = DimensionDef(name="coherence", description="Is it coherent?", scale="1-5")
    prompt = build_prompt(dim, "What is AI?", "AI is artificial intelligence.")
    assert "coherence" in prompt
    assert "Is it coherent?" in prompt
    assert "1-5" in prompt
    assert "What is AI?" in prompt


def test_build_prompt_includes_context():
    dim = DimensionDef(name="groundedness", description="Is it grounded?", scale="1-5")
    prompt = build_prompt(dim, "Q", "A", context="Some context here.")
    assert "Some context here." in prompt


@pytest.mark.asyncio
async def test_judge_evaluate():
    from geval.judge import GEvalJudge
    from geval.models import Rubric, DimensionDef

    rubric = Rubric(
        name="Test",
        dimensions=[DimensionDef(name="coherence", description="test", scale="1-5")],
    )

    mock_response = MagicMock()
    mock_response.content = [MagicMock(text='{"score": 4, "reasoning": "Good coherence."}')]

    with patch("geval.judge.anthropic.AsyncAnthropic") as mock_cls:
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(return_value=mock_response)

        judge = GEvalJudge()
        result = await judge.evaluate(rubric, "input", "output")

    assert result.composite_score == pytest.approx(4 / 5, abs=0.01)
    assert result.dimension_scores[0].score == 4
    assert result.dimension_scores[0].reasoning == "Good coherence."
