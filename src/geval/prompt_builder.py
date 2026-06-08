"""Jinja2-based prompt builder for G-Eval dimensions."""

from __future__ import annotations

from jinja2 import Environment, BaseLoader

from .models import DimensionDef

_TEMPLATE_STR = """You are an expert evaluator. Your task is to score an LLM-generated response on the dimension of **{{ dimension.name }}**.

**Dimension definition:** {{ dimension.description }}

**Scoring scale:** {{ dimension.scale }} ({{ dimension.min_score }} = worst, {{ dimension.max_score }} = best)

---

**Input / Question:**
{{ input_text }}

**Response to evaluate:**
{{ output_text }}

{% if context %}**Reference context (ground truth or source material):**
{{ context }}{% endif %}

---

**Instructions:**
1. Think step-by-step about how well the response performs on **{{ dimension.name }}**.
2. Consider the scoring criteria carefully.
3. Output ONLY valid JSON in this exact format — no prose, no markdown:

{"score": <integer between {{ dimension.min_score }} and {{ dimension.max_score }}>, "reasoning": "<one concise sentence explaining your score>"}"""

_env = Environment(loader=BaseLoader())
_template = _env.from_string(_TEMPLATE_STR)


def build_prompt(
    dimension: DimensionDef,
    input_text: str,
    output_text: str,
    context: str | None = None,
) -> str:
    return _template.render(
        dimension=dimension,
        input_text=input_text,
        output_text=output_text,
        context=context,
    )
