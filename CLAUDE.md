# CLAUDE.md — G-Eval Judge

This file gives Claude Code full context for this project. Read it before making any changes.

---

## What This Project Does

An implementation of **G-Eval** (Liu et al., 2023) — a chain-of-thought LLM-as-judge scoring framework. Users define evaluation rubrics in YAML (dimensions, descriptions, scale, weights). For each dimension, a Jinja2 prompt asks the judge LLM to reason step-by-step before outputting a JSON score. All dimensions are evaluated in parallel via `asyncio.gather`.

**Output**: a `GEvalResult` with per-dimension scores, reasoning, and a weighted composite score normalized 0–1.

---

## Repository Layout

```
geval-judge/
├── src/geval/
│   ├── models.py           # Pydantic: DimensionDef, Rubric, DimensionScore, GEvalResult
│   ├── rubric.py           # YAML parser → validated Rubric
│   ├── prompt_builder.py   # Jinja2 CoT prompt builder (one prompt per dimension)
│   ├── judge.py            # GEvalJudge: async evaluate() + sync evaluate_sync()
│   ├── batch.py            # JSONL batch evaluation → pd.DataFrame
│   ├── cli.py              # typer: `evaluate` and `batch` subcommands
│   └── api.py              # FastAPI POST /evaluate (optional, not wired by default)
├── rubrics/
│   ├── response_quality.yaml   # 3 dimensions: coherence, groundedness, helpfulness
│   └── code_review.yaml        # 4 dimensions: correctness, readability, security, efficiency
├── tests/
│   └── test_rubric.py      # rubric loading, prompt building, mocked async judge
└── pyproject.toml
```

---

## Tech Stack

| Component | Library |
|-----------|---------|
| LLM judge | `anthropic` SDK (async client) |
| Async parallelism | `asyncio.gather` |
| Prompt templating | `jinja2` |
| Rubric parsing | `pyyaml` + `pydantic` v2 |
| CLI | `typer` |
| Console output | `rich` |
| Tests | `pytest`, `pytest-asyncio` |

---

## Environment

```bash
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Commands

```bash
# Single evaluation
python -m geval evaluate \
  --rubric rubrics/response_quality.yaml \
  --input "What is RAG?" \
  --output "RAG stands for Retrieval-Augmented Generation..."

# Batch evaluation
python -m geval batch \
  --rubric rubrics/code_review.yaml \
  --dataset records.jsonl \
  --output results.csv

# Tests (mocked — no API key needed)
pytest
```

---

## Rubric YAML Schema

```yaml
name: "My Rubric"
dimensions:
  - name: coherence           # used as key in output
    description: "Is the response logically structured?"
    scale: 1-5                # "min-max" format — parsed by DimensionDef properties
    weight: 1.0               # relative weight for composite score
```

---

## Key Design Decisions

- **Async-first**: `GEvalJudge.evaluate()` is `async`; `evaluate_sync()` wraps with `asyncio.run()` for CLI convenience
- **`asyncio.gather`**: all dimension LLM calls fire simultaneously — N-dimension rubric ≈ 1× latency, not N×
- **Jinja2 templates**: prompt is defined once in `prompt_builder.py`; rubric YAML injects dimension-specific content at runtime
- **Weighted composite**: `sum(score/max_score * weight) / sum(weights)` — normalized 0–1 regardless of scale differences across dimensions
- **CoT enforced**: prompt explicitly says "think step-by-step before scoring" — improves score reliability

---

## Batch JSONL Schema

```json
{"input": "question or prompt", "output": "LLM response to evaluate", "context": "optional reference text"}
```

---

## Course Context

Built as part of the **UT Austin AI & Machine Learning** program (McCombs, 23-week executive program).
- **Course 03** — Prompt engineering (CoT prompting, structured LLM output)
- **Course 04** — Agentic AI: async parallel LLM execution, tool-use patterns

---

## Stretch Goals (not yet implemented)

- Score calibration: anchor examples (1=bad, 3=average, 5=excellent) injected into prompts
- Multi-model comparison: run same rubric through Claude vs GPT-4o, diff scores
- FastAPI `POST /evaluate` endpoint (skeleton exists in `api.py`)
