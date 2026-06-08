# G-Eval Judge

> A production-ready implementation of G-Eval — a chain-of-thought LLM-as-judge framework that lets you define custom evaluation rubrics in YAML and score any LLM output on arbitrary dimensions, without writing a single line of evaluation code.

---

## What Is This?

**G-Eval** is a technique from the paper *"G-Eval: NLG Evaluation Using GPT-4 with Better Human Alignment"* (Liu et al., 2023). The core insight: instead of writing rule-based metrics for LLM output quality (which are brittle and domain-specific), ask a capable LLM to evaluate the output using a structured rubric — and make it reason step-by-step (chain-of-thought) before scoring, which dramatically improves alignment with human judgments.

This project turns that idea into a reusable system:
- **Define your rubric in YAML** — no Python required to add new evaluation criteria
- **Dimensions are evaluated in parallel** via `asyncio.gather` — a 5-dimension rubric takes the same wall-clock time as a 1-dimension rubric
- **Jinja2 prompt templates** inject rubric context at runtime — consistent, inspectable prompts
- **Composite score** is a weighted mean across dimensions, normalized 0–1
- **Batch mode** processes hundreds of records with a progress bar

**Why does this matter?** As teams move from "does the model output something?" to "is the output actually good?", they need evaluation infrastructure that's as flexible as their product criteria. G-Eval provides that — a rubric that takes 2 minutes to write captures business-specific quality standards that no off-the-shelf metric can.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         G-Eval Pipeline                           │
│                                                                    │
│  rubric.yaml ──► [Rubric Parser] ──► Rubric (Pydantic)           │
│                                           │                        │
│                                           │  for each dimension    │
│                                           ▼                        │
│  input + output ──► [Prompt Builder] ──► prompt_str               │
│  (Jinja2 template)    CoT instruction                              │
│                       scoring scale                                │
│                       dimension def                                │
│                              │                                     │
│                              │  asyncio.gather (all dims parallel) │
│                              ▼                                     │
│                       [LLM Judge]                                  │
│                   {"score": 4, "reasoning": "..."}                 │
│                              │                                     │
│                              ▼                                     │
│                    [Score Aggregator]                              │
│               weighted mean → composite 0-1                        │
│               → GEvalResult (Pydantic)                             │
└──────────────────────────────────────────────────────────────────┘
```

### Why Chain-of-Thought Scoring Works

Without CoT, the LLM jumps straight to a number — which is noisy and inconsistent. With CoT, it first reasons through the evidence ("The response mentions X but omits Y...") before committing to a score. This mirrors how human evaluators work and produces scores that correlate significantly better with human judgments (per the original paper).

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM Judge | Claude (configurable model) via Anthropic SDK |
| Async parallelism | `asyncio.gather` |
| Prompt templating | Jinja2 |
| Rubric parsing | PyYAML + Pydantic v2 |
| CLI | typer |
| Console output | rich |
| Tests | pytest + pytest-asyncio |

---

## Installation

```bash
git clone https://github.com/RoUchiha/geval-judge.git
cd geval-judge
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

---

## Quick Start

### Evaluate a Single Response

```bash
python -m geval evaluate \
  --rubric rubrics/response_quality.yaml \
  --input "What is the capital of France?" \
  --output "The capital of France is Paris. It has been the capital since 987 AD and is home to landmarks like the Eiffel Tower." \
  --context "Paris is the capital and largest city of France."
```

**Output:**
```
╭──────────────────────────────────────────────────────╮
│              G-Eval: Response Quality                 │
├─────────────────┬───────┬─────┬───────────────────────┤
│ Dimension       │ Score │ Max │ Reasoning             │
├─────────────────┼───────┼─────┼───────────────────────┤
│ coherence       │   5   │  5  │ Clear, logical flow   │
│ groundedness    │   4   │  5  │ Mostly grounded but…  │
│ helpfulness     │   5   │  5  │ Directly answers…     │
╰─────────────────┴───────┴─────┴───────────────────────╯

Composite score: 0.920 / 1.0
```

### Batch Evaluation

```bash
python -m geval batch \
  --rubric rubrics/code_review.yaml \
  --dataset records.jsonl \
  --output results.csv
```

JSONL format:
```json
{"input": "Write a function to...", "output": "def foo(x):\n    return x", "context": "optional"}
```

### Python API

```python
import asyncio
from geval.rubric import load_rubric
from geval.judge import GEvalJudge

rubric = load_rubric("rubrics/response_quality.yaml")
judge = GEvalJudge()

result = judge.evaluate_sync(
    rubric,
    input_text="What is RAG?",
    output_text="RAG stands for Retrieval-Augmented Generation...",
)
print(f"Composite: {result.composite_score:.3f}")
for ds in result.dimension_scores:
    print(f"  {ds.dimension}: {ds.score}/{ds.max_score} — {ds.reasoning}")
```

---

## Writing Your Own Rubric

```yaml
name: "Customer Support Quality"
dimensions:
  - name: empathy
    description: "Does the response acknowledge the customer's frustration and show understanding?"
    scale: 1-5
    weight: 1.5
  - name: resolution
    description: "Does the response provide a clear path to resolving the customer's issue?"
    scale: 1-5
    weight: 2.0
  - name: professionalism
    description: "Is the tone professional, respectful, and on-brand?"
    scale: 1-3
    weight: 1.0
```

Save as `rubrics/customer_support.yaml` and pass with `--rubric`. No code changes needed.

---

## Running Tests

```bash
pytest --cov=src/geval
```

All LLM calls are mocked — no API key required for CI.

---

## Project Structure

```
geval-judge/
├── src/geval/
│   ├── rubric.py          # YAML parser → Pydantic Rubric
│   ├── prompt_builder.py  # Jinja2 CoT prompt generator
│   ├── judge.py           # async LLM runner
│   ├── batch.py           # JSONL batch evaluation
│   ├── cli.py             # typer CLI
│   └── models.py          # Pydantic data models
├── rubrics/
│   ├── response_quality.yaml   # coherence, groundedness, helpfulness
│   └── code_review.yaml        # correctness, readability, security, efficiency
├── tests/
└── pyproject.toml
```

---

## Extending This

- **Add score calibration**: provide anchor examples (score=1, score=3, score=5) in the rubric YAML; inject them into the prompt to reduce inter-run variance
- **Multi-model comparison**: run the same rubric with different judge models (`--model gpt-4o` vs `--model claude-opus-4-8`) and compare scores
- **FastAPI endpoint**: expose `POST /evaluate` using `api.py` for integration into CI/CD pipelines
