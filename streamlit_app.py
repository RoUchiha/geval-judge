"""
G-Eval Judge — Live Demo
Score any LLM output against a YAML rubric using chain-of-thought LLM-as-judge.
"""

import os, json, asyncio, re
import streamlit as st

st.set_page_config(page_title="G-Eval Judge", page_icon="⚖️", layout="wide")

st.title("⚖️ G-Eval Judge")
st.caption(
    "Score LLM outputs on custom dimensions using **chain-of-thought LLM-as-judge** (G-Eval, Liu et al. 2023). "
    "Select a rubric, enter your input/output pair, and get per-dimension scores with reasoning."
)
st.markdown("---")

RUBRICS = {
    "Response Quality": [
        {"name": "coherence",     "description": "Is the response logically structured, clear, and easy to follow?",                        "scale": "1-5", "weight": 1.0},
        {"name": "groundedness",  "description": "Are all claims supported by the provided context? No unsupported facts?",                  "scale": "1-5", "weight": 1.5},
        {"name": "helpfulness",   "description": "Does the response directly and completely address the user's question?",                   "scale": "1-5", "weight": 1.0},
    ],
    "Code Review": [
        {"name": "correctness",   "description": "Is the code logically correct and free of obvious bugs or edge-case failures?",            "scale": "1-5", "weight": 2.0},
        {"name": "readability",   "description": "Are variable names descriptive, structure clear, complexity minimised?",                   "scale": "1-5", "weight": 1.0},
        {"name": "security",      "description": "Does the code avoid injection, hardcoded secrets, insecure error handling?",               "scale": "1-5", "weight": 2.0},
        {"name": "efficiency",    "description": "Does the code avoid N+1 queries, unnecessary re-computation, or excessive memory use?",    "scale": "1-5", "weight": 1.0},
    ],
    "Customer Support": [
        {"name": "empathy",         "description": "Does the response acknowledge the customer's frustration and show genuine understanding?", "scale": "1-5", "weight": 1.5},
        {"name": "resolution",      "description": "Does the response provide a clear, actionable path to resolving the issue?",              "scale": "1-5", "weight": 2.0},
        {"name": "professionalism", "description": "Is the tone professional, respectful, concise, and on-brand?",                           "scale": "1-3", "weight": 1.0},
    ],
}

EXAMPLES = {
    "Response Quality": {
        "input":   "What is retrieval-augmented generation (RAG) and when should I use it?",
        "output":  "RAG is a technique that combines a retrieval system with a language model. Instead of relying solely on the model's training data, RAG fetches relevant documents from a knowledge base at inference time and uses them as context for the answer. You should use RAG when your application needs up-to-date information, domain-specific knowledge not in the model's training data, or when you need answers grounded in verifiable sources.",
        "context": "RAG (Retrieval-Augmented Generation) is an AI framework that retrieves relevant documents from an external knowledge base and provides them as context to an LLM at inference time.",
    },
    "Code Review": {
        "input":   "Write a Python function that checks if a number is prime.",
        "output":  "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0: return False\n    return True",
        "context": "",
    },
    "Customer Support": {
        "input":   "I've been waiting 3 weeks for my order and nobody has responded to my emails!",
        "output":  "I completely understand how frustrating this must be, and I sincerely apologise for the delay and lack of communication. That's not the experience we want for you. I'm looking into your order right now — can you provide your order number so I can give you an immediate update and escalate this to our fulfilment team?",
        "context": "",
    },
}

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Anthropic API Key", type="password",
        value=os.environ.get("ANTHROPIC_API_KEY",""))
    rubric_name = st.selectbox("Rubric", list(RUBRICS.keys()))
    st.markdown("---")
    st.markdown(f"**{rubric_name} dimensions:**")
    for d in RUBRICS[rubric_name]:
        st.markdown(f"• **{d['name']}** (scale {d['scale']}, weight {d['weight']}x)")

# ── inputs ────────────────────────────────────────────────────────────────────
ex = EXAMPLES[rubric_name]
col1, col2 = st.columns(2)
with col1:
    input_text  = st.text_area("Input / Question", value=ex["input"],  height=130)
    context     = st.text_area("Context (optional)", value=ex["context"], height=90)
with col2:
    output_text = st.text_area("LLM Output to Evaluate", value=ex["output"], height=230)

run = st.button("⚖️ Evaluate", type="primary", use_container_width=True)

# ── scoring ───────────────────────────────────────────────────────────────────
def build_prompt(dim, input_text, output_text, context):
    ctx_block = f"\n\nReference context:\n{context}" if context.strip() else ""
    return f"""You are an expert evaluator scoring an LLM response on the dimension of **{dim['name']}**.

Definition: {dim['description']}
Scale: {dim['scale']} ({dim['scale'].split('-')[0]} = worst, {dim['scale'].split('-')[1]} = best)

Input: {input_text}
Response to evaluate: {output_text}{ctx_block}

Think step-by-step, then output ONLY valid JSON:
{{"score": <integer {dim['scale']}>, "reasoning": "<one concise sentence>"}}"""

async def score_all(client, rubric, input_text, output_text, context):
    import anthropic as _ant
    results = []
    async with _ant.AsyncAnthropic(api_key=client.api_key) as aclient:
        tasks = [
            aclient.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=150,
                messages=[{"role":"user","content": build_prompt(d, input_text, output_text, context)}]
            )
            for d in rubric
        ]
        responses = await asyncio.gather(*tasks)
    for dim, resp in zip(rubric, responses):
        raw = resp.content[0].text.strip()
        raw = re.sub(r"^```[a-z]*\n?","",raw).rstrip("`").strip()
        data = json.loads(raw)
        results.append({**dim, "score": int(data["score"]), "reasoning": data["reasoning"]})
    return results

if run:
    if not api_key:
        st.error("Enter your Anthropic API key in the sidebar.")
        st.stop()

    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    rubric = RUBRICS[rubric_name]

    with st.spinner(f"Evaluating {len(rubric)} dimensions in parallel…"):
        try:
            scores = asyncio.run(score_all(client, rubric, input_text, output_text, context))
        except Exception as e:
            st.error(f"Evaluation failed: {e}")
            st.stop()

    # ── results ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader(f"📋 {rubric_name} — Evaluation Results")

    total_w = sum(d["weight"] for d in rubric)
    composite = sum((s["score"] / int(s["scale"].split("-")[1])) * s["weight"] for s in scores) / total_w

    comp_color = "#16a34a" if composite >= 0.7 else "#ca8a04" if composite >= 0.4 else "#dc2626"
    st.markdown(
        f"<h3 style='margin-bottom:4px'>Composite Score: "
        f"<span style='color:{comp_color}'>{composite:.3f} / 1.0</span></h3>",
        unsafe_allow_html=True
    )
    st.progress(composite)

    st.markdown("### Per-Dimension Breakdown")
    for s in scores:
        max_s = int(s["scale"].split("-")[1])
        ratio = s["score"] / max_s
        bar_color = "#16a34a" if ratio >= 0.7 else "#ca8a04" if ratio >= 0.4 else "#dc2626"
        c1, c2, c3 = st.columns([2, 3, 1])
        c1.markdown(f"**{s['name']}** `w={s['weight']}x`")
        c2.markdown(f"_{s['reasoning']}_")
        c3.markdown(
            f"<span style='background:{bar_color};color:#fff;padding:2px 10px;"
            f"border-radius:10px;font-weight:700'>{s['score']}/{max_s}</span>",
            unsafe_allow_html=True
        )
        st.progress(ratio)

    st.markdown("---")
    st.download_button(
        "⬇️ Download JSON",
        data=json.dumps({"rubric": rubric_name, "composite_score": composite, "dimension_scores": scores}, indent=2),
        file_name="geval_result.json", mime="application/json"
    )
