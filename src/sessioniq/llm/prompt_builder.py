"""
Dynamic prompt builder for purchase intent nudges.
Generates personalised marketing messages via Gemini 1.5 Pro.

Urgency levels based on purchase probability:
- explore  (p >= 0.40): soft recommendation, no discount
- nudge    (0.20 <= p < 0.40): moderate push, small discount
- rescue   (p < 0.20): urgent intervention, larger discount
"""

import json
from dataclasses import dataclass

import httpx
from dotenv import load_dotenv

load_dotenv()

URGENCY_THRESHOLDS = {
    "explore": 0.40,
    "nudge": 0.20,
}


@dataclass
class SessionContext:
    """Snapshot of a user session at inference time."""

    purchase_probability: float
    n_events: int
    n_carts: int
    session_duration_seconds: float
    top_shap_feature: str
    recommended_product_ids: list[int]
    avg_price: float


@dataclass
class NudgeOutput:
    """Structured output from the LLM."""

    message: str
    tone: str
    discount_pct: int
    urgency_level: str


def get_urgency_level(probability: float) -> str:
    if probability >= URGENCY_THRESHOLDS["explore"]:
        return "explore"
    elif probability >= URGENCY_THRESHOLDS["nudge"]:
        return "nudge"
    else:
        return "rescue"


def build_prompt(ctx: SessionContext, urgency: str) -> str:
    duration_min = ctx.session_duration_seconds / 60

    urgency_instructions = {
        "explore": (
            "The user is browsing actively. "
            "Suggest the recommended products naturally. "
            "No discount needed — keep the tone friendly and informative. "
            "discount_pct must be 0."
        ),
        "nudge": (
            "The user shows some interest but may leave. "
            "Highlight the recommended products and offer a small incentive. "
            "discount_pct should be between 5 and 10."
        ),
        "rescue": (
            "The user is very likely to abandon. "
            "Create urgency. Offer a meaningful discount. "
            "Keep the message short and compelling. "
            "discount_pct should be between 10 and 20."
        ),
    }

    return f"""You are a conversion optimisation expert for an e-commerce platform.

SESSION CONTEXT:
- Purchase probability: {ctx.purchase_probability:.1%}
- Events so far: {ctx.n_events}
- Items in cart: {ctx.n_carts}
- Time on site: {duration_min:.1f} minutes
- Key signal: {ctx.top_shap_feature}
- Recommended product IDs: {ctx.recommended_product_ids}
- Average price browsed: €{ctx.avg_price:.2f}

URGENCY LEVEL: {urgency.upper()}
INSTRUCTIONS: {urgency_instructions[urgency]}

Generate a pop-up message for this user. Respond ONLY with a JSON object, no markdown, no explanation:
{{
  "message": "2-line max pop-up message, friendly and natural",
  "tone": "one of: friendly / urgent / informative",
  "discount_pct": <integer>,
  "urgency_level": "{urgency}"
}}"""


def generate_nudge(ctx: SessionContext) -> NudgeOutput:
    urgency = get_urgency_level(ctx.purchase_probability)
    prompt = build_prompt(ctx, urgency)

    response = httpx.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
        },
        timeout=60,
    )
    raw = response.json()["response"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    data = json.loads(raw)
    return NudgeOutput(
        message=data["message"],
        tone=data["tone"],
        discount_pct=int(data["discount_pct"]),
        urgency_level=data["urgency_level"],
    )


if __name__ == "__main__":
    ctx = SessionContext(
        purchase_probability=0.15,
        n_events=4,
        n_carts=0,
        session_duration_seconds=180,
        top_shap_feature="session_duration_seconds",
        recommended_product_ids=[1005154, 1002100, 1002098],
        avg_price=320.0,
    )
    result = generate_nudge(ctx)
    print(f"Urgency:  {result.urgency_level}")
    print(f"Tone:     {result.tone}")
    print(f"Discount: {result.discount_pct}%")
    print(f"Message:  {result.message}")
