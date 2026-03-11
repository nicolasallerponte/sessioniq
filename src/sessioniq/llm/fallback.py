"""
Rule-based fallback for when the Gemini API is unavailable.
Returns deterministic NudgeOutput based on urgency level and session context.
"""

from sessioniq.llm.prompt_builder import NudgeOutput, SessionContext, get_urgency_level

TEMPLATES = {
    "explore": [
        "You might also like these products other users explored. Take a look!",
        "Based on your browsing, here are some picks you may enjoy.",
    ],
    "nudge": [
        "Still deciding? Here are similar products — and we're offering {discount}% off for the next 10 minutes.",
        "These products match what you've been looking at. Use code SAVE{discount} at checkout.",
    ],
    "rescue": [
        "Wait! Get {discount}% off right now before you go. This offer expires in 5 minutes.",
        "Don't leave empty-handed — here's an exclusive {discount}% discount just for you.",
    ],
}

DISCOUNTS = {"explore": 0, "nudge": 7, "rescue": 15}
TONES = {"explore": "informative", "nudge": "friendly", "rescue": "urgent"}


def generate_fallback_nudge(ctx: SessionContext) -> NudgeOutput:
    """
    Generate a rule-based nudge without calling the LLM.
    Used when GEMINI_API_KEY is missing or the API call fails.
    """
    urgency = get_urgency_level(ctx.purchase_probability)
    discount = DISCOUNTS[urgency]
    templates = TEMPLATES[urgency]

    # Pick template based on session parity (deterministic, no randomness)
    template = templates[ctx.n_events % len(templates)]
    message = template.format(discount=discount)

    return NudgeOutput(
        message=message,
        tone=TONES[urgency],
        discount_pct=discount,
        urgency_level=urgency,
    )


if __name__ == "__main__":
    from sessioniq.llm.prompt_builder import SessionContext

    ctx = SessionContext(
        purchase_probability=0.12,
        n_events=3,
        n_carts=0,
        session_duration_seconds=90,
        top_shap_feature="n_events_observed",
        recommended_product_ids=[1005154, 1002100],
        avg_price=150.0,
    )

    result = generate_fallback_nudge(ctx)
    print(f"Urgency:  {result.urgency_level}")
    print(f"Tone:     {result.tone}")
    print(f"Discount: {result.discount_pct}%")
    print(f"Message:  {result.message}")
