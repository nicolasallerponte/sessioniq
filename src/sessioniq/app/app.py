"""
SessionIQ — Flask web app for real-time purchase intent demo.
"""

import random
from pathlib import Path

import joblib
import pandas as pd
import polars as pl
from flask import Flask, jsonify, render_template_string, request

from sessioniq.llm.fallback import generate_fallback_nudge
from sessioniq.llm.prompt_builder import (
    SessionContext,
    generate_nudge,
    get_urgency_level,
)
from sessioniq.models.intent import FEATURE_COLS
from sessioniq.recommender.two_tower import load_recommender, recommend

app = Flask(__name__)

MODEL_DIR = Path("models")
PROCESSED_DIR = Path("data/processed")

# ── Load models once at startup ───────────────────────────────────────────────
print("Loading models…")
intent_model = joblib.load(MODEL_DIR / "intent_classifier.joblib")
rec_model, product2idx, idx2product = load_recommender()
sessions_df = pl.read_parquet(PROCESSED_DIR / "demo_sessions.parquet")
catalog_df = pl.read_parquet(PROCESSED_DIR / "demo_catalog.parquet")
session_ids = sessions_df["user_session"].unique().to_list()
print(f"Ready — {len(session_ids)} demo sessions, {len(catalog_df)} products")


def compute_features(events: list[dict]) -> pd.DataFrame:
    n = len(events)
    n_views = sum(1 for e in events if e["event_type"] == "view")
    n_carts = sum(1 for e in events if e["event_type"] == "cart")
    n_removals = sum(1 for e in events if e["event_type"] == "remove_from_cart")
    prices = [e["price"] for e in events if e["price"] > 0]
    t_start = pd.Timestamp(events[0]["ts"])
    t_end = pd.Timestamp(events[-1]["ts"])
    duration = (t_end - t_start).total_seconds() if n > 1 else 0
    cart_ts = [pd.Timestamp(e["ts"]) for e in events if e["event_type"] == "cart"]
    s_to_cart = (cart_ts[0] - t_start).total_seconds() if cart_ts else 0
    return pd.DataFrame(
        [
            {
                "n_events_observed": n,
                "n_views": n_views,
                "n_carts": n_carts,
                "n_removals": n_removals,
                "n_unique_products": len({e["product_id"] for e in events}),
                "n_unique_categories": len(
                    {e.get("category_code", "") for e in events}
                ),
                "n_unique_brands": len({e.get("brand", "") for e in events}),
                "avg_price": sum(prices) / len(prices) if prices else 0,
                "max_price": max(prices) if prices else 0,
                "session_duration_seconds": duration,
                "cart_view_ratio": n_carts / (n_views + 1),
                "seconds_to_first_cart": s_to_cart,
            }
        ]
    )


def get_product_info(pid: int) -> dict:
    info = catalog_df.filter(pl.col("product_id") == pid)
    if len(info) > 0:
        row = info.row(0, named=True)
        return {
            "brand": str(row.get("brand") or "unknown"),
            "category": str(row.get("category_code") or "unknown"),
            "price": float(row.get("price") or 0),
        }
    return {"brand": "unknown", "category": f"product {pid}", "price": 0.0}


# ── Routes ────────────────────────────────────────────────────────────────────


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/new_session")
def new_session():
    sid = random.choice(session_ids)
    events = (
        sessions_df.filter(pl.col("user_session") == sid)
        .sort("event_time")
        .select(
            [
                "event_type",
                "product_id",
                "category_code",
                "brand",
                "price",
                "event_time",
            ]
        )
        .to_dicts()
    )
    # Convert to serialisable format
    serialised = []
    for e in events:
        serialised.append(
            {
                "event_type": e["event_type"],
                "product_id": int(e["product_id"]),
                "category_code": str(e.get("category_code") or "unknown"),
                "brand": str(e.get("brand") or "unknown"),
                "price": float(e.get("price") or 0),
                "ts": str(e["event_time"]),
            }
        )
    return jsonify({"session_id": sid, "events": serialised})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    events = data["events"]  # list of events seen so far

    features = compute_features(events)
    prob = float(intent_model.predict_proba(features[FEATURE_COLS])[0][1])
    urgency = get_urgency_level(prob)

    # Recommendations based on last product
    last_pid = events[-1]["product_id"]
    recs_ids = recommend(last_pid, rec_model, product2idx, idx2product, top_k=3)
    recs = [{"id": pid, **get_product_info(pid)} for pid in recs_ids]

    # Nudge if needed
    nudge = None
    if urgency in ("nudge", "rescue"):
        ctx = SessionContext(
            purchase_probability=prob,
            n_events=len(events),
            n_carts=int(features["n_carts"].values[0]),
            session_duration_seconds=float(
                features["session_duration_seconds"].values[0]
            ),
            top_shap_feature="n_events_observed",
            recommended_product_ids=recs_ids,
            avg_price=float(features["avg_price"].values[0]),
        )
        try:
            result = generate_nudge(ctx)
            nudge = {
                "message": result.message,
                "tone": result.tone,
                "discount_pct": result.discount_pct,
                "urgency_level": result.urgency_level,
            }
        except Exception:
            result = generate_fallback_nudge(ctx)
            nudge = {
                "message": result.message,
                "tone": result.tone,
                "discount_pct": result.discount_pct,
                "urgency_level": result.urgency_level,
            }

    # Product info for the event
    product = get_product_info(last_pid)

    return jsonify(
        {
            "probability": prob,
            "urgency": urgency,
            "recommendations": recs,
            "nudge": nudge,
            "product": product,
        }
    )


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SessionIQ</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700&family=Geist+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root {
  --bg:       #fafafa;
  --surface:  #ffffff;
  --border:   #e8eaed;
  --border2:  #f0f2f4;
  --text:     #111827;
  --muted:    #6b7280;
  --subtle:   #9ca3af;
  --green:    #059669;
  --green-bg: #ecfdf5;
  --green-br: #a7f3d0;
  --amber:    #b45309;
  --amber-bg: #fffbeb;
  --amber-br: #fde68a;
  --red:      #dc2626;
  --red-bg:   #fef2f2;
  --red-br:   #fecaca;
  --accent:   #2563eb;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
html { font-size: 15px; }
body {
  font-family: 'Syne', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}

/* ── Onboarding ─────────────────────────────────────────────── */
#onboarding {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 2rem;
  background: var(--bg);
}
.ob-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 2.5rem;
  max-width: 520px;
  width: 100%;
  box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 8px 32px rgba(0,0,0,0.06);
}
.ob-logo {
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.03em;
  color: var(--text);
  margin-bottom: 0.4rem;
}
.ob-sub {
  font-size: 0.875rem;
  color: var(--muted);
  line-height: 1.6;
  margin-bottom: 2rem;
}
.ob-step {
  display: flex;
  gap: 1rem;
  padding: 0.9rem 0;
  border-top: 1px solid var(--border2);
}
.ob-step:last-of-type { border-bottom: 1px solid var(--border2); margin-bottom: 1.5rem; }
.ob-num {
  width: 24px; height: 24px;
  border-radius: 50%;
  background: var(--text);
  color: #fff;
  font-size: 0.7rem;
  font-weight: 700;
  font-family: 'Geist Mono', monospace;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.ob-step strong { display: block; font-size: 0.875rem; font-weight: 600; margin-bottom: 0.2rem; }
.ob-step span   { font-size: 0.8rem; color: var(--muted); line-height: 1.5; }
.ob-footer { font-size: 0.72rem; color: var(--subtle); font-family: 'Geist Mono', monospace; margin-bottom: 1.5rem; }
.ob-btns { display: flex; gap: 0.6rem; }

.btn {
  padding: 0.6rem 1.35rem;
  border-radius: 8px;
  border: none;
  cursor: pointer;
  font-size: 0.85rem;
  font-weight: 600;
  font-family: 'Syne', sans-serif;
  transition: all 0.15s;
}
.btn:active { transform: scale(0.97); }
.btn-dark  { background: var(--text); color: #fff; }
.btn-dark:hover { background: #1f2937; }
.btn-light { background: var(--border2); color: var(--muted); border: 1px solid var(--border); }
.btn-light:hover { background: var(--border); color: var(--text); }

/* ── App ─────────────────────────────────────────────────────── */
#app { display: none; flex-direction: column; min-height: 100vh; }

header {
  height: 52px;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0 1.5rem;
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  position: sticky; top: 0; z-index: 50;
}
.h-logo { font-size: 1.05rem; font-weight: 700; letter-spacing: -0.02em; flex-shrink: 0; }
.h-sep  { width: 1px; height: 16px; background: var(--border); }
.h-tag  {
  font-family: 'Geist Mono', monospace;
  font-size: 0.65rem;
  color: var(--subtle);
}
.h-ctrls { display: flex; gap: 0.4rem; margin-left: auto; }
.ctrl {
  height: 32px;
  padding: 0 0.85rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--muted);
  font-size: 0.78rem;
  font-weight: 500;
  font-family: 'Syne', sans-serif;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
  display: flex; align-items: center; gap: 0.3rem;
}
.ctrl:hover   { border-color: #9ca3af; color: var(--text); }
.ctrl:active  { transform: scale(0.97); }
.ctrl:disabled { opacity: 0.35; cursor: not-allowed; }
.ctrl.primary { background: var(--text); color: #fff; border-color: var(--text); }
.ctrl.primary:hover { background: #1f2937; }
.ctrl.stop    { background: var(--red-bg); color: var(--red); border-color: var(--red-br); }

/* ── Layout ──────────────────────────────────────────────────── */
main {
  display: grid;
  grid-template-columns: 1fr 1fr;
  flex: 1;
  gap: 0;
}
.panel {
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--border);
  background: var(--surface);
  min-height: calc(100vh - 52px);
}
.panel:last-child { border-right: none; background: var(--bg); }

.p-head {
  padding: 1.1rem 1.4rem 0.85rem;
  border-bottom: 1px solid var(--border);
}
.p-eyebrow {
  font-family: 'Geist Mono', monospace;
  font-size: 0.6rem;
  font-weight: 500;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--subtle);
  margin-bottom: 0.25rem;
}
.p-title { font-size: 1rem; font-weight: 700; letter-spacing: -0.02em; }
.p-desc  { font-size: 0.75rem; color: var(--muted); margin-top: 0.25rem; line-height: 1.5; }

/* ── Stats bar ───────────────────────────────────────────────── */
#stats-bar {
  display: none;
  border-bottom: 1px solid var(--border);
  background: var(--bg);
}
.stats-inner { display: flex; }
.stat-cell {
  flex: 1;
  padding: 0.55rem 1rem;
  border-right: 1px solid var(--border);
}
.stat-cell:last-child { border-right: none; }
.stat-lbl { font-family: 'Geist Mono', monospace; font-size: 0.58rem; color: var(--subtle); text-transform: uppercase; letter-spacing: 0.08em; }
.stat-val { font-family: 'Geist Mono', monospace; font-size: 0.95rem; font-weight: 500; color: var(--text); margin-top: 1px; }

/* ── Events ──────────────────────────────────────────────────── */
.event-list { flex: 1; overflow-y: auto; }
.empty-state {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  min-height: 320px; color: var(--subtle); text-align: center; padding: 2rem; gap: 0.6rem;
}
.empty-icon { font-size: 2rem; opacity: 0.35; }
.empty-state p { font-size: 0.82rem; line-height: 1.7; color: var(--subtle); }
.empty-state strong { color: var(--muted); }

.ev {
  display: flex; align-items: center; gap: 0.65rem;
  padding: 0.6rem 1.4rem;
  border-bottom: 1px solid var(--border2);
  font-size: 0.8rem;
  transition: background 0.1s;
  animation: fadeSlide 0.18s ease;
}
@keyframes fadeSlide { from { opacity:0; transform:translateY(-4px); } to { opacity:1; transform:none; } }
.ev:hover { background: var(--bg); }

.ev-n    { font-family: 'Geist Mono', monospace; font-size: 0.63rem; color: var(--border); width: 20px; flex-shrink:0; text-align:right; }
.ev-icon { font-size: 0.9rem; flex-shrink:0; }
.ev-tag  {
  font-family: 'Geist Mono', monospace; font-size: 0.62rem; font-weight: 500;
  padding: 2px 6px; border-radius: 3px; flex-shrink:0;
}
.t-view     { background:#eff6ff; color:#2563eb; }
.t-cart     { background:#fffbeb; color:#b45309; }
.t-purchase { background:#ecfdf5; color:#059669; }
.t-remove   { background:#fef2f2; color:#dc2626; }

.ev-info  { flex:1; min-width:0; }
.ev-brand { font-weight: 600; font-size: 0.8rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ev-cat   { font-size: 0.7rem; color: var(--subtle); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ev-price { font-family: 'Geist Mono', monospace; font-size: 0.73rem; color: var(--muted); flex-shrink:0; }

/* ── Right panel ─────────────────────────────────────────────── */
.r-section { padding: 1.25rem 1.4rem; border-bottom: 1px solid var(--border); }

.gauge-wrap {
  display: flex; flex-direction: column; align-items: center;
  padding: 1.4rem 1.4rem 1rem;
  border-bottom: 1px solid var(--border);
}
.gauge-svg-wrap { position: relative; width: 200px; }
.gauge-svg { width: 100%; }
.gauge-center {
  position: absolute;
  top: 54%; left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}
.gauge-pct {
  font-family: 'Geist Mono', monospace;
  font-size: 2rem; font-weight: 500; line-height: 1;
  transition: color 0.4s;
}
.gauge-sub { font-size: 0.62rem; color: var(--subtle); font-family: 'Geist Mono', monospace; margin-top: 3px; letter-spacing: 0.05em; }

.status-pill {
  margin-top: 1rem;
  padding: 5px 14px;
  border-radius: 999px;
  font-size: 0.73rem;
  font-weight: 600;
  font-family: 'Geist Mono', monospace;
  border: 1px solid transparent;
  transition: all 0.4s;
}
.s-explore { background: var(--green-bg); color: var(--green); border-color: var(--green-br); }
.s-nudge   { background: var(--amber-bg); color: var(--amber); border-color: var(--amber-br); }
.s-rescue  { background: var(--red-bg);   color: var(--red);   border-color: var(--red-br); }

.gauge-hint { margin-top: 0.5rem; font-size: 0.72rem; color: var(--subtle); text-align: center; }

/* ── Recs & Nudge ────────────────────────────────────────────── */
.sec-eyebrow {
  font-family: 'Geist Mono', monospace;
  font-size: 0.58rem; font-weight: 500;
  letter-spacing: 0.12em; text-transform: uppercase;
  color: var(--subtle); margin-bottom: 0.5rem;
}
.sec-desc { font-size: 0.74rem; color: var(--subtle); margin-bottom: 0.75rem; line-height: 1.4; }

.rec-item {
  display: flex; align-items: center; gap: 0.5rem;
  padding: 0.45rem 0.7rem;
  border: 1px solid var(--border);
  border-radius: 7px;
  margin-bottom: 0.3rem;
  background: var(--surface);
  font-size: 0.8rem;
  transition: border-color 0.15s;
}
.rec-item:hover { border-color: #9ca3af; }
.rec-arr   { color: var(--accent); font-weight: 700; font-size: 0.9rem; }
.rec-info  { flex: 1; }
.rec-brand { font-weight: 600; }
.rec-cat   { font-size: 0.7rem; color: var(--subtle); }
.rec-price { font-family: 'Geist Mono', monospace; font-size: 0.73rem; color: var(--muted); }

.nudge-box {
  border-radius: 8px; padding: 0.9rem 1rem;
  font-size: 0.84rem; line-height: 1.65; font-weight: 500;
  margin-top: 0.4rem; border: 1px solid transparent;
}
.nudge-meta { font-size: 0.68rem; color: var(--subtle); font-family: 'Geist Mono', monospace; margin-top: 0.4rem; }
.nudge-explore { background: var(--green-bg); border-color: var(--green-br); color: var(--green); }
.nudge-nudge   { background: var(--amber-bg); border-color: var(--amber-br); color: var(--amber); }
.nudge-rescue  { background: var(--red-bg);   border-color: var(--red-br);   color: var(--red); }

.no-nudge {
  font-size: 0.78rem; color: var(--subtle);
  padding: 0.75rem 1rem;
  background: var(--bg);
  border: 1px dashed var(--border);
  border-radius: 7px;
  font-family: 'Geist Mono', monospace;
  line-height: 1.5;
}
</style>
</head>
<body>

<!-- Onboarding -->
<div id="onboarding">
  <div class="ob-card">
    <div class="ob-logo">🛒 SessionIQ</div>
    <p class="ob-sub">Watch a machine learning system predict whether a user will buy — in real time, click by click.</p>

    <div class="ob-step">
      <div class="ob-num">1</div>
      <div>
        <strong>Start a session</strong>
        <span>Loads a real anonymised shopping session from a 2019 e-commerce dataset (13M events, 166k products).</span>
      </div>
    </div>
    <div class="ob-step">
      <div class="ob-num">2</div>
      <div>
        <strong>Advance click by click</strong>
        <span>Each click simulates one user action: view, add to cart, remove. Use Autopilot to play it out automatically.</span>
      </div>
    </div>
    <div class="ob-step">
      <div class="ob-num">3</div>
      <div>
        <strong>Watch the intent gauge</strong>
        <span>A LightGBM model trained on 9M sessions updates the purchase probability after every click. ROC-AUC 0.86.</span>
      </div>
    </div>
    <div class="ob-step">
      <div class="ob-num">4</div>
      <div>
        <strong>LLM nudge fires automatically</strong>
        <span>When abandonment risk is detected, llama3.2 (via Ollama, runs locally) generates a personalised recovery message.</span>
      </div>
    </div>

    <p class="ob-footer">// All inference runs locally · No data leaves your machine</p>
    <div class="ob-btns">
      <button class="btn btn-dark" onclick="startDemo()">▶ Start demo</button>
      <button class="btn btn-light" onclick="skipIntro()">Skip intro</button>
    </div>
  </div>
</div>

<!-- App -->
<div id="app">
  <header>
    <span class="h-logo">🛒 SessionIQ</span>
    <div class="h-sep"></div>
    <span class="h-tag">purchase intent demo</span>
    <div class="h-ctrls">
      <button class="ctrl primary" onclick="newSession()">▶ New session</button>
      <button class="ctrl" id="btn-next" disabled onclick="nextClick()">⏭ Next click</button>
      <button class="ctrl" id="btn-auto" disabled onclick="toggleAutopilot()">▶▶ Autopilot</button>
      <button class="ctrl" onclick="showOnboarding()">? How it works</button>
    </div>
  </header>

  <main>
    <!-- Left: timeline -->
    <div class="panel">
      <div class="p-head">
        <div class="p-eyebrow">Session Timeline</div>
        <div class="p-title">User actions, click by click</div>
        <div class="p-desc">Each row is one action the user took. The model updates its prediction after every click.</div>
      </div>
      <div id="stats-bar">
        <div class="stats-inner">
          <div class="stat-cell"><div class="stat-lbl">Clicks</div><div class="stat-val" id="s-clicks">0</div></div>
          <div class="stat-cell"><div class="stat-lbl">Views</div><div class="stat-val" id="s-views">0</div></div>
          <div class="stat-cell"><div class="stat-lbl">Carts</div><div class="stat-val" id="s-carts">0</div></div>
          <div class="stat-cell"><div class="stat-lbl">Avg price</div><div class="stat-val" id="s-price">—</div></div>
        </div>
      </div>
      <div class="event-list" id="event-list">
        <div class="empty-state">
          <div class="empty-icon">🖱️</div>
          <p>Press <strong>New session</strong> to load a real shopping session,<br>then <strong>Next click</strong> or <strong>Autopilot</strong> to advance.</p>
        </div>
      </div>
    </div>

    <!-- Right: intent brain -->
    <div class="panel">
      <div class="p-head">
        <div class="p-eyebrow">Intent Brain</div>
        <div class="p-title">What the model thinks right now</div>
        <div class="p-desc">LightGBM trained on 9M sessions predicts purchase probability from behaviour — no identity, no cookies.</div>
      </div>

      <div class="gauge-wrap">
        <div class="gauge-svg-wrap">
          <svg class="gauge-svg" viewBox="0 0 200 110">
            <!-- bg arc -->
            <path d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none" stroke="#f0f2f4" stroke-width="14" stroke-linecap="round"/>
            <!-- value arc -->
            <path id="gauge-arc" d="M 20 100 A 80 80 0 0 1 180 100"
              fill="none" stroke="#059669" stroke-width="14" stroke-linecap="round"
              stroke-dasharray="251.2" stroke-dashoffset="251.2"
              style="transition: stroke-dashoffset 0.5s ease, stroke 0.5s ease"/>
          </svg>
          <div class="gauge-center">
            <div class="gauge-pct" id="gauge-pct" style="color:#059669">0%</div>
            <div class="gauge-sub">INTENT SCORE</div>
          </div>
        </div>
        <div class="status-pill s-explore" id="gauge-status">🟢 Browsing — low intent</div>
        <div class="gauge-hint" id="gauge-hint">Start a session to see the model in action.</div>
      </div>

      <div class="r-section" id="recs-section" style="display:none">
        <div class="sec-eyebrow">Recommendations</div>
        <div class="sec-desc">ALS collaborative filtering — products similar users viewed alongside the last item.</div>
        <div id="recs-list"></div>
      </div>

      <div class="r-section" id="nudge-section" style="display:none">
        <div class="sec-eyebrow">LLM Nudge</div>
        <div class="sec-desc">Generated by llama3.2 via Ollama. Fires when abandonment risk is detected (probability &lt; 40%).</div>
        <div class="nudge-box" id="nudge-box"></div>
        <div class="nudge-meta" id="nudge-meta"></div>
      </div>

      <div class="r-section" id="no-nudge-section" style="display:none">
        <div class="no-nudge">// No nudge needed — user is still exploring comfortably.<br>LLM fires when probability drops below 40%.</div>
      </div>
    </div>
  </main>
</div>

<script>
  let allEvents = [], shownCount = 0, autopilotTimer = null, isAutopilot = false;
  let nViews = 0, nCarts = 0, totalPrice = 0;

  const U = {
    explore: { label: '🟢 Browsing — low intent',     pill: 's-explore', color: '#059669' },
    nudge:   { label: '🟡 Hesitating — medium intent', pill: 's-nudge',   color: '#b45309' },
    rescue:  { label: '🔴 About to leave — high risk', pill: 's-rescue',  color: '#dc2626' },
  };
  const TYPE_ICON  = { view:'👁', cart:'🛒', purchase:'✅', remove_from_cart:'✕' };
  const TYPE_CLS   = { view:'t-view', cart:'t-cart', purchase:'t-purchase', remove_from_cart:'t-remove' };

  function startDemo()    { hide('onboarding'); show('app'); newSession(); }
  function skipIntro()    { hide('onboarding'); show('app'); }
  function showOnboarding(){ stopAutopilot(); hide('app'); show('onboarding'); }
  function show(id) { document.getElementById(id).style.display = id === 'app' ? 'flex' : 'flex'; }
  function hide(id) { document.getElementById(id).style.display = 'none'; }

  async function newSession() {
    stopAutopilot();
    allEvents = []; shownCount = 0; nViews = 0; nCarts = 0; totalPrice = 0;
    resetUI();
    const r = await fetch('/api/new_session');
    const d = await r.json();
    allEvents = d.events;
    document.getElementById('btn-next').disabled = false;
    document.getElementById('btn-auto').disabled = false;
  }

  async function nextClick() {
    if (shownCount >= allEvents.length) { stopAutopilot(); return; }
    const e = allEvents[shownCount];
    shownCount++;

    // Stats
    if (e.event_type === 'view') nViews++;
    if (e.event_type === 'cart') nCarts++;
    if (e.price > 0) totalPrice += e.price;
    updateStats();
    addEvent(e, shownCount);

    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events: allEvents.slice(0, shownCount) }),
    });
    const d = await r.json();
    updateGauge(d.probability, d.urgency);
    updateRecs(d.recommendations);
    updateNudge(d.nudge, d.urgency);

    if (shownCount >= allEvents.length) { stopAutopilot(); document.getElementById('btn-next').disabled = true; }
  }

  function toggleAutopilot() {
    isAutopilot ? stopAutopilot() : startAutopilot();
  }
  function startAutopilot() {
    if (shownCount >= allEvents.length) return;
    isAutopilot = true;
    const btn = document.getElementById('btn-auto');
    btn.textContent = '⏹ Stop'; btn.classList.add('stop');
    document.getElementById('btn-next').disabled = true;
    runAutopilot();
  }
  async function runAutopilot() {
    if (!isAutopilot || shownCount >= allEvents.length) { stopAutopilot(); return; }
    await nextClick();
    if (isAutopilot && shownCount < allEvents.length) autopilotTimer = setTimeout(runAutopilot, 1200);
    else stopAutopilot();
  }
  function stopAutopilot() {
    isAutopilot = false; clearTimeout(autopilotTimer);
    const btn = document.getElementById('btn-auto');
    if (btn) { btn.textContent = '▶▶ Autopilot'; btn.classList.remove('stop'); }
    const bn = document.getElementById('btn-next');
    if (bn && shownCount < allEvents.length) bn.disabled = false;
  }

  function resetUI() {
    document.getElementById('event-list').innerHTML = `
      <div class="empty-state">
        <div class="empty-icon">🖱️</div>
        <p>Press <strong>New session</strong> to load a real shopping session,<br>then <strong>Next click</strong> or <strong>Autopilot</strong> to advance.</p>
      </div>`;
    document.getElementById('stats-bar').style.display = 'none';
    document.getElementById('recs-section').style.display = 'none';
    document.getElementById('nudge-section').style.display = 'none';
    document.getElementById('no-nudge-section').style.display = 'none';
    document.getElementById('btn-next').disabled = true;
    document.getElementById('btn-auto').disabled = true;
    updateGauge(0, 'explore');
    document.getElementById('gauge-hint').textContent = 'Start a session to see the model in action.';
  }

  function updateStats() {
    document.getElementById('stats-bar').style.display = 'block';
    document.getElementById('s-clicks').textContent = shownCount;
    document.getElementById('s-views').textContent  = nViews;
    document.getElementById('s-carts').textContent  = nCarts;
    const avg = shownCount > 0 ? totalPrice / shownCount : 0;
    document.getElementById('s-price').textContent  = avg > 0 ? '€' + avg.toFixed(0) : '—';
  }

  function addEvent(e, n) {
    const list = document.getElementById('event-list');
    const empty = list.querySelector('.empty-state');
    if (empty) empty.remove();

    const div = document.createElement('div');
    div.className = 'ev';
    div.innerHTML = `
      <span class="ev-n">${n}</span>
      <span class="ev-icon">${TYPE_ICON[e.event_type] || '•'}</span>
      <span class="ev-tag ${TYPE_CLS[e.event_type] || 't-view'}">${e.event_type.replace(/_/g,' ')}</span>
      <div class="ev-info">
        <div class="ev-brand">${e.brand}</div>
        <div class="ev-cat">${e.category_code}</div>
      </div>
      <span class="ev-price">€${e.price.toFixed(2)}</span>`;
    list.insertBefore(div, list.firstChild);
  }

  function updateGauge(prob, urgency) {
    const pct = Math.round(prob * 100);
    const u = U[urgency] || U.explore;
    const total = 251.2;
    const arc = document.getElementById('gauge-arc');
    arc.style.strokeDashoffset = total * (1 - prob);
    arc.style.stroke = u.color;
    const pctEl = document.getElementById('gauge-pct');
    pctEl.textContent = pct + '%';
    pctEl.style.color = u.color;
    const st = document.getElementById('gauge-status');
    st.textContent = u.label;
    st.className = 'status-pill ' + u.pill;
    if (prob > 0) document.getElementById('gauge-hint').textContent = 'Nudge fires automatically when probability drops below 40%.';
  }

  function updateRecs(recs) {
    if (!recs || !recs.length) return;
    document.getElementById('recs-section').style.display = 'block';
    document.getElementById('recs-list').innerHTML = recs.map(r => `
      <div class="rec-item">
        <span class="rec-arr">→</span>
        <div class="rec-info">
          <div class="rec-brand">${r.brand}</div>
          <div class="rec-cat">${r.category}</div>
        </div>
        <span class="rec-price">€${r.price.toFixed(2)}</span>
      </div>`).join('');
  }

  function updateNudge(nudge, urgency) {
    const sec  = document.getElementById('nudge-section');
    const noSec = document.getElementById('no-nudge-section');
    if (!nudge) {
      sec.style.display = 'none';
      noSec.style.display = shownCount > 0 && urgency === 'explore' ? 'block' : 'none';
      return;
    }
    noSec.style.display = 'none';
    sec.style.display = 'block';
    const disc = nudge.discount_pct > 0 ? ` · <strong>${nudge.discount_pct}% off</strong>` : '';
    document.getElementById('nudge-box').innerHTML = '💬 ' + nudge.message + disc;
    document.getElementById('nudge-box').className = 'nudge-box nudge-' + nudge.urgency_level;
    document.getElementById('nudge-meta').textContent = `tone: ${nudge.tone} · urgency: ${nudge.urgency_level}`;
  }
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5050)
