"""
Unit tests for sessioniq core modules.
These tests run without the dataset — no parquet files or trained models needed.
"""

import pandas as pd
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_event(event_type="view", price=50.0, product_id=1, ts="2019-10-01 10:00:00"):
    return {
        "event_type": event_type,
        "product_id": product_id,
        "category_code": "electronics.phone",
        "brand": "samsung",
        "price": price,
        "ts": ts,
    }


def _compute_features(events):
    """Import here so tests fail clearly if the module has import errors."""
    from sessioniq.app.app import compute_features

    return compute_features(events)


# ── Feature computation ───────────────────────────────────────────────────────


class TestComputeFeatures:
    def test_single_view_event(self):
        events = [_make_event("view", price=100.0)]
        df = _compute_features(events)
        assert df["n_events_observed"].values[0] == 1
        assert df["n_views"].values[0] == 1
        assert df["n_carts"].values[0] == 0
        assert df["avg_price"].values[0] == pytest.approx(100.0)

    def test_cart_view_ratio(self):
        events = [
            _make_event("view", price=50.0, ts="2019-10-01 10:00:00"),
            _make_event("view", price=50.0, ts="2019-10-01 10:01:00"),
            _make_event("cart", price=50.0, ts="2019-10-01 10:02:00"),
        ]
        df = _compute_features(events)
        # cart_view_ratio = n_carts / (n_views + 1) = 1 / (2 + 1)
        assert df["cart_view_ratio"].values[0] == pytest.approx(1 / 3)

    def test_session_duration(self):
        events = [
            _make_event("view", ts="2019-10-01 10:00:00"),
            _make_event("view", ts="2019-10-01 10:05:00"),  # 300s later
        ]
        df = _compute_features(events)
        assert df["session_duration_seconds"].values[0] == pytest.approx(300.0)

    def test_zero_duration_single_event(self):
        events = [_make_event("view")]
        df = _compute_features(events)
        assert df["session_duration_seconds"].values[0] == 0.0

    def test_unique_products_counted(self):
        events = [
            _make_event("view", product_id=1),
            _make_event("view", product_id=1),
            _make_event("view", product_id=2),
        ]
        df = _compute_features(events)
        assert df["n_unique_products"].values[0] == 2

    def test_avg_price_excludes_zero(self):
        events = [
            _make_event("view", price=0.0),
            _make_event("view", price=100.0),
        ]
        df = _compute_features(events)
        # Zero price excluded from average
        assert df["avg_price"].values[0] == pytest.approx(100.0)

    def test_max_price(self):
        events = [
            _make_event("view", price=30.0),
            _make_event("view", price=99.0),
            _make_event("cart", price=45.0),
        ]
        df = _compute_features(events)
        assert df["max_price"].values[0] == pytest.approx(99.0)

    def test_removal_counted(self):
        events = [
            _make_event("view"),
            _make_event("cart"),
            _make_event("remove_from_cart"),
        ]
        df = _compute_features(events)
        assert df["n_removals"].values[0] == 1

    def test_seconds_to_first_cart(self):
        events = [
            _make_event("view", ts="2019-10-01 10:00:00"),
            _make_event("view", ts="2019-10-01 10:02:00"),
            _make_event("cart", ts="2019-10-01 10:03:00"),  # 180s after start
        ]
        df = _compute_features(events)
        assert df["seconds_to_first_cart"].values[0] == pytest.approx(180.0)

    def test_no_cart_seconds_to_first_cart_is_zero(self):
        events = [
            _make_event("view", ts="2019-10-01 10:00:00"),
            _make_event("view", ts="2019-10-01 10:05:00"),
        ]
        df = _compute_features(events)
        assert df["seconds_to_first_cart"].values[0] == 0.0

    def test_output_is_dataframe(self):
        events = [_make_event()]
        df = _compute_features(events)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_expected_columns_present(self):
        expected = {
            "n_events_observed",
            "n_views",
            "n_carts",
            "n_removals",
            "n_unique_products",
            "n_unique_categories",
            "n_unique_brands",
            "avg_price",
            "max_price",
            "session_duration_seconds",
            "cart_view_ratio",
            "seconds_to_first_cart",
        }
        df = _compute_features([_make_event()])
        assert expected.issubset(set(df.columns))


# ── LLM urgency thresholds ────────────────────────────────────────────────────


class TestUrgencyLevel:
    def setup_method(self):
        from sessioniq.llm.prompt_builder import get_urgency_level

        self.get_urgency_level = get_urgency_level

    def test_high_prob_is_explore(self):
        assert self.get_urgency_level(0.80) == "explore"

    def test_boundary_explore(self):
        assert self.get_urgency_level(0.40) == "explore"

    def test_medium_prob_is_nudge(self):
        assert self.get_urgency_level(0.30) == "nudge"

    def test_boundary_nudge(self):
        assert self.get_urgency_level(0.20) == "nudge"

    def test_low_prob_is_rescue(self):
        assert self.get_urgency_level(0.10) == "rescue"

    def test_zero_prob_is_rescue(self):
        assert self.get_urgency_level(0.0) == "rescue"

    def test_one_prob_is_explore(self):
        assert self.get_urgency_level(1.0) == "explore"


# ── Fallback nudge ────────────────────────────────────────────────────────────


class TestFallbackNudge:
    def _make_ctx(self, prob=0.15):
        from sessioniq.llm.prompt_builder import SessionContext

        return SessionContext(
            purchase_probability=prob,
            n_events=5,
            n_carts=1,
            session_duration_seconds=120.0,
            top_shap_feature="n_carts",
            recommended_product_ids=[101, 102],
            avg_price=49.99,
        )

    def test_fallback_returns_nudge_output(self):
        from sessioniq.llm.fallback import generate_fallback_nudge
        from sessioniq.llm.prompt_builder import NudgeOutput

        ctx = self._make_ctx(prob=0.15)
        result = generate_fallback_nudge(ctx)
        assert isinstance(result, NudgeOutput)

    def test_fallback_message_not_empty(self):
        from sessioniq.llm.fallback import generate_fallback_nudge

        result = generate_fallback_nudge(self._make_ctx())
        assert len(result.message) > 0

    def test_fallback_urgency_matches_prob(self):
        from sessioniq.llm.fallback import generate_fallback_nudge

        result = generate_fallback_nudge(self._make_ctx(prob=0.10))
        assert result.urgency_level == "rescue"

    def test_fallback_explore_no_discount(self):
        from sessioniq.llm.fallback import generate_fallback_nudge

        result = generate_fallback_nudge(self._make_ctx(prob=0.80))
        assert result.urgency_level == "explore"
        assert result.discount_pct == 0
