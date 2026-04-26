"""Tests for the transaction categorizer."""
import pandas as pd

from src.tools.categorizer import Categorizer, categorize_by_rules


class TestCategorizeByRules:
    def test_food(self):
        assert categorize_by_rules("Starbucks Downtown") == "Food & Dining"
        assert categorize_by_rules("Chipotle Mexican Grill") == "Food & Dining"

    def test_groceries(self):
        assert categorize_by_rules("Whole Foods Market") == "Groceries"
        assert categorize_by_rules("Trader Joes") == "Groceries"

    def test_transportation(self):
        assert categorize_by_rules("Uber Trip") == "Transportation"
        assert categorize_by_rules("Shell Gas Station") == "Transportation"

    def test_entertainment(self):
        assert categorize_by_rules("Netflix") == "Entertainment"
        assert categorize_by_rules("Spotify Premium") == "Entertainment"

    def test_income(self):
        assert categorize_by_rules("Paycheck Deposit") == "Income"

    def test_housing(self):
        assert categorize_by_rules("Rent Payment") == "Housing"

    def test_case_insensitive(self):
        assert categorize_by_rules("NETFLIX") == "Entertainment"
        assert categorize_by_rules("netflix") == "Entertainment"

    def test_no_match_returns_none(self):
        assert categorize_by_rules("Zzq Unknown Merchant") is None

    def test_empty_returns_none(self):
        assert categorize_by_rules("") is None


class TestCategorizer:
    def test_rule_based_only(self):
        cat = Categorizer()
        assert cat.categorize_one("Starbucks") == "Food & Dining"
        assert cat.categorize_one("Unknown XYZ") == "Other"

    def test_falls_back_to_llm(self):
        def fake_llm(desc: str) -> str:
            return "Shopping"

        cat = Categorizer(llm_classifier=fake_llm)
        # Rules don't match this, so LLM should be called
        assert cat.categorize_one("Mystery Merchant 12345") == "Shopping"

    def test_rules_take_priority(self):
        def fake_llm(desc: str) -> str:
            return "Shopping"  # should never be called

        cat = Categorizer(llm_classifier=fake_llm)
        # Rule matches, LLM not invoked
        assert cat.categorize_one("Starbucks") == "Food & Dining"

    def test_llm_exception_falls_back_to_default(self):
        def bad_llm(desc: str) -> str:
            raise RuntimeError("LLM down")

        cat = Categorizer(llm_classifier=bad_llm, default_category="Other")
        assert cat.categorize_one("Unknown Thing") == "Other"

    def test_categorize_df(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "description": ["Netflix", "Unknown Thing"],
            "amount": [-15.49, -20.00],
        })
        cat = Categorizer()
        out = cat.categorize_df(df)
        assert "category" in out.columns
        assert out.iloc[0]["category"] == "Entertainment"
        assert out.iloc[1]["category"] == "Other"
