"""Tests for the LangGraph finance pipeline (mocked LLM)."""
from unittest.mock import patch

import pandas as pd
import pytest

from src.agents.graph import anomaly_node, build_finance_graph, categorize_node, summary_node


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-01-05", "2026-01-10", "2026-01-15"]),
        "description": ["Starbucks", "Paycheck", "Amazon"],
        "amount": [-6.50, 2000.0, -40.0],
    })


class TestCategorizeNode:
    def test_without_llm(self, sample_df):
        result = categorize_node({"raw_df": sample_df, "use_llm_categorizer": False})
        df = result["categorized_df"]
        assert "category" in df.columns
        assert df.iloc[0]["category"] == "Food & Dining"

    def test_use_llm_flag_respected(self, sample_df):
        # Even with LLM disabled, rules still work
        result = categorize_node({"raw_df": sample_df, "use_llm_categorizer": False})
        assert "category" in result["categorized_df"].columns


class TestAnomalyNode:
    def test_runs_on_categorized(self, sample_df):
        sample_df["category"] = ["Food & Dining", "Income", "Shopping"]
        result = anomaly_node({"categorized_df": sample_df})
        assert "anomalies" in result
        assert isinstance(result["anomalies"], list)


class TestSummaryNode:
    def test_produces_summaries(self, sample_df):
        sample_df["category"] = ["Food & Dining", "Income", "Shopping"]
        result = summary_node({"categorized_df": sample_df})
        assert len(result["summaries"]) == 1
        assert result["summaries"][0].month == "2026-01"


class TestFullGraph:
    def test_end_to_end(self, sample_df):
        graph = build_finance_graph()
        final = graph.invoke({"raw_df": sample_df, "use_llm_categorizer": False})
        assert "categorized_df" in final
        assert "anomalies" in final
        assert "summaries" in final
        assert len(final["summaries"]) >= 1
