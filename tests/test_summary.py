"""Tests for monthly summary aggregation."""
import pandas as pd

from src.tools.summary import monthly_summary


def _make_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    return df


class TestMonthlySummary:
    def test_empty_df(self):
        assert monthly_summary(pd.DataFrame()) == []

    def test_single_month(self):
        df = _make_df([
            {"date": "2026-01-05", "description": "Paycheck", "amount": 2000.0, "category": "Income"},
            {"date": "2026-01-10", "description": "Starbucks", "amount": -6.0, "category": "Food & Dining"},
            {"date": "2026-01-15", "description": "Amazon", "amount": -50.0, "category": "Shopping"},
        ])
        summaries = monthly_summary(df)
        assert len(summaries) == 1
        s = summaries[0]
        assert s.month == "2026-01"
        assert s.total_income == 2000.0
        assert s.total_spending == 56.0
        assert s.net == 1944.0
        assert s.spending_by_category == {"Food & Dining": 6.0, "Shopping": 50.0}

    def test_multiple_months_sorted(self):
        df = _make_df([
            {"date": "2026-02-01", "description": "X", "amount": -50.0, "category": "Other"},
            {"date": "2026-01-01", "description": "Y", "amount": -20.0, "category": "Other"},
        ])
        summaries = monthly_summary(df)
        assert len(summaries) == 2
        assert summaries[0].month == "2026-01"
        assert summaries[1].month == "2026-02"

    def test_filter_by_month(self):
        df = _make_df([
            {"date": "2026-01-05", "description": "X", "amount": -10.0, "category": "Other"},
            {"date": "2026-02-10", "description": "Y", "amount": -20.0, "category": "Other"},
        ])
        result = monthly_summary(df, month="2026-02")
        assert len(result) == 1
        assert result[0].month == "2026-02"

    def test_savings_rate(self):
        df = _make_df([
            {"date": "2026-01-01", "description": "Pay", "amount": 1000.0, "category": "Income"},
            {"date": "2026-01-02", "description": "Rent", "amount": -800.0, "category": "Housing"},
        ])
        s = monthly_summary(df)[0]
        assert s.savings_rate == 0.2  # (1000 - 800) / 1000

    def test_zero_income_savings_rate(self):
        df = _make_df([
            {"date": "2026-01-01", "description": "Rent", "amount": -800.0, "category": "Housing"},
        ])
        s = monthly_summary(df)[0]
        assert s.savings_rate == 0.0

    def test_top_merchants(self):
        df = _make_df([
            {"date": "2026-01-01", "description": "A", "amount": -100.0, "category": "X"},
            {"date": "2026-01-02", "description": "B", "amount": -50.0, "category": "X"},
            {"date": "2026-01-03", "description": "A", "amount": -30.0, "category": "X"},
        ])
        s = monthly_summary(df)[0]
        # A totals 130, B totals 50
        assert s.top_merchants[0][0] == "A"
        assert s.top_merchants[0][1] == 130.0
