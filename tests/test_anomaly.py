"""Tests for anomaly detection."""
import pandas as pd
import pytest

from src.tools.anomaly import detect_anomalies


@pytest.fixture
def normal_transactions():
    """Baseline normal transactions."""
    rows = []
    for i in range(20):
        rows.append({
            "date": pd.Timestamp(f"2026-01-{(i % 28) + 1:02d}"),
            "description": "Starbucks",
            "amount": -6.50,
            "category": "Food & Dining",
        })
    return pd.DataFrame(rows)


class TestDetectAnomalies:
    def test_empty_df(self):
        assert detect_anomalies(pd.DataFrame()) == []

    def test_no_anomalies_in_consistent_data(self, normal_transactions):
        anomalies = detect_anomalies(normal_transactions)
        # With consistent data, we shouldn't get high-severity anomalies
        high_sev = [a for a in anomalies if a.severity == "high"]
        assert len(high_sev) == 0

    def test_flags_category_outlier(self, normal_transactions):
        # Inject a massive coffee charge
        outlier = pd.DataFrame([{
            "date": pd.Timestamp("2026-01-15"),
            "description": "Starbucks",
            "amount": -500.00,  # ~77x normal
            "category": "Food & Dining",
        }])
        df = pd.concat([normal_transactions, outlier], ignore_index=True)
        anomalies = detect_anomalies(df)
        assert any("Food & Dining" in a.reason for a in anomalies)

    def test_flags_duplicate_charges(self):
        df = pd.DataFrame([
            {
                "date": pd.Timestamp("2026-01-10"),
                "description": "Netflix",
                "amount": -15.49,
                "category": "Entertainment",
            },
            {
                "date": pd.Timestamp("2026-01-11"),
                "description": "Netflix",
                "amount": -15.49,
                "category": "Entertainment",
            },
        ])
        anomalies = detect_anomalies(df)
        assert any("duplicate" in a.reason.lower() for a in anomalies)

    def test_no_duplicate_flag_when_far_apart(self):
        df = pd.DataFrame([
            {
                "date": pd.Timestamp("2026-01-01"),
                "description": "Netflix",
                "amount": -15.49,
                "category": "Entertainment",
            },
            {
                "date": pd.Timestamp("2026-02-10"),  # 40 days later
                "description": "Netflix",
                "amount": -15.49,
                "category": "Entertainment",
            },
        ])
        anomalies = detect_anomalies(df)
        assert not any("duplicate" in a.reason.lower() for a in anomalies)

    def test_handles_missing_category_column(self):
        df = pd.DataFrame([{
            "date": pd.Timestamp("2026-01-01"),
            "description": "X",
            "amount": -10.00,
        }])
        # Should not crash
        result = detect_anomalies(df)
        assert isinstance(result, list)

    def test_severity_values_are_valid(self, normal_transactions):
        outlier = pd.DataFrame([{
            "date": pd.Timestamp("2026-01-15"),
            "description": "Starbucks",
            "amount": -1000.00,
            "category": "Food & Dining",
        }])
        df = pd.concat([normal_transactions, outlier], ignore_index=True)
        anomalies = detect_anomalies(df)
        for a in anomalies:
            assert a.severity in {"high", "medium", "low"}
