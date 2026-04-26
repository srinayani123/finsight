"""Tests for CSV ingestion."""
from io import StringIO

import pytest

from src.tools.ingestion import ingest_csv


class TestIngestCsv:
    def test_basic_csv(self):
        csv = "date,description,amount\n2026-01-01,Coffee Shop,-5.00\n"
        df = ingest_csv(StringIO(csv))
        assert len(df) == 1
        assert df.iloc[0]["description"] == "Coffee Shop"
        assert df.iloc[0]["amount"] == -5.0

    def test_handles_alternate_column_names(self):
        csv = "Transaction Date,Merchant,Value\n2026-01-01,Starbucks,-6.50\n"
        df = ingest_csv(StringIO(csv))
        assert len(df) == 1
        assert df.iloc[0]["description"] == "Starbucks"

    def test_handles_uppercase_columns(self):
        csv = "DATE,DESCRIPTION,AMOUNT\n2026-01-01,Amazon,-25.00\n"
        df = ingest_csv(StringIO(csv))
        assert len(df) == 1

    def test_missing_columns_raises(self):
        csv = "date,description\n2026-01-01,Starbucks\n"
        with pytest.raises(ValueError, match="amount"):
            ingest_csv(StringIO(csv))

    def test_empty_csv_raises(self):
        with pytest.raises(ValueError):
            ingest_csv(StringIO("date,description,amount\n"))

    def test_strips_whitespace(self):
        csv = "date,description,amount\n2026-01-01,  Coffee  ,-5.00\n"
        df = ingest_csv(StringIO(csv))
        assert df.iloc[0]["description"] == "Coffee"

    def test_skips_invalid_rows(self):
        csv = (
            "date,description,amount\n"
            "2026-01-01,Coffee,-5.00\n"
            "invalid-date,Bad Row,-10.00\n"
            "2026-01-02,Tea,-3.50\n"
        )
        df = ingest_csv(StringIO(csv))
        assert len(df) == 2
        assert "Bad Row" not in df["description"].values

    def test_accepts_bytes_input(self):
        csv_bytes = b"date,description,amount\n2026-01-01,Coffee,-5.00\n"
        df = ingest_csv(csv_bytes)
        assert len(df) == 1

    def test_parses_positive_amounts(self):
        csv = "date,description,amount\n2026-01-01,Paycheck,2400.00\n"
        df = ingest_csv(StringIO(csv))
        assert df.iloc[0]["amount"] == 2400.0

    def test_converts_dates(self):
        csv = "date,description,amount\n2026-01-15,Coffee,-5.00\n"
        df = ingest_csv(StringIO(csv))
        assert df.iloc[0]["date"].year == 2026
        assert df.iloc[0]["date"].month == 1
        assert df.iloc[0]["date"].day == 15
