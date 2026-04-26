"""CSV ingestion — parse user transaction files with flexible column naming."""
from io import StringIO
from typing import Union

import pandas as pd

# Column aliases — user CSVs may use different casing/names
_DATE_ALIASES = {"date", "transaction date", "posted date", "trans date"}
_DESC_ALIASES = {"description", "merchant", "payee", "memo", "details"}
_AMOUNT_ALIASES = {"amount", "transaction amount", "value", "debit", "credit"}


def _find_column(columns: list[str], aliases: set[str]) -> str | None:
    """Find the first column matching an alias (case-insensitive)."""
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def ingest_csv(source: Union[str, StringIO, bytes]) -> pd.DataFrame:
    """
    Parse a transactions CSV into a normalized DataFrame.

    Required columns: date, description, amount
    Returns DataFrame with columns: date (datetime), description (str), amount (float)

    Raises:
        ValueError: if required columns can't be found or data is invalid.
    """
    if isinstance(source, bytes):
        source = StringIO(source.decode("utf-8", errors="replace"))

    try:
        df = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV is empty")

    date_col = _find_column(list(df.columns), _DATE_ALIASES)
    desc_col = _find_column(list(df.columns), _DESC_ALIASES)
    amount_col = _find_column(list(df.columns), _AMOUNT_ALIASES)

    missing = []
    if not date_col: missing.append("date")
    if not desc_col: missing.append("description")
    if not amount_col: missing.append("amount")
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            f"Got: {list(df.columns)}"
        )

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col], errors="coerce"),
        "description": df[desc_col].astype(str).str.strip(),
        "amount": pd.to_numeric(df[amount_col], errors="coerce"),
    })

    # Drop rows with invalid date/amount
    before = len(out)
    out = out.dropna(subset=["date", "amount"]).reset_index(drop=True)
    if len(out) == 0:
        raise ValueError(f"No valid rows after parsing ({before} total had invalid data)")

    return out
