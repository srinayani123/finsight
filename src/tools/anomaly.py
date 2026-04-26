"""Anomaly detection — statistical rules for flagging unusual transactions."""
from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass
class Anomaly:
    """A flagged transaction with a reason."""

    index: int
    date: str
    description: str
    amount: float
    reason: str
    severity: Literal["high", "medium", "low"]


def detect_anomalies(df: pd.DataFrame, z_threshold: float = 3.0) -> list[Anomaly]:
    """
    Flag anomalous transactions using multiple heuristics.

    Checks:
    - Amount outliers per category (z-score > threshold)
    - Duplicate charges (same merchant + amount within 3 days)
    - Very large single charges (top 1% of absolute amounts)

    Args:
        df: DataFrame with columns date, description, amount, category.
        z_threshold: Z-score threshold for per-category outliers.

    Returns:
        List of Anomaly objects.
    """
    if df.empty or "amount" not in df.columns:
        return []

    anomalies: list[Anomaly] = []
    flagged_indices: set[int] = set()

    # 1. Per-category z-score outliers (only for debits)
    if "category" in df.columns:
        debits = df[df["amount"] < 0].copy()
        if not debits.empty:
            debits["abs_amount"] = debits["amount"].abs()
            for category, group in debits.groupby("category"):
                if len(group) < 3:
                    continue
                mean = group["abs_amount"].mean()
                std = group["abs_amount"].std()
                if std == 0 or pd.isna(std):
                    continue
                for idx, row in group.iterrows():
                    z = (row["abs_amount"] - mean) / std
                    if z > z_threshold:
                        anomalies.append(
                            Anomaly(
                                index=int(idx),
                                date=str(row["date"])[:10],
                                description=row["description"],
                                amount=float(row["amount"]),
                                reason=f"Unusually large {category} charge (z={z:.1f})",
                                severity="high" if z > 4 else "medium",
                            )
                        )
                        flagged_indices.add(int(idx))

    # 2. Duplicate charges (same description + amount within 3 days)
    sorted_df = df.sort_values("date").reset_index(drop=True)
    for i in range(len(sorted_df) - 1):
        row = sorted_df.iloc[i]
        for j in range(i + 1, len(sorted_df)):
            other = sorted_df.iloc[j]
            delta_days = (other["date"] - row["date"]).days
            if delta_days > 3:
                break
            if (
                row["description"] == other["description"]
                and abs(row["amount"] - other["amount"]) < 0.01
                and int(other.name) not in flagged_indices  # avoid double-flagging
            ):
                anomalies.append(
                    Anomaly(
                        index=int(other.name),
                        date=str(other["date"])[:10],
                        description=other["description"],
                        amount=float(other["amount"]),
                        reason=f"Possible duplicate of transaction on {str(row['date'])[:10]}",
                        severity="medium",
                    )
                )
                flagged_indices.add(int(other.name))

    # 3. Top 1% largest charges (overall)
    debits = df[df["amount"] < 0]
    if len(debits) >= 20:
        threshold = debits["amount"].abs().quantile(0.99)
        for idx, row in debits.iterrows():
            if int(idx) in flagged_indices:
                continue
            if abs(row["amount"]) >= threshold:
                anomalies.append(
                    Anomaly(
                        index=int(idx),
                        date=str(row["date"])[:10],
                        description=row["description"],
                        amount=float(row["amount"]),
                        reason="Top 1% largest charge overall",
                        severity="low",
                    )
                )

    return anomalies
