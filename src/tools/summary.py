"""Monthly summary aggregation for categorized transactions."""
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class MonthlySummary:
    """Aggregated monthly financial summary."""

    month: str  # YYYY-MM
    total_income: float
    total_spending: float
    net: float
    savings_rate: float  # 0.0 to 1.0
    spending_by_category: dict[str, float] = field(default_factory=dict)
    transaction_count: int = 0
    top_merchants: list[tuple[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "month": self.month,
            "total_income": round(self.total_income, 2),
            "total_spending": round(self.total_spending, 2),
            "net": round(self.net, 2),
            "savings_rate": round(self.savings_rate, 3),
            "spending_by_category": {
                k: round(v, 2) for k, v in self.spending_by_category.items()
            },
            "transaction_count": self.transaction_count,
            "top_merchants": [(m, round(v, 2)) for m, v in self.top_merchants],
        }


def monthly_summary(df: pd.DataFrame, month: str | None = None) -> list[MonthlySummary]:
    """
    Aggregate transactions into per-month summaries.

    Args:
        df: DataFrame with columns date, description, amount, category.
        month: Optional YYYY-MM filter. If None, returns all months.

    Returns:
        List of MonthlySummary objects, one per month (sorted by month).
    """
    if df.empty:
        return []

    work = df.copy()
    work["month"] = work["date"].dt.strftime("%Y-%m")

    if month:
        work = work[work["month"] == month]
        if work.empty:
            return []

    summaries: list[MonthlySummary] = []
    for m, group in work.groupby("month"):
        income = float(group[group["amount"] > 0]["amount"].sum())
        spending = float(abs(group[group["amount"] < 0]["amount"].sum()))
        net = income - spending
        savings_rate = (net / income) if income > 0 else 0.0

        by_cat: dict[str, float] = {}
        if "category" in group.columns:
            debits = group[group["amount"] < 0]
            for cat, cat_group in debits.groupby("category"):
                by_cat[str(cat)] = float(abs(cat_group["amount"].sum()))

        # Top 5 merchants by spend
        debits = group[group["amount"] < 0]
        merchant_totals = (
            debits.groupby("description")["amount"]
            .apply(lambda s: float(abs(s.sum())))
            .sort_values(ascending=False)
            .head(5)
        )
        top_merchants = [(str(k), float(v)) for k, v in merchant_totals.items()]

        summaries.append(
            MonthlySummary(
                month=str(m),
                total_income=income,
                total_spending=spending,
                net=net,
                savings_rate=savings_rate,
                spending_by_category=by_cat,
                transaction_count=len(group),
                top_merchants=top_merchants,
            )
        )

    return sorted(summaries, key=lambda s: s.month)
