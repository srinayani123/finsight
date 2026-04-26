"""Natural-language Q&A over categorized transactions."""
import logging

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from src.agents.llm_factory import build_llm

logger = logging.getLogger(__name__)

_QA_PROMPT = """You are a personal finance assistant. You will be given:
1. A compact summary of the user's transactions (pre-computed, accurate).
2. A question from the user.

Rules:
- Answer using ONLY the data provided. If the data doesn't contain the answer, say so plainly.
- Be concise — 1-3 sentences.
- Use dollar figures with $ prefix and 2 decimals.
- Do not invent transactions or make up totals.
- If the question is ambiguous, answer the most likely interpretation and note the assumption in one short clause.
- The merchant list is comprehensive — if a merchant isn't there, it doesn't exist in the data.
"""


def _build_data_context(df: pd.DataFrame, max_chars: int = 6000) -> str:
    """Build a compact text representation of the transactions for the LLM."""
    if df.empty:
        return "No transactions."

    lines = [
        f"Total transactions: {len(df)}",
        f"Date range: {df['date'].min().date()} to {df['date'].max().date()}",
    ]

    if "category" in df.columns:
        debits = df[df["amount"] < 0]
        if not debits.empty:
            cat_totals = debits.groupby("category")["amount"].apply(
                lambda s: round(float(abs(s.sum())), 2)
            )
            lines.append("\nSpending by category:")
            for cat, total in cat_totals.sort_values(ascending=False).items():
                lines.append(f"  {cat}: ${total}")

    df_copy = df.copy()
    df_copy["month"] = df_copy["date"].dt.strftime("%Y-%m")
    lines.append("\nMonthly net cash flow:")
    for month, group in df_copy.groupby("month"):
        income = float(group[group["amount"] > 0]["amount"].sum())
        spending = float(abs(group[group["amount"] < 0]["amount"].sum()))
        lines.append(
            f"  {month}: income=${income:.2f}, spending=${spending:.2f}, "
            f"net=${income - spending:.2f}"
        )

    # Per-month per-merchant breakdown — lets the agent answer
    # "how much on X in February" type questions accurately.
    debits = df[df["amount"] < 0].copy()
    if not debits.empty:
        debits["month"] = debits["date"].dt.strftime("%Y-%m")
        merchant_stats = (
            debits.groupby("description")
            .agg(
                total=("amount", lambda s: round(float(abs(s.sum())), 2)),
                count=("amount", "count"),
            )
            .sort_values("total", ascending=False)
        )

        lines.append("\nAll merchants (total spend, visit count):")
        for merchant, row in merchant_stats.iterrows():
            lines.append(
                f"  {merchant}: ${row['total']} ({int(row['count'])} visits)"
            )

        # Per-month merchant totals — only for merchants with multiple visits,
        # to keep context size manageable.
        recurring = merchant_stats[merchant_stats["count"] > 1].index.tolist()
        if recurring:
            lines.append("\nMonthly breakdown for recurring merchants:")
            for merchant in recurring:
                merchant_rows = debits[debits["description"] == merchant]
                monthly = merchant_rows.groupby("month")["amount"].apply(
                    lambda s: round(float(abs(s.sum())), 2)
                )
                breakdown = ", ".join(f"{m}=${v}" for m, v in monthly.items())
                lines.append(f"  {merchant}: {breakdown}")

    result = "\n".join(lines)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[Context truncated]"
    return result


def _escape_dollars(text: str) -> str:
    """Escape $ signs so Streamlit doesn't interpret them as LaTeX delimiters."""
    return text.replace("$", "\\$")


def answer_question(df: pd.DataFrame, question: str) -> str:
    """Answer a natural-language question about the user's transactions."""
    if not question or not question.strip():
        return "Please ask a question."
    if df.empty:
        return "No transaction data loaded. Upload a CSV first."

    context = _build_data_context(df)
    try:
        llm = build_llm()
        response = llm.invoke(
            [
                SystemMessage(content=_QA_PROMPT),
                HumanMessage(
                    content=f"Data:\n{context}\n\nQuestion: {question.strip()}"
                ),
            ]
        )
        return _escape_dollars(response.content.strip())
    except Exception as exc:
        logger.exception("Q&A failed")
        return f"Sorry, couldn't answer that — {exc}"
    