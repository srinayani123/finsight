"""Transaction categorizer — rule-based with LLM fallback for unknowns."""
import logging
from typing import Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Keyword-based category rules (fast, deterministic, no LLM cost)
_CATEGORY_RULES: dict[str, list[str]] = {
    "Food & Dining": [
        "starbucks", "mcdonald", "chipotle", "doordash", "uber eats", "grubhub",
        "restaurant", "cafe", "coffee", "pizza", "sushi", "deli", "bakery",
    ],
    "Groceries": [
        "whole foods", "trader joe", "safeway", "kroger", "costco", "walmart",
        "grocery", "supermarket", "aldi", "publix", "wegmans",
    ],
    "Transportation": [
        "uber", "lyft", "shell", "chevron", "exxon", "gas", "fuel", "parking",
        "mta", "bart", "caltrain", "transit", "toll",
    ],
    "Shopping": [
        "amazon", "target", "best buy", "ebay", "etsy", "nordstrom", "macy",
        "apple store", "nike", "zara", "h&m",
    ],
    "Entertainment": [
        "netflix", "spotify", "hulu", "disney", "youtube", "movie", "cinema",
        "theater", "concert", "steam", "playstation",
    ],
    "Utilities": [
        "electric", "water", "gas company", "pg&e", "internet", "comcast",
        "verizon", "att", "t-mobile", "utility", "wifi",
    ],
    "Healthcare": [
        "cvs", "walgreens", "pharmacy", "doctor", "clinic", "hospital",
        "dental", "medical", "health",
    ],
    "Income": [
        "payroll", "paycheck", "direct deposit", "salary", "deposit from",
        "refund", "reimbursement",
    ],
    "Housing": [
        "rent", "mortgage", "landlord", "hoa",
    ],
    "Fitness": [
        "gym", "yoga", "peloton", "fitness", "crossfit",
    ],
    "Transfers": [
        "venmo", "zelle", "paypal", "cash app", "transfer to",
    ],
}


def categorize_by_rules(description: str) -> Optional[str]:
    """Match a transaction description against keyword rules. Returns None if no match."""
    if not description:
        return None
    desc_lower = description.lower()
    for category, keywords in _CATEGORY_RULES.items():
        if any(kw in desc_lower for kw in keywords):
            return category
    return None


class Categorizer:
    """Categorizes transactions using rules first, then an optional LLM callable for unknowns."""

    def __init__(
        self,
        llm_classifier: Optional[Callable[[str], str]] = None,
        default_category: str = "Other",
    ) -> None:
        self.llm_classifier = llm_classifier
        self.default_category = default_category

    def categorize_one(self, description: str) -> str:
        """Categorize a single transaction description."""
        rule_match = categorize_by_rules(description)
        if rule_match:
            return rule_match
        if self.llm_classifier:
            try:
                result = self.llm_classifier(description)
                if result and isinstance(result, str):
                    return result.strip()
            except Exception as exc:
                logger.warning("LLM classification failed: %s", exc)
        return self.default_category

    def categorize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add a 'category' column to a transactions DataFrame."""
        out = df.copy()
        out["category"] = out["description"].apply(self.categorize_one)
        return out
