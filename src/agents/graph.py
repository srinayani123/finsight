"""LangGraph pipeline: ingest -> categorize -> detect anomalies -> summarize."""
import logging
from typing import Any, TypedDict

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from src.agents.llm_factory import build_llm
from src.tools import Categorizer, detect_anomalies, monthly_summary
from src.tools.anomaly import Anomaly
from src.tools.summary import MonthlySummary

logger = logging.getLogger(__name__)


class FinanceState(TypedDict, total=False):
    """State shared across finance pipeline nodes."""

    raw_df: pd.DataFrame
    categorized_df: pd.DataFrame
    anomalies: list[Anomaly]
    summaries: list[MonthlySummary]
    use_llm_categorizer: bool
    error: str


_CATEGORIZE_PROMPT = """You are a transaction categorizer. Given a merchant or description string, \
return EXACTLY ONE category from this list:
Food & Dining, Groceries, Transportation, Shopping, Entertainment, Utilities, \
Healthcare, Income, Housing, Fitness, Transfers, Other

Respond with only the category name. No explanation, no punctuation."""


def _llm_classifier_factory():
    """Build a callable that classifies descriptions using the LLM."""
    llm = build_llm()

    def classify(description: str) -> str:
        try:
            response = llm.invoke(
                [
                    SystemMessage(content=_CATEGORIZE_PROMPT),
                    HumanMessage(content=description),
                ]
            )
            return response.content.strip().split("\n")[0].strip()
        except Exception as exc:
            logger.warning("LLM classification failed for '%s': %s", description, exc)
            return "Other"

    return classify


def categorize_node(state: FinanceState) -> dict[str, Any]:
    """Categorize all transactions — rules first, optional LLM for unknowns."""
    df = state["raw_df"]
    llm_fn = _llm_classifier_factory() if state.get("use_llm_categorizer") else None
    categorizer = Categorizer(llm_classifier=llm_fn)
    categorized = categorizer.categorize_df(df)
    return {"categorized_df": categorized}


def anomaly_node(state: FinanceState) -> dict[str, Any]:
    """Run anomaly detection on categorized transactions."""
    df = state["categorized_df"]
    anomalies = detect_anomalies(df)
    return {"anomalies": anomalies}


def summary_node(state: FinanceState) -> dict[str, Any]:
    """Generate monthly summaries."""
    df = state["categorized_df"]
    summaries = monthly_summary(df)
    return {"summaries": summaries}


def build_finance_graph():
    """Build the finance processing pipeline.

    Flow: categorize -> anomaly -> summary -> END
    """
    graph = StateGraph(FinanceState)
    graph.add_node("categorize", categorize_node)
    graph.add_node("anomaly", anomaly_node)
    graph.add_node("summary", summary_node)
    graph.set_entry_point("categorize")
    graph.add_edge("categorize", "anomaly")
    graph.add_edge("anomaly", "summary")
    graph.add_edge("summary", END)
    return graph.compile()
