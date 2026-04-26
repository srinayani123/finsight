"""Streamlit UI — FinSight personal finance agent."""
import io
import logging

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from src.agents import answer_question, build_finance_graph
from src.tools import ingest_csv

load_dotenv()
logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="FinSight",
    page_icon="💸",
    layout="wide",
)


def _init_state() -> None:
    """Initialize session state keys."""
    defaults = {
        "categorized_df": None,
        "anomalies": [],
        "summaries": [],
        "qa_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _run_pipeline(df: pd.DataFrame, use_llm: bool) -> None:
    """Run the LangGraph finance pipeline and store results in session state."""
    graph = build_finance_graph()
    with st.spinner("Categorizing, detecting anomalies, summarizing..."):
        final = graph.invoke({"raw_df": df, "use_llm_categorizer": use_llm})
    st.session_state.categorized_df = final["categorized_df"]
    st.session_state.anomalies = final["anomalies"]
    st.session_state.summaries = final["summaries"]


def _sidebar() -> tuple[bool, bool]:
    """Render the sidebar. Returns (use_llm_categorizer, use_sample)."""
    st.sidebar.header("Settings")
    use_llm = st.sidebar.checkbox(
        "Use LLM for unknown categories",
        value=False,
        help="Falls back to LLM when keyword rules don't match. Slower, costs API calls.",
    )
    use_sample = st.sidebar.button("Load sample data", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption(
        "Upload a CSV with columns `date`, `description`, `amount`. "
        "Negative amounts are debits, positive are credits."
    )
    return use_llm, use_sample


def _render_upload(use_llm: bool, use_sample: bool) -> None:
    """Render the upload widget and trigger pipeline on new data."""
    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if use_sample:
        sample_path = "sample_data/sample_transactions.csv"
        try:
            df = ingest_csv(sample_path)
            _run_pipeline(df, use_llm)
            st.success(f"Loaded sample data: {len(df)} transactions.")
        except Exception as exc:
            st.error(f"Couldn't load sample: {exc}")
        return

    if uploaded is not None:
        try:
            content = uploaded.read()
            df = ingest_csv(io.BytesIO(content))
            _run_pipeline(df, use_llm)
            st.success(f"Loaded {len(df)} transactions.")
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Couldn't process CSV: {exc}")


def _render_summary() -> None:
    """Render the monthly summary and spending chart."""
    summaries = st.session_state.summaries
    if not summaries:
        return

    st.subheader("Monthly summary")

    latest = summaries[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Month", latest.month)
    col2.metric("Income", f"${latest.total_income:,.2f}")
    col3.metric("Spending", f"${latest.total_spending:,.2f}")
    col4.metric("Savings rate", f"{latest.savings_rate:.1%}")

    if latest.spending_by_category:
        cat_df = pd.DataFrame(
            list(latest.spending_by_category.items()),
            columns=["category", "amount"],
        ).sort_values("amount", ascending=False)
        fig = px.pie(
            cat_df,
            values="amount",
            names="category",
            title=f"Spending by category — {latest.month}",
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    if len(summaries) > 1:
        trend_df = pd.DataFrame([
            {
                "month": s.month,
                "income": s.total_income,
                "spending": s.total_spending,
                "net": s.net,
            }
            for s in summaries
        ])
        fig = px.bar(
            trend_df,
            x="month",
            y=["income", "spending"],
            barmode="group",
            title="Monthly income vs spending",
        )
        st.plotly_chart(fig, use_container_width=True)


def _render_anomalies() -> None:
    """Render the anomalies table."""
    anomalies = st.session_state.anomalies
    st.subheader(f"Anomalies ({len(anomalies)})")

    if not anomalies:
        st.info("No anomalies detected.")
        return

    rows = [
        {
            "Date": a.date,
            "Description": a.description,
            "Amount": f"${a.amount:,.2f}",
            "Severity": a.severity,
            "Reason": a.reason,
        }
        for a in anomalies
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_transactions() -> None:
    """Render the categorized transactions table."""
    df = st.session_state.categorized_df
    if df is None or df.empty:
        return

    st.subheader(f"Transactions ({len(df)})")

    col1, col2 = st.columns(2)
    with col1:
        categories = sorted(df["category"].unique())
        selected_cats = st.multiselect(
            "Filter by category", categories, default=categories
        )
    with col2:
        search = st.text_input("Search description", "")

    filtered = df[df["category"].isin(selected_cats)]
    if search:
        filtered = filtered[
            filtered["description"].str.contains(search, case=False, na=False)
        ]

    display = filtered.copy()
    display["date"] = display["date"].dt.strftime("%Y-%m-%d")
    display["amount"] = display["amount"].apply(lambda x: f"${x:,.2f}")
    st.dataframe(display, use_container_width=True, hide_index=True)


def _render_qa() -> None:
    """Render the natural-language Q&A box."""
    df = st.session_state.categorized_df
    if df is None or df.empty:
        return

    st.subheader("Ask a question")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input(
            "e.g. 'how much did I spend on coffee last month?'",
            key="qa_input",
        )
        submitted = st.form_submit_button("Ask")

    if submitted and question:
        with st.spinner("Thinking..."):
            answer = answer_question(df, question)
        st.session_state.qa_history.insert(0, {"q": question, "a": answer})

    for entry in st.session_state.qa_history[:5]:
        with st.chat_message("user"):
            st.write(entry["q"])
        with st.chat_message("assistant"):
            st.write(entry["a"])


def main() -> None:
    _init_state()
    st.title("💸 FinSight")
    st.caption("Personal finance agent — categorize, detect anomalies, and ask questions about your transactions.")

    use_llm, use_sample = _sidebar()
    _render_upload(use_llm, use_sample)

    if st.session_state.categorized_df is not None:
        _render_summary()
        _render_anomalies()
        _render_qa()
        _render_transactions()


if __name__ == "__main__":
    main()
