"""Streamlit UI for the Agentic Analytics Assistant.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import os

import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from agentic_analytics.agent import AnalyticsAgent, ChartSpec
from agentic_analytics.database import Database

load_dotenv()

DB_PATH = os.environ.get("DATABASE_PATH", "data/sample.db")

st.set_page_config(page_title="Agentic Analytics Assistant", page_icon="📊", layout="wide")

EXAMPLES = [
    "What were the top 5 best-selling products by revenue?",
    "How many orders were placed each month?",
    "Which countries have the most customers?",
    "What is the total revenue from completed orders by category?",
    "What share of orders ended up cancelled or refunded?",
]


def render_chart(spec: ChartSpec, df) -> None:
    if spec.type == "none" or spec.x is None or spec.y is None:
        return
    try:
        if spec.type == "bar":
            fig = px.bar(df, x=spec.x, y=spec.y, title=spec.title)
        elif spec.type == "line":
            fig = px.line(df, x=spec.x, y=spec.y, markers=True, title=spec.title)
        elif spec.type == "pie":
            fig = px.pie(df, names=spec.x, values=spec.y, title=spec.title)
        elif spec.type == "scatter":
            fig = px.scatter(df, x=spec.x, y=spec.y, title=spec.title)
        else:
            return
        st.plotly_chart(fig, use_container_width=True)
    except Exception as exc:  # noqa: BLE001 - charting is best-effort
        st.info(f"(Couldn't render the suggested chart: {exc})")


def render_trace(steps) -> None:
    icons = {
        "think": "🧠 Reasoned",
        "tool_call": "🛠️ Called a tool",
        "tool_result": "📥 Tool result",
        "policy_block": "🛡️ Policy blocked",
        "error": "⚠️ Error",
        "result": "✅ Finished",
        "answer": "💬 Composed answer",
    }
    for step in steps:
        label = icons.get(step.kind, step.kind)
        detail = step.detail
        # Render run_sql tool calls as a SQL block for readability.
        if step.kind == "tool_call" and detail.startswith("run_sql:"):
            st.markdown(f"**{label}** — `run_sql`")
            st.code(detail.split(":", 1)[1].strip(), language="sql")
        elif step.kind == "tool_result" and "TABLE " in detail:
            st.markdown(f"**{label}** — schema")
            st.code(detail, language="text")
        else:
            st.markdown(f"**{label}** — {detail}")


def main() -> None:
    st.title("📊 Agentic Analytics Assistant")
    st.caption(
        "Ask a question about your data in plain English. An autonomous agent "
        "inspects the schema, writes SQL, runs it, fixes its own errors, and "
        "answers with a chart."
    )

    if not os.path.exists(DB_PATH):
        st.error(
            f"Database not found at `{DB_PATH}`. Run `python seed_data.py` first "
            "to create the sample database."
        )
        st.stop()

    db = Database(DB_PATH)

    with st.sidebar:
        st.header("Setup")
        has_env_key = bool(os.environ.get("GEMINI_API_KEY"))
        key_input = st.text_input(
            "Gemini API key",
            type="password",
            help="Leave blank to use GEMINI_API_KEY from your .env / environment.",
            placeholder="set in .env" if has_env_key else "paste your free key",
        )
        st.markdown(
            "Get a free key at "
            "[aistudio.google.com](https://aistudio.google.com/app/apikey)."
        )
        st.divider()
        st.subheader("Tables")
        for t in db.describe():
            st.markdown(f"**{t.name}** — " + ", ".join(c.name for c in t.columns))

    st.subheader("Try an example")
    cols = st.columns(len(EXAMPLES))
    for col, ex in zip(cols, EXAMPLES):
        if col.button(ex, use_container_width=True):
            st.session_state["question"] = ex

    question = st.text_input(
        "Your question",
        value=st.session_state.get("question", ""),
        placeholder="e.g. Which signup channel brought the most revenue?",
    )

    if st.button("Ask", type="primary") and question.strip():
        api_key = key_input.strip() or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            st.error(
                "No Gemini API key. Paste one in the sidebar or set GEMINI_API_KEY "
                "in a .env file."
            )
            st.stop()

        try:
            agent = AnalyticsAgent(db=db, api_key=api_key)
        except Exception as exc:  # noqa: BLE001
            st.error(str(exc))
            st.stop()

        with st.spinner("The agent is working..."):
            result = agent.ask(question.strip())

        if result.success:
            st.success(result.answer)
        else:
            st.warning(result.answer)

        if result.chart.type != "none" and result.data is not None:
            render_chart(result.chart, result.data)

        if result.data is not None:
            with st.expander(f"Data ({len(result.data)} rows)", expanded=False):
                st.dataframe(result.data, use_container_width=True)

        if result.sql:
            with st.expander("SQL the agent ran", expanded=False):
                st.code(result.sql, language="sql")

        with st.expander("Agent trace (what it did step by step)", expanded=False):
            render_trace(result.steps)


if __name__ == "__main__":
    main()
