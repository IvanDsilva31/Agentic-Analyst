"""The tools the agent can call.

Each tool is a small, explicit capability the LLM may invoke by name. The model
never touches the database directly — it can only act through these functions,
and every consequential call (``run_sql``) is checked against the policy engine
before it runs. A blocked or failed call returns a plain-text explanation that
goes back to the model as an observation, so the agent can adapt.

``build_tools`` binds the tools to a single :class:`RunContext` so they can read
the database/policies and record what they did.
"""

from __future__ import annotations

import sqlite3

import pandas as pd
from langchain_core.tools import StructuredTool

from .database import UnsafeQueryError
from .state import ChartSpec, RunContext

_CHART_TYPES = {"bar", "line", "pie", "scatter", "none"}


def build_tools(ctx: RunContext) -> list[StructuredTool]:
    """Create the toolset bound to one run context."""

    def list_tables() -> str:
        """List the names of every table in the database."""
        names = ctx.db.list_tables()
        ctx.log("tool_call", "list_tables()")
        ctx.log("tool_result", f"{len(names)} tables: {', '.join(names)}")
        return "Tables: " + ", ".join(names)

    def get_schema() -> str:
        """Return the full schema: every table, its columns and a few sample rows.
        Call this before writing SQL so you use real column names."""
        schema = ctx.db.schema_text()
        ctx.log("tool_call", "get_schema()")
        ctx.log("tool_result", schema)
        return schema

    def run_sql(sql: str) -> str:
        """Run a single read-only SQLite SELECT (or WITH ... SELECT) query and
        return the result rows as CSV. The query is validated against policy
        first; if it is rejected or errors, the reason is returned so you can fix
        it and try again."""
        ctx.log("tool_call", f"run_sql:\n{sql}")

        budget = ctx.engine.check_sql_budget(ctx.sql_attempts)
        if not budget.allowed:
            ctx.log("policy_block", budget.reason)
            return f"POLICY: {budget.reason} Use submit_answer with what you have."

        decision = ctx.engine.check_sql(sql)
        if not decision.allowed:
            ctx.log("policy_block", decision.reason)
            return f"POLICY REJECTED the query: {decision.reason}"

        ctx.sql_attempts += 1
        try:
            df = ctx.db.run_query(sql, max_rows=ctx.engine.config.row_limit)
        except (sqlite3.Error, UnsafeQueryError, pd.errors.DatabaseError) as exc:
            ctx.log("error", f"SQL error: {exc}")
            return f"ERROR running query: {exc}\nFix the query and try again."

        ctx.last_sql = sql
        ctx.last_df = df
        ctx.log("tool_result", f"{len(df)} row(s) returned.")
        preview = df.head(30).to_csv(index=False)
        truncated = "" if len(df) <= 30 else f"\n(showing first 30 of {len(df)} rows)"
        return f"{len(df)} row(s).\n{preview}{truncated}"

    def submit_answer(
        answer: str,
        chart_type: str = "none",
        x: str | None = None,
        y: str | None = None,
        title: str = "",
    ) -> str:
        """Finish the task. Provide the final plain-language answer (cite the key
        numbers) and, if the last result is chartable, a chart spec. chart_type
        is one of bar, line, pie, scatter, none. x and y must be column names that
        appear in the last query result. Call this exactly once when you are done."""
        chart = _coerce_chart(chart_type, x, y, title, ctx.last_df)
        ctx.final_answer = answer.strip()
        ctx.final_chart = chart
        ctx.finished = True
        ctx.log("answer", answer.strip())
        return "Answer recorded. Task complete."

    return [
        StructuredTool.from_function(list_tables),
        StructuredTool.from_function(get_schema),
        StructuredTool.from_function(run_sql),
        StructuredTool.from_function(submit_answer),
    ]


def _coerce_chart(
    chart_type: str,
    x: str | None,
    y: str | None,
    title: str,
    df: pd.DataFrame | None,
) -> ChartSpec:
    """Validate a model-proposed chart against the real result columns."""
    ctype = (chart_type or "none").lower()
    if ctype not in _CHART_TYPES:
        ctype = "none"
    cols = set(df.columns) if df is not None else set()
    if x not in cols:
        x = None
    if y not in cols:
        y = None
    if ctype != "none" and (x is None or y is None):
        ctype = "none"
    return ChartSpec(type=ctype, x=x, y=y, title=title or "")
