"""The autonomous analytics agent.

Given a plain-English question it:
  1. inspects the database schema,
  2. writes a SQL query,
  3. runs it,
  4. if the query errors, feeds the error back to the model and retries (self-fix),
  5. summarizes the result in plain language and proposes a chart.

Every step is recorded in a trace so the UI can show the agent's reasoning.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .database import Database, UnsafeQueryError
from .gemini_client import GeminiClient

MAX_FIX_ATTEMPTS = 3


@dataclass
class Step:
    kind: Literal["schema", "sql", "error", "fix", "result", "answer"]
    detail: str


@dataclass
class ChartSpec:
    type: Literal["bar", "line", "pie", "scatter", "none"] = "none"
    x: str | None = None
    y: str | None = None
    title: str = ""


@dataclass
class AgentResult:
    question: str
    answer: str
    sql: str | None
    data: pd.DataFrame | None
    chart: ChartSpec
    steps: list[Step] = field(default_factory=list)
    success: bool = True


_SQL_PROMPT = """You are a SQL analyst working with a SQLite database.

Database schema:
{schema}

Write a single read-only SQLite SELECT query that answers this question:
"{question}"

Rules:
- Output ONLY the SQL query, no explanation, no markdown fences.
- Use only SELECT (or WITH ... SELECT). Never modify data.
- Prefer readable column aliases (e.g. AS total_revenue).
- When aggregating money, remember order_items has quantity and unit_price.
- Limit large result sets sensibly (e.g. top 20) unless the question needs all rows.
"""

_FIX_PROMPT = """The previous SQLite query failed.

Database schema:
{schema}

Question: "{question}"

Failed query:
{sql}

SQLite error:
{error}

Write a corrected single read-only SELECT query. Output ONLY the SQL, no markdown, no explanation.
"""

_ANSWER_PROMPT = """A user asked: "{question}"

This SQL was run:
{sql}

It returned {n_rows} row(s). Here is the result (first rows):
{preview}

Respond with a JSON object with these keys:
- "answer": a concise, plain-language answer to the question (2-4 sentences). Cite the key numbers.
- "chart": an object describing the best chart for this result, with keys:
    - "type": one of "bar", "line", "pie", "scatter", or "none" (use "none" for a single value or non-chartable result)
    - "x": the column name for the x-axis (or category), or null
    - "y": the column name for the y-axis (the measure), or null
    - "title": a short chart title

Pick "line" for time series, "bar" for category comparisons, "pie" for parts of a whole.
Only use column names that actually appear in the result.
"""


def _clean_sql(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # strip a leading ```sql / ``` and trailing ```
        text = text.split("```", 2)[1] if text.count("```") >= 2 else text.strip("`")
        if text.lower().startswith("sql"):
            text = text[3:]
    return text.strip().strip("`").strip()


class AnalyticsAgent:
    def __init__(self, db: Database, llm: GeminiClient):
        self.db = db
        self.llm = llm

    def ask(self, question: str) -> AgentResult:
        steps: list[Step] = []

        # 1. Inspect schema.
        schema = self.db.schema_text()
        steps.append(Step("schema", schema))

        # 2. Generate the first query.
        sql = _clean_sql(
            self.llm.generate(_SQL_PROMPT.format(schema=schema, question=question))
        )
        steps.append(Step("sql", sql))

        # 3 + 4. Run, and self-fix on error.
        df: pd.DataFrame | None = None
        last_error: str | None = None
        for attempt in range(MAX_FIX_ATTEMPTS + 1):
            try:
                df = self.db.run_query(sql)
                last_error = None
                break
            except (sqlite3.Error, UnsafeQueryError, pd.errors.DatabaseError) as exc:
                last_error = str(exc)
                steps.append(Step("error", f"Attempt {attempt + 1}: {last_error}"))
                if attempt == MAX_FIX_ATTEMPTS:
                    break
                sql = _clean_sql(
                    self.llm.generate(
                        _FIX_PROMPT.format(
                            schema=schema, question=question, sql=sql, error=last_error
                        )
                    )
                )
                steps.append(Step("fix", sql))

        if df is None:
            return AgentResult(
                question=question,
                answer=(
                    "I couldn't write a working query for that after several "
                    f"attempts. Last error: {last_error}"
                ),
                sql=sql,
                data=None,
                chart=ChartSpec(),
                steps=steps,
                success=False,
            )

        steps.append(Step("result", f"{len(df)} row(s) returned."))

        # 5. Summarize + propose a chart.
        preview = df.head(30).to_csv(index=False)
        parsed = self.llm.generate_json(
            _ANSWER_PROMPT.format(
                question=question, sql=sql, n_rows=len(df), preview=preview
            )
        )
        answer = str(parsed.get("answer", "")).strip() or "Here are the results."
        chart = _parse_chart(parsed.get("chart"), df)
        steps.append(Step("answer", answer))

        return AgentResult(
            question=question,
            answer=answer,
            sql=sql,
            data=df,
            chart=chart,
            steps=steps,
            success=True,
        )


def _parse_chart(spec: object, df: pd.DataFrame) -> ChartSpec:
    if not isinstance(spec, dict):
        return ChartSpec()
    ctype = str(spec.get("type", "none")).lower()
    if ctype not in {"bar", "line", "pie", "scatter", "none"}:
        ctype = "none"
    x = spec.get("x")
    y = spec.get("y")
    cols = set(df.columns)
    # Drop axes the model invented that aren't in the result.
    if x not in cols:
        x = None
    if y not in cols:
        y = None
    if ctype != "none" and (x is None or y is None):
        ctype = "none"
    return ChartSpec(type=ctype, x=x, y=y, title=str(spec.get("title", "")))
