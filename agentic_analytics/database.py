"""SQLite access: schema introspection and read-only query execution.

The agent only ever runs SELECT queries. We enforce that defensively here so a
hallucinated UPDATE/DROP can never touch the data.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

import pandas as pd

# Statements the agent is never allowed to run.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma)\b",
    re.IGNORECASE,
)


class UnsafeQueryError(ValueError):
    """Raised when a generated query is not a plain read-only SELECT."""


@dataclass
class Column:
    name: str
    type: str
    pk: bool


@dataclass
class Table:
    name: str
    columns: list[Column]
    sample_rows: list[dict]


class Database:
    """Thin wrapper over a SQLite file with introspection helpers."""

    def __init__(self, path: str):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        # Open read-only so nothing can mutate the database, even by accident.
        conn = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def list_tables(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' ORDER BY name"
            ).fetchall()
        return [r["name"] for r in rows]

    def describe(self, sample_rows: int = 3) -> list[Table]:
        """Return every table with its columns and a few sample rows."""
        tables: list[Table] = []
        with self._connect() as conn:
            for name in self.list_tables():
                cols = [
                    Column(name=c["name"], type=c["type"] or "", pk=bool(c["pk"]))
                    for c in conn.execute(f'PRAGMA table_info("{name}")').fetchall()
                ]
                samples = [
                    dict(r)
                    for r in conn.execute(
                        f'SELECT * FROM "{name}" LIMIT {int(sample_rows)}'
                    ).fetchall()
                ]
                tables.append(Table(name=name, columns=cols, sample_rows=samples))
        return tables

    def schema_text(self) -> str:
        """Render the schema as compact text for the LLM prompt."""
        lines: list[str] = []
        for t in self.describe():
            col_defs = ", ".join(
                f"{c.name} {c.type}".strip() + (" PK" if c.pk else "") for c in t.columns
            )
            lines.append(f"TABLE {t.name} ({col_defs})")
            if t.sample_rows:
                lines.append(f"  sample: {t.sample_rows}")
        return "\n".join(lines)

    @staticmethod
    def assert_select_only(sql: str) -> None:
        stripped = sql.strip().rstrip(";").strip()
        if not stripped:
            raise UnsafeQueryError("Empty query.")
        # Only a single statement, and it must start with SELECT or WITH.
        if ";" in stripped:
            raise UnsafeQueryError("Multiple statements are not allowed.")
        if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
            raise UnsafeQueryError("Only SELECT queries are allowed.")
        if _FORBIDDEN.search(stripped):
            raise UnsafeQueryError("Query contains a forbidden keyword.")

    def run_query(self, sql: str, max_rows: int = 5000) -> pd.DataFrame:
        """Validate and execute a read-only query, returning a DataFrame.

        Raises sqlite3.Error on a bad query so the agent can read the message
        and try to fix itself.
        """
        self.assert_select_only(sql)
        with self._connect() as conn:
            df = pd.read_sql_query(sql, conn)
        if len(df) > max_rows:
            df = df.head(max_rows)
        return df
