"""Shared data types: the agent trace, the chart spec, the final result, and the
mutable run context that tools and the harness share during one question.

These live in their own module so ``tools.py``, ``graph.py`` and ``agent.py`` can
all import them without a circular dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

from .database import Database
from .policies import PolicyEngine

StepKind = Literal[
    "think", "tool_call", "tool_result", "policy_block", "result", "answer", "error"
]


@dataclass
class Step:
    """One entry in the agent trace shown in the UI."""

    kind: StepKind
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


@dataclass
class RunContext:
    """Mutable state for a single question.

    Tools read the database and policy engine from here and record their effects
    (the last query + DataFrame, the final answer) so the harness can assemble an
    :class:`AgentResult` after the graph finishes. It also holds the counters the
    run policies are enforced against.
    """

    db: Database
    engine: PolicyEngine
    steps: list[Step] = field(default_factory=list)

    turns: int = 0          # agent (think) turns taken, for the step budget
    sql_attempts: int = 0
    last_sql: str | None = None
    last_df: pd.DataFrame | None = None

    # Set by the terminal submit_answer tool.
    final_answer: str | None = None
    final_chart: ChartSpec = field(default_factory=ChartSpec)
    finished: bool = False

    def log(self, kind: StepKind, detail: str) -> None:
        self.steps.append(Step(kind, detail))
