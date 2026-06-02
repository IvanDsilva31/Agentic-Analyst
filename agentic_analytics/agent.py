"""The agent facade.

This ties the three pieces together for the UI:

  * a **harness** (the LangGraph loop in :mod:`graph`),
  * **tools** the model calls (:mod:`tools`),
  * **policies** that gate those calls (:mod:`policies`).

``AnalyticsAgent.ask`` runs one question through the graph and returns an
:class:`AgentResult` — the same shape the Streamlit UI has always consumed, so
nothing downstream had to change.

The trace types (:class:`Step`, :class:`ChartSpec`, :class:`AgentResult`) are
re-exported here for backwards compatibility; they now live in :mod:`state`.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from .database import Database
from .graph import build_graph, system_message
from .policies import PolicyConfig, PolicyEngine
from .state import AgentResult, ChartSpec, RunContext, Step

__all__ = ["AnalyticsAgent", "AgentResult", "ChartSpec", "Step"]


class AnalyticsAgent:
    def __init__(
        self,
        db: Database,
        *,
        api_key: str | None = None,
        model: str | None = None,
        policy: PolicyConfig | None = None,
    ):
        self.db = db
        self.api_key = api_key
        self.model = model
        self.engine = PolicyEngine(config=policy or PolicyConfig())

    def ask(self, question: str) -> AgentResult:
        ctx = RunContext(db=self.db, engine=self.engine)
        app = build_graph(ctx, api_key=self.api_key, model=self.model)

        messages = [system_message(), HumanMessage(content=question)]
        # recursion_limit bounds graph hops; our own max_steps policy stops sooner,
        # but give LangGraph headroom (each turn is ~2 hops: agent -> tools).
        recursion_limit = self.engine.config.max_steps * 2 + 5

        try:
            app.invoke({"messages": messages}, config={"recursion_limit": recursion_limit})
        except Exception as exc:  # noqa: BLE001 - surface any harness failure to the user
            ctx.log("error", f"Harness error: {exc}")
            return AgentResult(
                question=question,
                answer=f"The agent hit an error: {exc}",
                sql=ctx.last_sql,
                data=ctx.last_df,
                chart=ChartSpec(),
                steps=ctx.steps,
                success=False,
            )

        if ctx.finished and ctx.final_answer:
            ctx.log("result", f"Done in {ctx.turns} step(s).")
            return AgentResult(
                question=question,
                answer=ctx.final_answer,
                sql=ctx.last_sql,
                data=ctx.last_df,
                chart=ctx.final_chart,
                steps=ctx.steps,
                success=True,
            )

        # The agent stopped without calling submit_answer (ran out of budget or
        # replied in plain text). Return whatever we managed to gather.
        fallback = ctx.final_answer or (
            "I couldn't fully answer that within my step/query budget. "
            "Here's what I gathered."
        )
        return AgentResult(
            question=question,
            answer=fallback,
            sql=ctx.last_sql,
            data=ctx.last_df,
            chart=ctx.final_chart,
            steps=ctx.steps,
            success=ctx.last_df is not None,
        )
