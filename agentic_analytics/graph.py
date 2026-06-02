"""The harness: a LangGraph agent loop.

This is the engine that turns a question into an answer. Unlike a hardcoded
pipeline, the *model* decides what to do at each step — inspect the schema, run a
query, fix a failed query, or finish — and the graph just keeps looping until the
agent calls ``submit_answer`` or hits a policy limit.

    ┌──────────┐  tool_calls?   ┌──────────┐
    │  agent   ├───── yes ─────▶│  tools   │
    │ (think)  │◀───────────────┤ (act +   │
    └────┬─────┘                │  policy) │
         │ done / no tool       └──────────┘
         ▼
       (END)

The agent node is the LLM; the tools node executes the chosen tool through the
policy gate. Every step is recorded in the run context's trace.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from .llm import build_llm
from .state import RunContext
from .tools import build_tools

_SYSTEM_PROMPT = """You are an autonomous data analyst answering questions about a \
SQLite database. You work by calling tools — you cannot see the data otherwise.

Your workflow:
1. Call get_schema to learn the real tables and columns (do this before writing SQL).
2. Write a single read-only SELECT (or WITH ... SELECT) and run it with run_sql.
3. If run_sql returns an ERROR or a POLICY rejection, read the message, fix your
   query, and try again.
4. When you have the data you need, call submit_answer with a concise answer that
   cites the key numbers, plus a chart spec if the result is chartable.

Policies you must respect (enforced for you):
- Read-only: only SELECT / WITH queries. No INSERT/UPDATE/DELETE/DDL.
- You have a limited number of query attempts and total steps. Don't waste them —
  if you're running low, call submit_answer with the best answer you have.

Notes about the data: money lives in order_items as quantity * unit_price.
Prefer readable column aliases (e.g. AS total_revenue). Limit large result sets
sensibly (e.g. top 20) unless the question needs every row."""


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(ctx: RunContext, *, api_key: str | None = None, model: str | None = None):
    """Compile a LangGraph agent bound to ``ctx`` and its policies."""
    llm = build_llm(api_key=api_key, model=model)
    tools = build_tools(ctx)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: _State) -> _State:
        ctx.turns += 1
        ai: AIMessage = llm_with_tools.invoke(state["messages"])
        if ai.content:
            text = ai.content if isinstance(ai.content, str) else str(ai.content)
            ctx.log("think", text)
        return {"messages": [ai]}

    def tools_node(state: _State) -> _State:
        last: AIMessage = state["messages"][-1]
        outputs: list[ToolMessage] = []
        for call in last.tool_calls:
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"Unknown tool: {call['name']}"
            else:
                content = tool.invoke(call["args"])
            outputs.append(
                ToolMessage(content=str(content), tool_call_id=call["id"], name=call["name"])
            )
        return {"messages": outputs}

    def route(state: _State) -> str:
        if ctx.finished:
            return END
        if not ctx.engine.check_step_budget(ctx.turns).allowed:
            ctx.log("policy_block", "Step budget reached; ending run.")
            return END
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return END  # model answered in plain text without calling a tool

    graph = StateGraph(_State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def system_message() -> SystemMessage:
    return SystemMessage(content=_SYSTEM_PROMPT)
