"""The policy layer.

Policies are the rules the agent must obey. They are *separate* from the agent's
reasoning and from the tools: the harness routes every tool call through this
module before it is allowed to take effect. A rejected call is not an exception
the program crashes on — it is turned into an observation that is fed back to the
model, so the agent can read *why* it was blocked and correct itself.

Two kinds of policy live here:

  * **SQL policies**   — what a generated query is allowed to do
                         (read-only, single statement, table allowlist).
  * **Run policies**   — how hard the agent is allowed to try
                         (max steps, max failed SQL attempts, row limit).

Keeping them in one place means you can audit and tighten the agent's authority
without touching the agent loop or the tools.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Keywords a read-only analyst must never use.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|detach|pragma|vacuum)\b",
    re.IGNORECASE,
)

# Pull table names out of FROM / JOIN clauses to check them against an allowlist.
_TABLE_REF = re.compile(r"\b(?:from|join)\s+[\"'`]?([a-zA-Z_][a-zA-Z0-9_]*)[\"'`]?", re.IGNORECASE)


@dataclass
class PolicyConfig:
    """Tunable limits on the agent's authority. Edit these to tighten/loosen it."""

    # SQL policies
    allowed_tables: frozenset[str] | None = None  # None = any table in the DB
    # Run policies
    max_steps: int = 8          # total agent turns (think -> act) before we stop
    max_sql_attempts: int = 4   # how many times run_sql may be called in a session
    row_limit: int = 5000       # rows returned to the user / model


@dataclass
class Decision:
    """The result of checking one action against policy."""

    allowed: bool
    reason: str = ""

    @classmethod
    def ok(cls) -> "Decision":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> "Decision":
        return cls(allowed=False, reason=reason)


@dataclass
class PolicyEngine:
    """Evaluates actions against a :class:`PolicyConfig`.

    The engine is stateless about the conversation except for the counters it is
    asked to enforce; the harness passes the current counts in.
    """

    config: PolicyConfig = field(default_factory=PolicyConfig)

    # --- SQL policies -----------------------------------------------------

    def check_sql(self, sql: str) -> Decision:
        stripped = (sql or "").strip().rstrip(";").strip()
        if not stripped:
            return Decision.deny("Empty query.")
        if ";" in stripped:
            return Decision.deny("Only a single statement is allowed (found ';').")
        if not re.match(r"^(select|with)\b", stripped, re.IGNORECASE):
            return Decision.deny("Only read-only SELECT / WITH queries are allowed.")
        if _FORBIDDEN.search(stripped):
            return Decision.deny(
                "Query contains a forbidden, data-modifying keyword. "
                "This agent may only read data."
            )
        if self.config.allowed_tables is not None:
            referenced = {t.lower() for t in _TABLE_REF.findall(stripped)}
            allowed = {t.lower() for t in self.config.allowed_tables}
            blocked = referenced - allowed
            if blocked:
                return Decision.deny(
                    f"Query references table(s) not on the allowlist: {sorted(blocked)}. "
                    f"Allowed tables: {sorted(allowed)}."
                )
        return Decision.ok()

    # --- Run policies -----------------------------------------------------

    def check_step_budget(self, steps_taken: int) -> Decision:
        if steps_taken >= self.config.max_steps:
            return Decision.deny(
                f"Step budget exhausted ({self.config.max_steps} steps). Stopping."
            )
        return Decision.ok()

    def check_sql_budget(self, attempts_made: int) -> Decision:
        if attempts_made >= self.config.max_sql_attempts:
            return Decision.deny(
                f"SQL attempt budget exhausted ({self.config.max_sql_attempts} attempts)."
            )
        return Decision.ok()
