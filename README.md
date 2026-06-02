# 📊 Agentic Analytics Assistant

Ask a question about your data in plain English. An autonomous agent inspects the
database schema, writes SQL, runs it, **fixes its own queries when they error**, and
returns a plain-language answer with a chart.

Built as a **harness + tools + policies**:

- **Harness** — a [LangGraph](https://langchain-ai.github.io/langgraph/) agent loop
  drives the model: *think → call a tool → observe → repeat* until it's done.
- **Tools** — the model can only act through explicit, named functions
  (`list_tables`, `get_schema`, `run_sql`, `submit_answer`).
- **Policies** — every consequential call is gated by a policy engine
  (read-only SQL, table allowlist, step/query/row budgets) *before* it runs.

Powered by the **free** [Google Gemini API](https://aistudio.google.com/app/apikey)
via `langchain-google-genai`. LangGraph and LangChain are open source — the whole
stack is free.

---

## How it works

```
Your question
     │
     ▼
  ┌────────────┐   wants to use a tool?   ┌──────────────────────────┐
  │   agent    ├──────────  yes  ────────▶│  tools (policy-gated)    │
  │  (Gemini)  │◀────────  observation  ──┤  list_tables / get_schema│
  └─────┬──────┘                          │  run_sql / submit_answer │
        │ submit_answer / budget hit      └──────────────────────────┘
        ▼
   answer + chart
```

The model decides each step; the harness just loops and the policy engine enforces
the rules. Every step (reasoning, tool calls, policy blocks, errors) is recorded in
an **agent trace** you can expand in the UI.

The agent is **read-only by design**: the policy engine rejects anything that isn't a
single `SELECT`/`WITH` statement *and* the database is opened in read-only mode, so a
hallucinated `DROP`/`UPDATE` can never touch your data. Policy rejections are fed back
to the model as observations, so it self-corrects just like it does on a SQL error.

## Quickstart

Requires Python 3.10+.

```bash
# 1. Install dependencies (a virtualenv is recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Add your free Gemini API key
cp .env.example .env
#   then edit .env and paste your key from https://aistudio.google.com/app/apikey

# 3. Build the sample database (e-commerce: customers, products, orders, order_items)
python seed_data.py

# 4. Launch the app
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501) and ask away.

## Example questions

- *What were the top 5 best-selling products by revenue?*
- *How many orders were placed each month?*
- *Which countries have the most customers?*
- *What is the total revenue from completed orders by category?*
- *What share of orders ended up cancelled or refunded?*

## Configuration

All optional, set in `.env`:

| Variable         | Default              | Description                                      |
| ---------------- | -------------------- | ------------------------------------------------ |
| `GEMINI_API_KEY` | _(required)_         | Your free Gemini API key.                        |
| `GEMINI_MODEL`   | `gemini-2.5-flash`   | Any Gemini model your key can access.            |
| `DATABASE_PATH`  | `data/sample.db`     | Point at your own SQLite database to query it.   |

### Use your own database

Any SQLite file works — set `DATABASE_PATH` to it and skip `seed_data.py`. The agent
introspects whatever schema it finds.

## Project layout

```
app.py                       Streamlit UI
seed_data.py                 Generates the sample SQLite database
agentic_analytics/
  ├── graph.py               Harness: the LangGraph agent loop
  ├── tools.py               Tools the model can call (list_tables, get_schema, run_sql, submit_answer)
  ├── policies.py            Policy engine: read-only SQL, allowlist, step/query/row budgets
  ├── llm.py                 Gemini chat model via langchain-google-genai
  ├── state.py               Trace types + the shared per-question run context
  ├── agent.py               Facade: runs one question through the graph → AgentResult
  └── database.py            Read-only SQLite access + schema introspection
```

### Tuning the policies

Pass a `PolicyConfig` to `AnalyticsAgent` to tighten or loosen the agent's authority:

```python
from agentic_analytics.policies import PolicyConfig

agent = AnalyticsAgent(
    db=db,
    api_key=api_key,
    policy=PolicyConfig(
        allowed_tables={"orders", "order_items", "products"},  # default: any table
        max_steps=8,         # agent turns before it must stop
        max_sql_attempts=4,  # how many queries it may run
        row_limit=5000,      # rows returned to user / model
    ),
)
```

## License

MIT — see [LICENSE](LICENSE).
