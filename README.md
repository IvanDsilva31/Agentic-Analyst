# 📊 Agentic Analytics Assistant

Ask a question about your data in plain English. An autonomous agent inspects the
database schema, writes SQL, runs it, **fixes its own queries when they error**, and
returns a plain-language answer with a chart.

Powered by the **free** [Google Gemini API](https://aistudio.google.com/app/apikey).

---

## How it works

```
Your question
     │
     ▼
1. 🔍 Inspect schema     — read tables, columns, sample rows
2. ✍️  Write SQL          — Gemini generates a read-only SELECT
3. ▶️  Run it             — execute against SQLite
4. 🔧 Self-fix on error  — feed the DB error back to Gemini and retry (up to 3×)
5. 💬 Answer + chart     — plain-language summary + a chart spec, rendered with Plotly
```

Every step is recorded in an **agent trace** you can expand in the UI.

The agent is **read-only by design**: generated queries are validated to be a single
`SELECT`/`WITH` statement and the database is opened in read-only mode, so a
hallucinated `DROP`/`UPDATE` can never touch your data.

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
  ├── agent.py               The autonomous loop (schema → SQL → run → self-fix → answer)
  ├── database.py            Read-only SQLite access + schema introspection
  └── gemini_client.py       Wrapper around the Gemini API
```

## License

MIT — see [LICENSE](LICENSE).
