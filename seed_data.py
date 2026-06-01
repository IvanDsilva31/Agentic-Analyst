"""Generate a sample e-commerce SQLite database for the analytics assistant.

Run once before starting the app:

    python seed_data.py

Creates data/sample.db with customers, products, orders, and order_items.
The data is deterministic (fixed random seed) so results are reproducible.
"""

from __future__ import annotations

import os
import random
import sqlite3
from datetime import date, timedelta

DB_PATH = os.environ.get("DATABASE_PATH", "data/sample.db")

random.seed(42)

COUNTRIES = ["USA", "Canada", "UK", "Germany", "France", "India", "Australia", "Brazil"]
SIGNUP_CHANNELS = ["organic", "paid_search", "social", "referral", "email"]

CATEGORIES = {
    "Electronics": [
        ("Wireless Headphones", 129.99),
        ("Smart Watch", 249.99),
        ("Bluetooth Speaker", 79.99),
        ("USB-C Charger", 24.99),
        ("Laptop Stand", 39.99),
    ],
    "Home & Kitchen": [
        ("Espresso Machine", 199.99),
        ("Chef's Knife", 59.99),
        ("Cast Iron Skillet", 44.99),
        ("Air Purifier", 149.99),
        ("Robot Vacuum", 299.99),
    ],
    "Apparel": [
        ("Running Shoes", 89.99),
        ("Rain Jacket", 119.99),
        ("Merino Wool Socks", 19.99),
        ("Baseball Cap", 24.99),
        ("Leather Belt", 49.99),
    ],
    "Books": [
        ("The Pragmatic Programmer", 39.99),
        ("Designing Data-Intensive Apps", 54.99),
        ("Atomic Habits", 16.99),
        ("Sapiens", 22.99),
    ],
}

ORDER_STATUSES = ["completed", "completed", "completed", "completed", "shipped", "cancelled", "refunded"]


def build_schema(cur: sqlite3.Cursor) -> None:
    cur.executescript(
        """
        DROP TABLE IF EXISTS order_items;
        DROP TABLE IF EXISTS orders;
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS customers;

        CREATE TABLE customers (
            customer_id    INTEGER PRIMARY KEY,
            name           TEXT NOT NULL,
            email          TEXT NOT NULL,
            country        TEXT NOT NULL,
            signup_date    DATE NOT NULL,
            signup_channel TEXT NOT NULL
        );

        CREATE TABLE products (
            product_id   INTEGER PRIMARY KEY,
            name         TEXT NOT NULL,
            category     TEXT NOT NULL,
            unit_price   REAL NOT NULL
        );

        CREATE TABLE orders (
            order_id     INTEGER PRIMARY KEY,
            customer_id  INTEGER NOT NULL REFERENCES customers(customer_id),
            order_date   DATE NOT NULL,
            status       TEXT NOT NULL
        );

        CREATE TABLE order_items (
            order_item_id INTEGER PRIMARY KEY,
            order_id      INTEGER NOT NULL REFERENCES orders(order_id),
            product_id    INTEGER NOT NULL REFERENCES products(product_id),
            quantity      INTEGER NOT NULL,
            unit_price    REAL NOT NULL
        );
        """
    )


def seed(cur: sqlite3.Cursor) -> None:
    first_names = ["Alex", "Jordan", "Sam", "Taylor", "Morgan", "Casey", "Riley", "Jamie",
                   "Avery", "Quinn", "Drew", "Skyler", "Cameron", "Reese", "Harper", "Rowan"]
    last_names = ["Smith", "Johnson", "Lee", "Patel", "Garcia", "Mueller", "Dubois", "Silva",
                  "Nguyen", "Kim", "Brown", "Wilson", "Costa", "Ferraro", "Olsen"]

    # --- products ---
    products = []  # (product_id, unit_price)
    pid = 1
    for category, items in CATEGORIES.items():
        for name, price in items:
            cur.execute(
                "INSERT INTO products (product_id, name, category, unit_price) VALUES (?, ?, ?, ?)",
                (pid, name, category, price),
            )
            products.append((pid, price))
            pid += 1

    # --- customers ---
    start = date(2023, 1, 1)
    n_customers = 200
    for cid in range(1, n_customers + 1):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        email = f"{name.lower().replace(' ', '.')}{cid}@example.com"
        signup = start + timedelta(days=random.randint(0, 540))
        cur.execute(
            "INSERT INTO customers (customer_id, name, email, country, signup_date, signup_channel) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (cid, name, email, random.choice(COUNTRIES), signup.isoformat(),
             random.choice(SIGNUP_CHANNELS)),
        )

    # --- orders + items ---
    order_id = 1
    item_id = 1
    n_orders = 1200
    for _ in range(n_orders):
        cid = random.randint(1, n_customers)
        order_date = start + timedelta(days=random.randint(30, 720))
        status = random.choice(ORDER_STATUSES)
        cur.execute(
            "INSERT INTO orders (order_id, customer_id, order_date, status) VALUES (?, ?, ?, ?)",
            (order_id, cid, order_date.isoformat(), status),
        )
        for _ in range(random.randint(1, 4)):
            prod_id, price = random.choice(products)
            qty = random.randint(1, 3)
            cur.execute(
                "INSERT INTO order_items (order_item_id, order_id, product_id, quantity, unit_price) "
                "VALUES (?, ?, ?, ?, ?)",
                (item_id, order_id, prod_id, qty, price),
            )
            item_id += 1
        order_id += 1


def main() -> None:
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        build_schema(cur)
        seed(cur)
        conn.commit()
        counts = {
            t: cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("customers", "products", "orders", "order_items")
        }
    finally:
        conn.close()
    print(f"Created {DB_PATH}")
    for table, n in counts.items():
        print(f"  {table:12} {n:>5} rows")


if __name__ == "__main__":
    main()
