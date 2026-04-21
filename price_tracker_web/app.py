import os
import re
import sqlite3
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, url_for, redirect, send_file
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(APP_DIR, "price_tracker.db")

app = Flask(__name__)


# ----------------------------
# DB
# ----------------------------
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT NOT NULL,
            title TEXT,
            price REAL NOT NULL,
            target REAL NOT NULL,
            decision TEXT NOT NULL,
            trend TEXT NOT NULL,
            predicted REAL,
            ai_comment TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


init_db()


# ----------------------------
# Scrape helpers
# ----------------------------
def normalize_price_to_float(price_text: str) -> float:
    """
    Examples:
      "£51.77" -> 51.77
      "$1,299.00" -> 1299.00
      "NT$ 1,280" -> 1280
    """
    if not price_text:
        raise ValueError("Empty price text")

    s = price_text.strip()
    s = s.replace(",", "")
    # keep digits and dot only (remove currency symbols and letters)
    s = re.sub(r"[^0-9.]", "", s)
    if s == "" or s == ".":
        raise ValueError(f"Could not normalize price: {price_text}")
    return float(s)


def fetch_books_to_scrape(url: str):
    """
    For the course demo site (books.toscrape.com), price is in:
      p.price_color
      title in h1
    """
    r = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    title = soup.select_one("div.product_main h1")
    price = soup.select_one("p.price_color")

    if not title or not price:
        raise ValueError("Could not parse title/price from this page.")

    title_text = title.get_text(strip=True)
    price_text = price.get_text(strip=True)
    price_value = normalize_price_to_float(price_text)

    # for display we reuse currency symbol from original string if possible
    currency_symbol = ""
    m = re.search(r"[£$¥]|NT\$", price_text)
    if m:
        currency_symbol = m.group(0)
    else:
        # fallback: first non-digit char if exists
        for ch in price_text:
            if not ch.isdigit() and ch not in [".", ",", " "]:
                currency_symbol = ch
                break

    return title_text, price_value, currency_symbol


def fetch_product(url: str):
    # 現状は books.toscrape 用（Amazon等は別対応が必要）
    return fetch_books_to_scrape(url)


# ----------------------------
# Trend & "Simple ML"
# ----------------------------
def compute_trend(prices):
    if len(prices) < 2:
        return "N/A"
    diff = prices[-1] - prices[-2]
    if abs(diff) < 1e-9:
        return "FLAT"
    return "UP" if diff > 0 else "DOWN"


def predict_next_price(prices):
    """
    Super simple baseline:
    - if only 1 data point -> predict same
    - else -> last + (last - prev)  (1-step momentum)
    """
    if not prices:
        return None
    if len(prices) == 1:
        return float(prices[-1])
    return float(prices[-1] + (prices[-1] - prices[-2]))


def make_ai_comment(price, target, trend, predicted):
    decision = "BUY" if price <= target else "WAIT"
    if predicted is None:
        return f"Decision: {decision}. Trend: {trend}."
    return f"Decision: {decision}. Trend: {trend}. Next price may be {predicted:.2f}."


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    result = None

    default_url = ""
    url_txt = os.path.join(APP_DIR, "URL.txt")
    if os.path.exists(url_txt):
        try:
            with open(url_txt, "r", encoding="utf-8") as f:
                default_url = f.read().strip()
        except:
            default_url = ""

    if request.method == "POST":
        url = (request.form.get("url") or "").strip()
        target_str = (request.form.get("target") or "").strip()

        if not url:
            error = "Fetch failed: URL is required."
        elif not target_str:
            error = "Fetch failed: Target price is required."
        else:
            try:
                target = float(target_str)
                title, price, currency = fetch_product(url)

                decision = "BUY" if price <= target else "WAIT"

                # Pull recent history for trend/prediction
                conn = get_conn()
                cur = conn.cursor()
                cur.execute(
                    "SELECT price FROM price_history WHERE url=? ORDER BY id ASC",
                    (url,),
                )
                past = [row["price"] for row in cur.fetchall()]
                series = past + [price]

                trend = compute_trend(series)
                predicted = predict_next_price(series)
                ai_comment = make_ai_comment(price, target, trend, predicted)

                created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                cur.execute(
                    """
                    INSERT INTO price_history
                    (url, title, price, target, decision, trend, predicted, ai_comment, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (url, title, price, target, decision, trend, predicted, ai_comment, created_at),
                )
                conn.commit()
                conn.close()

                result = {
                    "url": url,
                    "title": title,
                    "currency": currency,
                    "price": price,
                    "target": target,
                    "decision": decision,
                    "trend": trend,
                    "predicted": predicted,
                    "ai_comment": ai_comment,
                }

            except Exception as e:
                error = f"Fetch failed: {e}"

    return render_template("index.html", result=result, error=error, default_url=default_url)


@app.route("/history")
def history():
    url = (request.args.get("url") or "").strip()

    conn = get_conn()
    cur = conn.cursor()

    if url:
        cur.execute("SELECT * FROM price_history WHERE url=? ORDER BY id DESC LIMIT 200", (url,))
    else:
        cur.execute("SELECT * FROM price_history ORDER BY id DESC LIMIT 200")

    rows = cur.fetchall()
    conn.close()

    return render_template("history.html", rows=rows, url=url)


@app.route("/graph")
def graph():
    url = (request.args.get("url") or "").strip()
    return render_template("graph.html", url=url)


@app.route("/plot.png")
def plot_png():
    url = (request.args.get("url") or "").strip()

    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT created_at, price, target FROM price_history WHERE url=? ORDER BY id ASC",
        (url,),
    )
    rows = cur.fetchall()
    conn.close()

    if not rows:
        # empty figure
        fig = plt.figure()
        plt.title("No data yet")
    else:
        times = [r["created_at"] for r in rows]
        prices = [r["price"] for r in rows]
        targets = [r["target"] for r in rows]
        target_line = targets[-1] if targets else None

        fig = plt.figure(figsize=(10, 4))
        plt.plot(times, prices, marker="o")
        if target_line is not None:
            plt.axhline(target_line, linestyle="--")
        plt.title("Price Trend")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()

    out_path = os.path.join(APP_DIR, "_plot_tmp.png")
    fig.savefig(out_path)
    plt.close(fig)
    return send_file(out_path, mimetype="image/png")


if __name__ == "__main__":
    app.run(debug=True)
