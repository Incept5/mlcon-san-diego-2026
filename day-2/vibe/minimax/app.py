"""
GDPR Article Browser
====================
Flask server that serves a professional UI for browsing the scraped
GDPR articles.  Click a summary to expand the full article text.

Usage:
    python app.py            # serves on http://localhost:7004

The JSON file lives at data/articles.json and is loaded at startup.
The page is rendered server-side (no JS framework) so it scales
cleanly from 6 to 99 articles.
"""

import json
import re
from pathlib import Path

from flask import Flask, abort, render_template, request

BASE = Path(__file__).parent
DATA_PATH = BASE / "data" / "articles.json"
PORT = 7004

app = Flask(__name__)


def load_articles() -> list[dict]:
    if not DATA_PATH.exists():
        return []
    with DATA_PATH.open() as f:
        doc = json.load(f)
    return doc.get("GDPR", [])


# Simple in-process cache; refresh via the `/refresh` endpoint or restart.
ARTICLES = load_articles()


def article_sort_key(a: dict) -> int:
    m = re.search(r"\d+", a.get("article", ""))
    return int(m.group()) if m else 0


@app.route("/")
def index():
    articles = sorted(ARTICLES, key=article_sort_key)
    return render_template(
        "index.html",
        articles=articles,
        count=len(articles),
    )


@app.route("/article/<int:n>")
def article(n: int):
    """Return the full text of article `n` as an HTML fragment."""
    label = f"Art. {n}"
    for a in ARTICLES:
        if a["article"] == label:
            paragraphs = [p for p in a["full_article"].split("\n") if p.strip()]
            return render_template(
                "_article.html",
                article=a,
                paragraphs=paragraphs,
            )
    abort(404)


@app.route("/api/articles")
def api_articles():
    """JSON list of {article, summary} — used by the client-side search box."""
    return json.dumps(
        [{"article": a["article"], "summary": a["summary"]} for a in ARTICLES]
    )


@app.route("/refresh")
def refresh():
    """Reload articles.json from disk (handy after re-running the scraper)."""
    global ARTICLES
    ARTICLES = load_articles()
    return f"Reloaded {len(ARTICLES)} articles."


if __name__ == "__main__":
    print(f"GDPR Browser — {len(ARTICLES)} articles loaded from {DATA_PATH}")
    print(f"Open http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
