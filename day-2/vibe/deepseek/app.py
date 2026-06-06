#!/usr/bin/env python3
"""Flask web app for browsing GDPR articles — summaries + full text."""

import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
GDPR_FILE = Path(__file__).resolve().parent / "GDPR.json"


def load_articles() -> list[dict]:
    with open(GDPR_FILE, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("GDPR", [])


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/articles")
def api_articles():
    """Return all articles (summary only for list view)."""
    articles = load_articles()
    # Return lightweight list for the sidebar
    return jsonify([
        {"article": a["article"], "summary": a["summary"]}
        for a in articles
    ])


@app.route("/api/articles/<article_id>")
def api_article_detail(article_id: str):
    """Return full detail for one article."""
    articles = load_articles()
    for a in articles:
        if a["article"] == article_id:
            return jsonify(a)
    return jsonify({"error": "not found"}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7001, debug=True)
