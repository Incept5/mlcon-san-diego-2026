#!/usr/bin/env python3
"""Scrape GDPR articles from gdpr-info.eu for articles 20–25."""

import argparse
import json
import re
import time
from pathlib import Path

import requests


BASE_URL = "https://gdpr-info.eu"


def fetch_page(article_number: int) -> str:
    """Fetch the full HTML page for a given GDPR article."""
    url = f"{BASE_URL}/art{article_number}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def clean_text(raw: str) -> str:
    """Strip HTML tags, collapse whitespace, produce clean plain text."""
    # Remove script/style
    text = re.sub(r"<(?:script|style|nav|aside|footer|header)[^>]*>.*?</(?:script|style|nav|aside|footer|header)>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    # Remove all other tags
    text = re.sub(r"<[^>]+>", " ", raw)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_article_number(raw: str) -> int | None:
    """Parse the article number from page title or meta."""
    m = re.search(r"Article\s+(\d+)\b", raw, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None


def scrape_article(raw_html: str, article_number: int) -> dict | None:
    """Extract title and body text from the scraped page."""
    title_match = re.search(r"<title>(.*?)</title>", raw_html, re.IGNORECASE)
    title = f"Article {article_number}" if not title_match else title_match.group(1).strip()

    # Try to find the main content section
    body_match = re.search(r'<main[^>]*>(.*?)</main>', raw_html, re.DOTALL | re.IGNORECASE)
    if body_match:
        body_raw = body_match.group(1)
    else:
        body_raw = raw_html

    text = clean_text(body_raw)

    # Find the actual article content (skip nav, breadcrumbs etc.)
    # Articles on gdpr-info.eu are in paragraphs with legal text
    p_tags = re.findall(r"<p[^>]*>(.*?)</p>", body_raw, re.DOTALL | re.IGNORECASE)
    paragraphs = []
    for p in p_tags:
        cleaned = clean_text(p)
        if len(cleaned) > 10:  # skip trivial paragraphs
            paragraphs.append(cleaned)

    full_text = "\n\n".join(paragraphs)

    if not full_text.strip():
        return None

    return {
        "article": f"Art. {article_number}",
        "full_article": full_text,
    }


def save_json(data: list[dict], path: Path):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(data)} articles to {path}")


def main():
    parser = argparse.ArgumentParser(description="Scrape GDPR articles 20–25")
    parser.add_argument("--start", type=int, default=20)
    parser.add_argument("--end", type=int, default=25)
    parser.add_argument("--output", type=str, default="gdpr_articles_raw.json")
    args = parser.parse_args()

    output_path = Path(args.output)
    articles = []

    for n in range(args.start, args.end + 1):
        print(f"[{n}/{args.end}] Fetching Art. {n} …")
        try:
            html = fetch_page(n)
            article = scrape_article(html, n)
            if article:
                articles.append(article)
                print(f"  ✓ Got text ({len(article['full_article'])} chars)")
            else:
                print(f"  ✗ No extractable text for Art. {n}")
        except Exception as e:
            print(f"  ✗ Error: {e}")

        time.sleep(1)  # polite delay

    save_json(articles, output_path)


if __name__ == "__main__":
    main()
