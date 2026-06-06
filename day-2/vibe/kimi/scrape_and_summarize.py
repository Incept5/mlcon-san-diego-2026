#!/usr/bin/env python3
"""
Scrape GDPR articles 20-25 from https://gdpr-info.eu/ and generate
AI summaries using a local Ollama instance (qwen3.5:4b).

Usage:
    python scrape_and_summarize.py           # articles 20-25 (default)
    python scrape_and_summarize.py 20 25     # custom range
    python scrape_and_summarize.py --force   # overwrite existing JSON
"""

import argparse
import json
import re
import sys
from pathlib import Path

import ollama
import requests
from bs4 import BeautifulSoup

HERE = Path(__file__).parent
DATA_FILE = HERE / "gdpr_articles.json"
BASE_URL = "https://gdpr-info.eu/art-{num}-gdpr/"

OLLAMA_MODEL = "qwen3.5:4b"
OLLAMA_OPTIONS = {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "num_ctx": 4096,
}

SUMMARY_PROMPT = """You are a concise legal-summary assistant.
Read the following GDPR article text and produce a single-sentence summary
that captures the core right or obligation described. Be precise and avoid
florid language. Output ONLY the summary sentence — no preamble, no quotes.

Article text:
{text}
"""


def fetch_article(num: int) -> str:
    """Fetch and extract clean article text from the GDPR website."""
    url = BASE_URL.format(num=num)
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, footer, sidebar, ads
    for sel in ("nav", "footer", "header", "aside", ".nav", ".menu", ".sidebar",
                ".widget", ".cookie-banner", ".comments", "script", "style"):
        for tag in soup.select(sel):
            tag.decompose()

    # Try to locate the main article content
    content = (
        soup.select_one('div[data-elementor-type="single-post"]')
        or soup.select_one(".entry-content")
        or soup.select_one("article")
        or soup.select_one("main")
        or soup.find("body")
    )

    if not content:
        raise RuntimeError(f"Could not find article content for Art. {num}")

    # Prefer paragraphs and list items; drop very short fragments
    texts = []
    for tag in content.find_all(["p", "li", "h1", "h2", "h3", "h4"]):
        txt = tag.get_text(separator=" ", strip=True)
        if len(txt) > 3:
            texts.append(txt)

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for t in texts:
        if t not in seen:
            seen.add(t)
            deduped.append(t)

    article_text = "\n\n".join(deduped)

    # Clean up excessive whitespace
    article_text = re.sub(r"\n{3,}", "\n\n", article_text)
    article_text = re.sub(r" {2,}", " ", article_text)
    return article_text.strip()


def summarize(text: str) -> str:
    """Ask the local LLM for a one-sentence summary."""
    prompt = SUMMARY_PROMPT.format(text=text[:8000])  # safety cap
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            think=False,
            options=OLLAMA_OPTIONS,
        )
        summary = response.get("response", "").strip()
        # Remove accidental surrounding quotes
        summary = summary.strip('"').strip("'")
        return summary
    except Exception as exc:
        print(f"  ⚠️  Ollama error: {exc}")
        return "Summary unavailable."


def build_json(articles: list[dict]) -> dict:
    return {"GDPR": articles}


def main():
    parser = argparse.ArgumentParser(description="Scrape & summarize GDPR articles")
    parser.add_argument("start", nargs="?", type=int, default=20, help="First article")
    parser.add_argument("end", nargs="?", type=int, default=25, help="Last article")
    parser.add_argument("--force", action="store_true", help="Overwrite existing JSON")
    args = parser.parse_args()

    if DATA_FILE.exists() and not args.force:
        print(f"📄 {DATA_FILE.name} already exists. Use --force to regenerate.")
        sys.exit(0)

    results = []
    for num in range(args.start, args.end + 1):
        print(f"\n🔍 Fetching Art. {num} ...")
        try:
            full_text = fetch_article(num)
        except Exception as exc:
            print(f"  ❌ Failed to fetch Art. {num}: {exc}")
            continue

        print(f"   Scraped {len(full_text)} chars. Summarizing ...")
        summary = summarize(full_text)
        print(f"   Summary: {summary[:100]}...")

        results.append({
            "article": f"Art. {num}",
            "summary": summary,
            "full_article": full_text,
        })

    if not results:
        print("No articles collected — aborting.")
        sys.exit(1)

    DATA_FILE.write_text(
        json.dumps(build_json(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n✅ Saved {len(results)} articles to {DATA_FILE}")


if __name__ == "__main__":
    main()
