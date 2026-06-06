"""Scrape GDPR articles from gdpr-info.eu and summarise each with a local LLM.

Local-inference demo (no API key): qwen3.5:4b running on Ollama (:11434).

Usage:
    python scrape_gdpr.py            # default: articles 20-25 (smoke test)
    python scrape_gdpr.py 1 99       # full run: every article that exists in [1, 99]

Output: gdpr.json -> {"GDPR": [{"article", "title", "summary", "full_article"}, ...]}
"""

import json
import re
import sys
import time

import ollama
import requests
from bs4 import BeautifulSoup

MODEL = "qwen3.5:4b"
BASE_URL = "https://gdpr-info.eu/art-{n}-gdpr/"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; gdpr-scraper/1.0)"}
OUTPUT = "gdpr.json"


def fetch_article(n: int):
    """Return (article_label, title, full_article_text) or None if the page is missing."""
    url = BASE_URL.format(n=n)
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    h1 = soup.find("h1")
    if not h1:
        return None
    heading = h1.get_text(" ", strip=True)
    # e.g. "Art. 22 GDPR Automated individual decision-making, including profiling"
    m = re.match(r"(Art\.\s*\d+)\s*GDPR\s*(.*)", heading)
    if not m:
        return None
    article_label = m.group(1).strip()
    title = m.group(2).strip()

    content = soup.find(class_="entry-content")
    if not content:
        return None

    # The source wraps each sentence in a <sup> superscript sentence-number; drop them.
    for sup in content.find_all("sup"):
        sup.decompose()

    full_article = render_content(content)
    if not full_article.strip():
        return None

    return article_label, title, full_article


def render_content(content) -> str:
    """Render the entry-content node into numbered/lettered plain text matching the law layout."""
    lines = []

    def walk(ol, depth: int):
        # numbering style: top-level = "1.", first nesting = "(a)", deeper = roman-ish fallback
        items = [c for c in ol.find_all("li", recursive=False)]
        for idx, li in enumerate(items, start=1):
            # text of this li excluding nested lists
            own_parts = []
            for child in li.children:
                if getattr(child, "name", None) in ("ol", "ul"):
                    continue
                text = child.get_text(" ", strip=True) if hasattr(child, "get_text") else str(child).strip()
                if text:
                    own_parts.append(text)
            own_text = " ".join(own_parts).strip()
            own_text = re.sub(r"\s+", " ", own_text)

            if depth == 0:
                marker = f"{idx}."
            elif depth == 1:
                marker = f"({chr(ord('a') + idx - 1)})"
            else:
                marker = f"({idx})"

            indent = "    " * depth
            if own_text:
                lines.append(f"{indent}{marker} {own_text}")
            else:
                lines.append(f"{indent}{marker}")

            for nested in li.find_all(["ol", "ul"], recursive=False):
                walk(nested, depth + 1)

    top_lists = content.find_all(["ol", "ul"], recursive=False)
    if top_lists:
        for ol in top_lists:
            walk(ol, 0)
    else:
        # Some articles are plain paragraphs rather than a numbered list.
        for p in content.find_all("p", recursive=False):
            t = re.sub(r"\s+", " ", p.get_text(" ", strip=True))
            if t:
                lines.append(t)

    return "\n".join(lines)


def summarise(article_label: str, title: str, full_article: str) -> str:
    """Ask the local LLM for a plain-English summary of the article."""
    prompt = (
        f"You are a legal-plain-language assistant. Summarise the following GDPR article "
        f"in 2-3 clear sentences for a non-lawyer. Explain what right or obligation it creates "
        f"and who it affects. Do not add a preamble, do not repeat the article number, just give the summary.\n\n"
        f"{article_label} — {title}\n\n{full_article}"
    )
    resp = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        think=False,  # qwen3.5 non-thinking mode
        options={"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    )
    return resp["message"]["content"].strip()


def main():
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 25

    results = []
    for n in range(start, end + 1):
        fetched = fetch_article(n)
        if fetched is None:
            print(f"  Art. {n}: not found / skipped")
            continue
        article_label, title, full_article = fetched
        print(f"  {article_label} ({title}) — summarising with {MODEL}...", flush=True)
        summary = summarise(article_label, title, full_article)
        results.append(
            {
                "article": article_label,
                "title": title,
                "summary": summary,
                "full_article": full_article,
            }
        )
        time.sleep(0.3)  # be polite to the site

    payload = {"GDPR": results}
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(results)} articles to {OUTPUT}")


if __name__ == "__main__":
    main()
