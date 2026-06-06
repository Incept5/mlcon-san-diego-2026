#!/usr/bin/env python3
"""Scrape GDPR articles 20-25 from gdpr-info.eu and summarize via Ollama."""

import json
import re
import sys
import time

import requests
from bs4 import BeautifulSoup, NavigableString, Tag

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3.5:4b"
ARTICLES = list(range(20, 26))  # 20–25 inclusive
OUTPUT_FILE = "GDPR.json"

# ── helpers ────────────────────────────────────────────────────────────────


def fetch_article(num: int) -> dict | None:
    """Scrape one GDPR article. Returns dict with title, number, full_text."""
    url = f"https://gdpr-info.eu/art-{num}-gdpr/"
    print(f"  Fetching {url} ...", end=" ", flush=True)

    try:
        resp = requests.get(url, timeout=15, headers={"User-Agent": "GDPR-scraper/1.0"})
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"FAILED: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Title
    h1 = soup.find("h1")
    title = h1.get_text(strip=True) if h1 else f"Art. {num} GDPR"
    # Clean title — remove repeated "GDPR" and extra spaces
    title = re.sub(r"\s+", " ", title).strip()

    # Article body
    content_div = soup.find("div", class_="entry-content")
    if not content_div:
        print("FAILED: no .entry-content")
        return None

    # Clean out <aside>, <script>, <style>, <nav>
    for tag in content_div.find_all(["aside", "script", "style", "nav", "footer"]):
        tag.decompose()

    # Collect paragraphs, preserving structure
    paragraphs: list[str] = []
    for child in content_div.children:
        if isinstance(child, NavigableString):
            txt = str(child).strip()
            if txt and len(txt) > 3:
                paragraphs.append(txt)
        elif isinstance(child, Tag):
            if child.name in ("p", "div"):
                txt = child.get_text(" ", strip=True)
                if txt and len(txt) > 3:
                    paragraphs.append(txt)
            elif child.name in ("ol", "ul"):
                for li in child.find_all("li"):
                    txt = li.get_text(" ", strip=True)
                    if txt:
                        paragraphs.append(f"• {txt}")
            elif child.name in ("h2", "h3", "h4"):
                # Skip "Suitable Recitals" and similar boilerplate
                txt = child.get_text(strip=True)
                if "recital" not in txt.lower() and "suitable" not in txt.lower():
                    paragraphs.append(txt)

    # Stop before "Suitable Recitals" if it snuck in
    clean: list[str] = []
    for p in paragraphs:
        if "suitable recitals" in p.lower():
            break
        clean.append(p)

    full_text = "\n\n".join(clean)
    if not full_text:
        print("FAILED: empty body")
        return None

    print(f"OK ({len(full_text)} chars)")
    return {
        "article": f"Art. {num}",
        "title": title,
        "full_article": full_text,
    }


def summarize(text: str, article_label: str) -> str:
    """Ask Ollama to produce a 1–2 sentence plain-language summary."""
    prompt = (
        "You are a legal editor. Read the following GDPR article and produce "
        "a 1–2 sentence plain-English summary that captures what the article is about. "
        "Make it useful for someone browsing a list to decide whether to read the full text. "
        "Do NOT include markdown, quotes, or preamble. Just the summary.\n\n"
        f"ARTICLE: {text}\n\nSUMMARY:"
    )

    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "think": False,  # disable thinking mode for Qwen3.5 on Ollama
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 250,
        },
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()
        summary = result.get("response", "").strip()
        # Clean up quotes
        summary = summary.strip('"').strip("'").strip()
        return summary
    except requests.RequestException as e:
        print(f"    Ollama error: {e}")
        return "[summary unavailable]"


# ── main ───────────────────────────────────────────────────────────────────


def main():
    print("=== GDPR Article Scraper + Summarizer ===\n")

    # Check Ollama
    print("Checking Ollama …", end=" ", flush=True)
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        if MODEL not in models:
            print(f"WARNING: {MODEL} not found in {models}")
        else:
            print("OK")
    except requests.RequestException as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    # Scrape
    articles_raw: list[dict] = []
    for num in ARTICLES:
        art = fetch_article(num)
        if art:
            articles_raw.append(art)
        time.sleep(0.5)  # be polite

    print(f"\nScraped {len(articles_raw)} articles.\n")

    # Summarize
    print("Generating summaries via Ollama …")
    gdpr_list: list[dict] = []
    for art in articles_raw:
        label = art["article"]
        print(f"  Summarizing {label} …", end=" ", flush=True)
        summary = summarize(art["full_article"], label)
        print(summary[:80] + ("…" if len(summary) > 80 else ""))
        gdpr_list.append({
            "article": art["article"],
            "summary": summary,
            "full_article": art["full_article"],
        })
        time.sleep(0.3)

    # Save
    output = {"GDPR": gdpr_list}
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(gdpr_list)} articles to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
