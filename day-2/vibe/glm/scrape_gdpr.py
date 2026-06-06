"""
Scrape GDPR articles from gdpr-info.eu and generate summaries
using a local LLM (qwen3.5:4b) via Ollama.

Usage:
    python scrape_gdpr.py              # articles 20-25 (smoke test)
    python scrape_gdpr.py --all        # all 99 articles
    python scrape_gdpr.py 40 50        # articles 40-50
"""

import json
import re
import sys
import time
import urllib.request


def _strip_html(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    # Replace <br>, <li>, </p> etc. with newlines for readability
    html = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</li>", "\n", html, flags=re.IGNORECASE)
    html = re.sub(r"</p>", "\n\n", html, flags=re.IGNORECASE)
    # Strip all remaining tags
    text = re.sub(r"<[^>]+>", "", html)
    # Decode common HTML entities
    for entity, char in [("&#8217;", "'"), ("&#8216;", "'"),
                         ("&#8220;", '"'), ("&#8221;", '"'),
                         ("&#8211;", "–"), ("&#8212;", "—"),
                         ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                         ("&nbsp;", " ")]:
        text = text.replace(entity, char)
    # Collapse whitespace but preserve paragraph breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fetch_article(article_num: int) -> dict:
    """Fetch and parse a single GDPR article from gdpr-info.eu."""
    url = f"https://gdpr-info.eu/art-{article_num}-gdpr/"
    req = urllib.request.Request(url, headers={"User-Agent": "GDPRScraper/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")

    # Extract the <h1 class="entry-title"> — contains the article number and title
    h1_match = re.search(
        r'<h1\s+class="entry-title">(.*?)</h1>', html, re.DOTALL
    )
    if h1_match:
        h1_raw = h1_match.group(1)
        # The h1 has <span class="dsgvo-number">Art. NN GDPR </span>
        # and <span class="dsgvo-title">Title text</span>
        num_match = re.search(r'class="dsgvo-number">(.*?)</span>', h1_raw)
        title_match = re.search(r'class="dsgvo-title">(.*?)</span>', h1_raw)
        article_label = num_match.group(1).strip() if num_match else f"Art. {article_num}"
        # Normalize: "Art. 22 GDPR" -> "Art. 22"
        article_label = re.sub(r"\s*GDPR\s*$", "", article_label).strip()
        article_title = title_match.group(1).strip() if title_match else article_label
    else:
        article_label = f"Art. {article_num}"
        article_title = article_label

    # Extract content from entry-content div, stopping before recitals/nav
    # The article body is the <ol> inside entry-content, before the recitals section
    content_match = re.search(
        r'<div\s+class="entry-content">(.*?)<div\s+class="empfehlung-erwaegungsgruende">',
        html, re.DOTALL
    )
    if not content_match:
        # Fallback: try stopping at page-navigation
        content_match = re.search(
            r'<div\s+class="entry-content">(.*?)<div\s+class="page-navigation">',
            html, re.DOTALL
        )

    if content_match:
        full_text = _strip_html(content_match.group(1))
    else:
        full_text = ""

    return {
        "article": article_label,
        "title": f"{article_label} – {article_title}",
        "full_article": full_text,
    }


# ── Ollama summariser ─────────────────────────────────────────────────

def summarize_with_ollama(article_text: str, article_label: str) -> str:
    """Send article text to local Ollama qwen3.5:4b for a concise summary."""
    prompt = (
        f"Summarize GDPR {article_label} in 2-3 clear sentences. "
        f"Focus on what right or obligation it creates and for whom. "
        f"Write in plain English."
    )

    payload = json.dumps({
        "model": "qwen3.5:4b",
        "messages": [{"role": "user", "content": prompt + f"\n\nArticle text:\n{article_text}"}],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 256,
        },
    }).encode()

    req = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode())

    summary = data.get("message", {}).get("content", "").strip()
    return summary


# ── Main ──────────────────────────────────────────────────────────────

def main():
    # Determine article range
    if "--all" in sys.argv:
        start, end = 1, 99
    elif len(sys.argv) == 3 and sys.argv[1].isdigit() and sys.argv[2].isdigit():
        start, end = int(sys.argv[1]), int(sys.argv[2])
    else:
        start, end = 20, 25  # default smoke test

    articles = []
    for num in range(start, end + 1):
        print(f"📡 Fetching Art. {num}...", end=" ", flush=True)
        try:
            art = fetch_article(num)
        except Exception as e:
            print(f"❌ fetch error: {e}")
            continue

        if not art["full_article"]:
            print(f"⚠️  empty article, skipping")
            continue

        print(f"🤖 Summarizing...", end=" ", flush=True)
        try:
            art["summary"] = summarize_with_ollama(art["full_article"], art["article"])
        except Exception as e:
            print(f"❌ ollama error: {e}")
            art["summary"] = "(summary unavailable)"

        print(f"✅ {art['article']}")
        articles.append(art)
        time.sleep(0.5)  # polite crawl delay

    output = {"GDPR": articles}
    out_path = "gdpr_articles.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(articles)} articles to {out_path}")


if __name__ == "__main__":
    main()
