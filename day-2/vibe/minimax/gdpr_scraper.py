"""
GDPR Article Scraper
====================
Scrapes articles from https://gdpr-info.eu/ and uses a local Ollama model
(qwen3.5:4b) to generate a short summary for each article.

Output JSON shape:
{
  "GDPR": [
    {
      "article": "Art. 22",
      "summary": "...",
      "full_article": "..."
    },
    ...
  ]
}

Usage:
    python gdpr_scraper.py                 # default: articles 20-25 (smoke test)
    python gdpr_scraper.py --all           # all articles
    python gdpr_scraper.py --from 1 --to 11
"""

import argparse
import json
import re
import time
import urllib.request
import urllib.error
from html.parser import HTMLParser
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_OUT = Path(__file__).parent / "data" / "articles.json"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

class ArticleTextExtractor(HTMLParser):
    """Extract the body of a single <div class="entry-content"> block.

    The block contains the GDPR article text first, then 'Suitable Recitals'
    and pagination. We stop at the first non-article sub-element.
    """

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_entry_content = False
        self.depth = 0           # depth inside entry-content
        self.in_article_block = False   # inside the <ol>/<p> that holds the law
        self.article_depth = 0
        self.stop_tags = {
            "div", "aside",
        }
        self._capture = []
        self._title_number = None
        self._title_name = None
        self._in_title = False
        self._in_number = False
        self._in_name = False
        self._title_text = []

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        classes = (attrs_d.get("class") or "").split()

        if tag == "h1" and "entry-title" in classes:
            self._in_title = True
            return
        if self._in_title and "dsgvo-number" in classes:
            self._in_number = True
            return
        if self._in_title and "dsgvo-title" in classes:
            self._in_name = True
            return

        if tag == "div" and "entry-content" in classes:
            self.in_entry_content = True
            self.depth = 1
            return
        if self.in_entry_content:
            self.depth += 1
            # article body is the first <ol> or <p> at the top level
            if self.depth == 2 and tag in ("ol", "p", "ul") and not self.in_article_block:
                self.in_article_block = True
                self.article_depth = 1
                self._capture.append(f"<{tag}>")
                return
            if self.in_article_block:
                self.article_depth += 1
                self._capture.append(self.get_starttag_text() or f"<{tag}>")
            return

        if self._in_title:
            return

    def handle_endtag(self, tag):
        if self._in_title and tag == "h1":
            self._in_title = False
            self._in_number = False
            self._in_name = False
            return
        if self._in_number and tag == "span":
            self._in_number = False
            return
        if self._in_name and tag == "span":
            self._in_name = False
            return

        if not self.in_entry_content:
            return
        if self.in_article_block:
            self.article_depth -= 1
            self._capture.append(f"</{tag}>")
            if self.article_depth <= 0:
                self.in_article_block = False
        self.depth -= 1
        if self.depth <= 0:
            self.in_entry_content = False

    def handle_data(self, data):
        if self._in_number:
            self._title_number = (self._title_number or "") + data
        elif self._in_name:
            self._title_name = (self._title_name or "") + data
        elif self.in_article_block:
            self._capture.append(data)

    # -- public API --------------------------------------------------------

    @property
    def title(self) -> str:
        parts = [p for p in (self._title_number, self._title_name) if p]
        return " ".join(p.strip() for p in parts)

    @property
    def html(self) -> str:
        return "".join(self._capture)


def fetch(url: str, retries: int = 3, delay: float = 0.5) -> str:
    """HTTP GET with a UA header and a couple of retries."""
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError) as e:
            last_err = e
            time.sleep(delay * (attempt + 1))
    raise RuntimeError(f"Failed to fetch {url}: {last_err}")


def html_to_text(html: str) -> str:
    """Strip remaining tags, decode entities, normalize whitespace."""
    # Drop <sup> wrappers but keep their text (footnote numbers).
    html = re.sub(r"<\s*sup[^>]*>(.*?)</\s*sup\s*>", r"\1", html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    text = (
        text.replace("&nbsp;", " ")
            .replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", '"')
            .replace("&#39;", "'")
    )
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # tidy spacing around punctuation
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # add newlines between numbered paragraphs ("1. ... 2. ... 3. ...")
    text = re.sub(r"(?<!^)(?=\b\d+\.\s+[A-Z])", "\n", text)
    return text.strip()


def parse_article(html: str) -> dict:
    """Return {title, full_article} for a single article HTML page."""
    parser = ArticleTextExtractor()
    parser.feed(html)
    return {
        "title": parser.title,
        "full_article": html_to_text(parser.html),
    }


# ---------------------------------------------------------------------------
# LLM summary
# ---------------------------------------------------------------------------

def summarise_with_ollama(article_label: str, full_text: str, model: str = DEFAULT_MODEL,
                          timeout: int = 120) -> str:
    """Ask the local Ollama model for a 1-2 sentence summary of the article."""
    # Trim to keep the prompt small; articles are short but may include
    # boilerplate.
    text = full_text[:3000]
    prompt = (
        "Summarise the following GDPR article in EXACTLY one sentence, "
        "between 15 and 35 words, written in plain English. "
        "Focus on the core right or obligation it establishes. "
        "No headings, no quotes, no preamble.\n\n"
        f"Article label: {article_label}\n"
        f"Article text:\n{text}\n\n"
        "One-sentence summary:"
    )
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_predict": 80,
        },
        "messages": [
            {"role": "system", "content": "You write single-sentence legal summaries."},
            {"role": "user", "content": prompt},
        ],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read().decode("utf-8"))
    summary = out["message"]["content"].strip()
    # strip stray quotation marks / leading "Summary:" tokens
    summary = re.sub(r"^Summary:\s*", "", summary, flags=re.I)
    summary = summary.strip().strip('"').strip()
    # strip trailing " (N words)" annotations the model sometimes adds
    summary = re.sub(r"\s*\(\s*\d+\s*words?\s*\)\s*\.?\s*$", "", summary, flags=re.I)
    # collapse to a single line
    summary = re.sub(r"\s+", " ", summary)
    return summary


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

ARTICLE_URL = "https://gdpr-info.eu/art-{n}-gdpr/"
ARTICLE_RE = re.compile(r"art-(\d+)-gdpr", re.I)


def find_article_links(index_html: str):
    """Return the list of (n, title) for all article menu items."""
    results = []
    for m in re.finditer(
        r'<a href="https://gdpr-info\.eu/art-(\d+)-gdpr/[^"]*">'
        r'\s*Art\.\s*(\d+)\s*'
        r'<div class="menu-item-description">([^<]+)</div>',
        index_html,
    ):
        n = int(m.group(1))
        title = m.group(3).strip()
        results.append((n, title))
    return results


def scrape_articles(nums, model: str = DEFAULT_MODEL, progress: bool = True):
    out = []
    for n in nums:
        url = ARTICLE_URL.format(n=n)
        if progress:
            print(f"  • fetching Art. {n} …", end="", flush=True)
        html = fetch(url)
        parsed = parse_article(html)
        if not parsed["full_article"]:
            print("   (skipped — no content)")
            continue
        # Prefer the site-supplied menu title for the canonical name; fall back
        # to scraped <h1>.
        try:
            summary = summarise_with_ollama(f"Art. {n}", parsed["full_article"], model=model)
        except Exception as e:
            print(f"   LLM error: {e}")
            summary = parsed["title"] or "(summary unavailable)"
        out.append({
            "article": f"Art. {n}",
            "summary": summary,
            "full_article": parsed["full_article"],
        })
        if progress:
            print(f"  ✓  {summary[:60]}{'…' if len(summary) > 60 else ''}")
        # be polite to the upstream site
        time.sleep(0.4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="from_n", type=int, default=20)
    p.add_argument("--to",   dest="to_n",   type=int, default=25)
    p.add_argument("--all",  action="store_true",
                   help="Scrape every article (1-99).")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--append", action="store_true",
                   help="Merge into the existing JSON file instead of replacing it.")
    args = p.parse_args()

    nums = list(range(1, 100)) if args.all else list(range(args.from_n, args.to_n + 1))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing data if appending
    existing = []
    if args.append and out_path.exists():
        try:
            with out_path.open() as f:
                doc = json.load(f)
            existing = doc.get("GDPR", [])
            have = {a["article"] for a in existing}
            nums = [n for n in nums if f"Art. {n}" not in have]
            print(f"Appending to existing file with {len(existing)} articles; "
                  f"{len(nums)} new to fetch.")
        except Exception as e:
            print(f"Could not read existing file ({e}); starting fresh.")

    print(f"Scraping {len(nums)} GDPR articles with model {args.model} …")
    scraped = scrape_articles(nums, model=args.model)
    articles = existing + scraped

    with out_path.open("w") as f:
        json.dump({"GDPR": articles}, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(articles)} articles to {out_path}")


if __name__ == "__main__":
    main()
