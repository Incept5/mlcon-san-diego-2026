#!/usr/bin/env python3
"""
GDPR Article Viewer — professional single-page web app.
Serves gdpr_articles.json with search, filtering, and article detail view.

Usage:
    python gdpr_viewer.py           # serves on port 7003
    python gdpr_viewer.py 8080      # custom port
"""

import json
import os
import sys
import threading
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 7003
DATA_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gdpr_articles.json")

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GDPR Article Explorer</title>
<style>
  :root {
    --primary: #1a237e;
    --primary-light: #3949ab;
    --accent: #ffc107;
    --bg: #f5f5f5;
    --card-bg: #ffffff;
    --text: #212121;
    --text-secondary: #616161;
    --border: #e0e0e0;
    --shadow: 0 2px 8px rgba(0,0,0,0.08);
    --radius: 8px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
  }

  /* Header */
  .top-banner {
    background: #0d47a1;
    color: #fff;
    text-align: center;
    padding: 0.6rem 1rem;
    font-size: 1.4rem;
    font-weight: 800;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    border-bottom: 3px solid var(--accent);
  }
  .header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: #fff;
    padding: 1.5rem 2rem 1.5rem;
    position: sticky;
    top: 0;
    z-index: 100;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }
  .header-inner {
    max-width: 1200px;
    margin: 0 auto;
  }
  .header h1 {
    font-size: 1.75rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
  }
  .header p {
    opacity: 0.85;
    font-size: 0.95rem;
  }
  .search-bar {
    margin-top: 1rem;
    position: relative;
    max-width: 500px;
  }
  .search-bar input {
    width: 100%;
    padding: 0.7rem 1rem 0.7rem 2.5rem;
    border: none;
    border-radius: var(--radius);
    font-size: 0.95rem;
    background: rgba(255,255,255,0.95);
    color: var(--text);
    outline: none;
    transition: box-shadow 0.2s;
  }
  .search-bar input:focus {
    box-shadow: 0 0 0 3px rgba(255,193,7,0.4);
  }
  .search-bar::before {
    content: '🔍';
    position: absolute;
    left: 0.75rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 1rem;
  }
  .stats {
    margin-top: 0.75rem;
    font-size: 0.8rem;
    opacity: 0.7;
  }

  /* Layout */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 1.5rem;
  }
  .grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
  }
  @media (max-width: 768px) {
    .grid { grid-template-columns: 1fr; }
  }

  /* Article List */
  .list-panel {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
  }
  .article-card {
    background: var(--card-bg);
    border-radius: var(--radius);
    padding: 1.25rem;
    cursor: pointer;
    transition: all 0.2s;
    border: 2px solid transparent;
    box-shadow: var(--shadow);
  }
  .article-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 16px rgba(0,0,0,0.12);
  }
  .article-card.active {
    border-color: var(--accent);
    box-shadow: 0 4px 16px rgba(255,193,7,0.3);
  }
  .article-card .art-badge {
    display: inline-block;
    background: var(--primary);
    color: #fff;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
  }
  .article-card .summary {
    font-size: 0.9rem;
    color: var(--text-secondary);
    line-height: 1.5;
  }

  /* Detail Panel */
  .detail-panel {
    background: var(--card-bg);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    position: sticky;
    top: 140px;
    max-height: calc(100vh - 160px);
    overflow-y: auto;
  }
  .detail-empty {
    padding: 4rem 2rem;
    text-align: center;
    color: var(--text-secondary);
  }
  .detail-empty .icon { font-size: 3rem; margin-bottom: 1rem; }
  .detail-header {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
    color: #fff;
    padding: 1.5rem;
    border-radius: var(--radius) var(--radius) 0 0;
  }
  .detail-header h2 {
    font-size: 1.3rem;
    font-weight: 700;
  }
  .detail-header .art-num {
    display: inline-block;
    background: var(--accent);
    color: var(--primary);
    font-weight: 700;
    font-size: 0.8rem;
    padding: 0.2rem 0.6rem;
    border-radius: 4px;
    margin-bottom: 0.5rem;
  }
  .detail-body {
    padding: 1.5rem;
  }
  .detail-body h3 {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-secondary);
    margin-bottom: 0.75rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 0.5rem;
  }
  .summary-text {
    background: #fffde7;
    border-left: 4px solid var(--accent);
    padding: 1rem;
    border-radius: 0 var(--radius) var(--radius) 0;
    margin-bottom: 1.5rem;
    font-size: 0.95rem;
    line-height: 1.6;
  }
  .full-text {
    font-size: 0.95rem;
    line-height: 1.75;
    white-space: pre-wrap;
    color: var(--text);
  }

  /* Scrollbar */
  .detail-panel::-webkit-scrollbar { width: 6px; }
  .detail-panel::-webkit-scrollbar-track { background: transparent; }
  .detail-panel::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
  }
</style>
</head>
<body>

<div class="top-banner">🚀 Kimi K2.6</div>

<div class="header">
  <div class="header-inner">
    <h1>📜 GDPR Article Explorer</h1>
    <p>Browse, search, and read articles of the General Data Protection Regulation</p>
    <div class="search-bar">
      <input type="text" id="search" placeholder="Search articles by keyword..." autocomplete="off">
    </div>
    <div class="stats" id="stats"></div>
  </div>
</div>

<div class="container">
  <div class="grid">
    <div class="list-panel" id="articleList"></div>
    <div class="detail-panel" id="detailPanel">
      <div class="detail-empty">
        <div class="icon">📋</div>
        <p>Select an article from the list to view its full text</p>
      </div>
    </div>
  </div>
</div>

<script>
const ARTICLES = __ARTICLES__;
let activeIdx = null;

function renderList(articles) {
  const list = document.getElementById('articleList');
  list.innerHTML = articles.map((a) => `
    <div class="article-card ${activeIdx === a.article ? 'active' : ''}"
         onclick="selectArticle('${a.article}')">
      <div class="art-badge">${a.article}</div>
      <div class="summary">${escHtml(a.summary)}</div>
    </div>
  `).join('');
  document.getElementById('stats').textContent =
    `Showing ${articles.length} of ${ARTICLES.length} articles`;
}

function selectArticle(artNum) {
  activeIdx = artNum;
  const art = ARTICLES.find(a => a.article === artNum);
  if (!art) return;
  const panel = document.getElementById('detailPanel');
  panel.innerHTML = `
    <div class="detail-header">
      <div class="art-num">${art.article}</div>
      <h2>${escHtml(art.article)}</h2>
    </div>
    <div class="detail-body">
      <h3>📋 AI-Generated Summary</h3>
      <div class="summary-text">${escHtml(art.summary)}</div>
      <h3>📄 Full Article Text</h3>
      <div class="full-text">${escHtml(art.full_article)}</div>
    </div>
  `;
  renderList(filteredArticles());
}

function filteredArticles() {
  const q = document.getElementById('search').value.toLowerCase().trim();
  if (!q) return ARTICLES;
  return ARTICLES.filter(a =>
    (a.summary || '').toLowerCase().includes(q) ||
    (a.full_article || '').toLowerCase().includes(q) ||
    (a.article || '').toLowerCase().includes(q)
  );
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s || '';
  return d.innerHTML;
}

document.getElementById('search').addEventListener('input', () => {
  renderList(filteredArticles());
});

renderList(ARTICLES);
</script>
</body>
</html>"""


class GDPRHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler — serves the SPA with articles injected."""

    def do_GET(self):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            articles = data.get("GDPR", [])
        except FileNotFoundError:
            articles = []

        html = HTML_TEMPLATE.replace("__ARTICLES__", json.dumps(articles, ensure_ascii=False))

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def log_message(self, fmt, *args):
        print(f"  {args[0]}" if args else "")


def main():
    server = HTTPServer(("0.0.0.0", PORT), GDPRHandler)
    url = f"http://localhost:{PORT}"
    print(f"\n📜 GDPR Article Explorer running at {url}")
    print(f"   Loaded data from: {DATA_FILE}")
    print(f"   Press Ctrl+C to stop\n")

    threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
