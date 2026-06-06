"""GDPR Explorer — browse GDPR articles by AI-generated summary, read the full text.

Reads gdpr.json (produced by scrape_gdpr.py) and serves a single-page UI on port 7005.

    python server.py
    -> http://localhost:7005
"""

import json
import os

from flask import Flask, jsonify, render_template_string

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(HERE, "gdpr.json")
PORT = 7005

app = Flask(__name__)


def load_data():
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


@app.route("/api/articles")
def api_articles():
    return jsonify(load_data())


@app.route("/")
def index():
    data = load_data()
    count = len(data.get("GDPR", []))
    return render_template_string(PAGE, count=count)


PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GDPR Explorer</title>
<style>
  :root {
    --bg: #0f1320;
    --panel: #171c2e;
    --panel-2: #1e2640;
    --border: #2a3354;
    --text: #e8ecf5;
    --muted: #97a0bd;
    --accent: #4f7cff;
    --accent-soft: rgba(79,124,255,.14);
    --gold: #f2c14e;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  .megabar {
    text-align: center;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -.5px;
    padding: 22px 20px;
    color: #fff;
    background: linear-gradient(90deg, #4f7cff, #7b4fff 55%, #f2c14e);
    border-bottom: 1px solid var(--border);
    text-shadow: 0 1px 12px rgba(0,0,0,.25);
  }
  @media (max-width: 880px) { .megabar { font-size: 30px; padding: 16px; } }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    background: radial-gradient(1200px 600px at 80% -10%, #1b2b4d 0, transparent 60%),
                radial-gradient(900px 500px at -10% 10%, #16203f 0, transparent 55%), var(--bg);
    color: var(--text);
    -webkit-font-smoothing: antialiased;
  }
  header {
    padding: 28px 36px 22px;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, rgba(79,124,255,.10), transparent);
  }
  header .brand { display: flex; align-items: center; gap: 14px; }
  .badge {
    font-size: 13px; font-weight: 700; letter-spacing: .5px;
    color: #0f1320; background: var(--gold);
    padding: 4px 10px; border-radius: 6px;
  }
  header h1 { margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -.3px; }
  header p { margin: 6px 0 0; color: var(--muted); font-size: 14px; }
  .layout {
    display: grid;
    grid-template-columns: 420px 1fr;
    gap: 0;
    height: calc(100vh - 195px);
  }
  .list-pane {
    border-right: 1px solid var(--border);
    overflow-y: auto;
    background: var(--panel);
  }
  .search-wrap { position: sticky; top: 0; padding: 16px; background: var(--panel); border-bottom: 1px solid var(--border); z-index: 2; }
  .search-wrap input {
    width: 100%; padding: 11px 14px; border-radius: 10px;
    border: 1px solid var(--border); background: var(--bg); color: var(--text);
    font-size: 14px; outline: none;
  }
  .search-wrap input:focus { border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  .card {
    padding: 16px 18px; border-bottom: 1px solid var(--border);
    cursor: pointer; transition: background .12s ease;
  }
  .card:hover { background: var(--panel-2); }
  .card.active { background: var(--accent-soft); border-left: 3px solid var(--accent); padding-left: 15px; }
  .card .art { font-size: 12px; font-weight: 700; color: var(--accent); letter-spacing: .4px; }
  .card .title { font-size: 14.5px; font-weight: 600; margin: 3px 0 6px; }
  .card .sum { font-size: 13px; color: var(--muted); line-height: 1.5;
    display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; }
  .detail-pane { overflow-y: auto; padding: 40px 56px 80px; }
  .detail-empty { color: var(--muted); font-size: 15px; margin-top: 80px; text-align: center; }
  .detail .art { font-size: 14px; font-weight: 700; color: var(--accent); letter-spacing: .5px; }
  .detail h2 { font-size: 30px; margin: 8px 0 22px; font-weight: 700; letter-spacing: -.4px; }
  .summary-box {
    background: var(--panel); border: 1px solid var(--border); border-left: 3px solid var(--gold);
    border-radius: 10px; padding: 18px 20px; margin-bottom: 30px;
  }
  .summary-box .label { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--gold); margin-bottom: 8px; }
  .summary-box p { margin: 0; font-size: 16px; line-height: 1.6; }
  .full-label { font-size: 11px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: var(--muted); margin-bottom: 12px; }
  .full-text { white-space: pre-wrap; font-size: 15px; line-height: 1.7; color: #dbe1f0;
    background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: 22px 24px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
  .meta { color: var(--muted); font-size: 13px; }
  .count-pill { font-size: 12px; color: var(--muted); padding: 8px 16px; }
  ::-webkit-scrollbar { width: 10px; }
  ::-webkit-scrollbar-thumb { background: #2c3658; border-radius: 8px; }
  @media (max-width: 880px) {
    .layout { grid-template-columns: 1fr; height: auto; }
    .list-pane { border-right: none; border-bottom: 1px solid var(--border); max-height: 45vh; }
  }
</style>
</head>
<body>
<div class="megabar">Claude Opus 4.8</div>
<header>
  <div class="brand">
    <span class="badge">GDPR</span>
    <h1>GDPR Explorer</h1>
  </div>
  <p>Browse articles by plain-English summary — generated locally with qwen3.5:4b on Ollama. Select a summary to read the full legal text. &nbsp;<span class="meta">{{ count }} article(s) loaded.</span></p>
</header>

<div class="layout">
  <div class="list-pane">
    <div class="search-wrap">
      <input id="search" type="text" placeholder="Search summaries, titles, article numbers…" autocomplete="off">
    </div>
    <div class="count-pill" id="countPill"></div>
    <div id="list"></div>
  </div>
  <div class="detail-pane">
    <div id="detail"><div class="detail-empty">← Select an article from the list to view its summary and full text.</div></div>
  </div>
</div>

<script>
let ARTICLES = [];
let selected = null;

function esc(s){ const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }

function renderList(filter){
  const q = (filter||'').trim().toLowerCase();
  const list = document.getElementById('list');
  const shown = ARTICLES.filter(a => !q ||
      a.summary.toLowerCase().includes(q) ||
      (a.title||'').toLowerCase().includes(q) ||
      a.article.toLowerCase().includes(q));
  document.getElementById('countPill').textContent =
      shown.length + (q ? ' match'+(shown.length===1?'':'es') : ' article'+(shown.length===1?'':'s'));
  list.innerHTML = shown.map((a) => {
    const i = ARTICLES.indexOf(a);
    const active = (selected===i) ? ' active' : '';
    return `<div class="card${active}" data-i="${i}">
      <div class="art">${esc(a.article)}</div>
      <div class="title">${esc(a.title||'')}</div>
      <div class="sum">${esc(a.summary)}</div>
    </div>`;
  }).join('') || '<div class="count-pill">No matches.</div>';
  list.querySelectorAll('.card').forEach(c =>
    c.addEventListener('click', () => select(parseInt(c.dataset.i))));
}

function select(i){
  selected = i;
  const a = ARTICLES[i];
  document.getElementById('detail').innerHTML = `
    <div class="detail">
      <div class="art">${esc(a.article)}</div>
      <h2>${esc(a.title||'')}</h2>
      <div class="summary-box">
        <div class="label">AI Summary</div>
        <p>${esc(a.summary)}</p>
      </div>
      <div class="full-label">Full Article Text</div>
      <div class="full-text">${esc(a.full_article)}</div>
    </div>`;
  renderList(document.getElementById('search').value);
  document.querySelector('.detail-pane').scrollTop = 0;
}

fetch('/api/articles').then(r=>r.json()).then(d => {
  ARTICLES = d.GDPR || [];
  renderList('');
});
document.getElementById('search').addEventListener('input', e => renderList(e.target.value));
</script>
</body>
</html>
"""


if __name__ == "__main__":
    print(f"GDPR Explorer running at http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
