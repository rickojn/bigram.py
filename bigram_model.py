#!/usr/bin/env python3
"""
Word-level bigram language model.

Usage:
    python bigram_model.py <text_file>

At the prompt:
    Enter           generate (using current limit)
    q               quit
    F1              change the token limit

After building the bigram table, an interactive HTML visualisation is written
to <text_file_stem>_bigram_table.html in the same directory as the script.
"""

import sys
import re
import math
import random
import os
import json


# ── Tokenisation ────────────────────────────────────────────────────────────

EOS = "__eos__"

def tokenise(text: str) -> list[str]:
    text = text.lower()
    # One or more newlines become an EOS token
    text = re.sub(r'\n+', f' {EOS} ', text)
    # . ! ? are kept as individual tokens
    text = re.sub(r'([.!?])', r' \1 ', text)
    # Strip remaining punctuation except apostrophes, . ! ?
    text = re.sub(r"[^\w\s'.!?]", ' ', text)
    return text.split()


# ── Vocabulary ───────────────────────────────────────────────────────────────

def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    unique = [EOS] + sorted({t for t in tokens if t != EOS})
    token_to_idx = {t: i for i, t in enumerate(unique)}
    return token_to_idx, unique


# ── Bigram table ─────────────────────────────────────────────────────────────

def build_bigram_counts(tokens, token_to_idx):
    V = len(token_to_idx)
    counts = [[0] * V for _ in range(V)]
    for a, b in zip(tokens, tokens[1:]):
        i, j = token_to_idx[a], token_to_idx[b]
        counts[i][j] += 1
    return counts


def softmax_row(row):
    if not any(row):
        n = len(row)
        return [1.0 / n] * n
    logits = [float(v) if v > 0 else float('-inf') for v in row]
    max_val = max(v for v in logits if v != float('-inf'))
    exps = [math.exp(v - max_val) if v != float('-inf') else 0.0 for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def sample_next(probs, idx_to_token):
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return idx_to_token[i]
    return idx_to_token[-1]


# ── Generation ───────────────────────────────────────────────────────────────

def generate(prompt_tokens, max_new_tokens, token_to_idx, idx_to_token, bigram_counts):
    generated = []
    current = prompt_tokens[-1]
    for _ in range(max_new_tokens):
        idx = token_to_idx[current]
        probs = softmax_row(bigram_counts[idx])
        next_token = sample_next(probs, idx_to_token)
        if next_token == EOS:
            break
        generated.append(next_token)
        current = next_token
    return generated


# ── HTML export ───────────────────────────────────────────────────────────────

def export_html(path: str, idx_to_token: list[str], bigram_counts: list[list[int]]) -> str:
    """
    Write a self-contained interactive HTML bigram-table viewer.
    Returns the output file path.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                            f"{stem}_bigram_table.html")

    # Build a sparse representation: for each row only store non-zero entries
    # { rowIdx: [[colIdx, count], ...] } — keeps the JSON payload small
    sparse = {}
    for i, row in enumerate(bigram_counts):
        nonzero = [[j, c] for j, c in enumerate(row) if c > 0]
        if nonzero:
            sparse[i] = nonzero

    display_tokens = ["&lt;EOS&gt;" if t == EOS else t for t in idx_to_token]

    vocab_json   = json.dumps(display_tokens)
    sparse_json  = json.dumps(sparse)
    title        = stem.replace("_", " ").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — Bigram Table</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');

  :root {{
    --bg:        #0f0e11;
    --surface:   #1a1820;
    --border:    #2e2b38;
    --accent:    #c8a96e;
    --accent2:   #7c6fcd;
    --text:      #e8e4dc;
    --muted:     #6b6578;
    --cell-zero: #0f0e11;
    --cell-low:  #1e1a2e;
    --cell-mid:  #3d2f6b;
    --cell-high: #7c6fcd;
    --cell-max:  #c8a96e;
  }}

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'DM Mono', monospace;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}

  header {{
    padding: 2rem 2.5rem 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: baseline;
    gap: 1.5rem;
    flex-wrap: wrap;
  }}

  header h1 {{
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    color: var(--accent);
    letter-spacing: 0.02em;
    white-space: nowrap;
  }}

  .meta {{
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
  }}

  .controls {{
    padding: 1rem 2.5rem;
    display: flex;
    gap: 1rem;
    align-items: center;
    flex-wrap: wrap;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
  }}

  .controls label {{
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }}

  input[type=text] {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-family: 'DM Mono', monospace;
    font-size: 0.82rem;
    padding: 0.4rem 0.75rem;
    width: 18rem;
    outline: none;
    transition: border-color 0.2s;
  }}
  input[type=text]:focus {{ border-color: var(--accent2); }}

  /* Filter axis button group */
  .toggle-group {{
    display: flex;
    gap: 0.25rem;
  }}
  .toggle-group button {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--muted);
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    padding: 0.35rem 0.7rem;
    cursor: pointer;
    letter-spacing: 0.05em;
    transition: all 0.15s;
  }}
  .toggle-group button.active {{
    background: var(--accent2);
    border-color: var(--accent2);
    color: #fff;
  }}

  /* Counts / Probabilities pill toggle */
  .pill-toggle {{
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.72rem;
    color: var(--muted);
    letter-spacing: 0.05em;
  }}
  .pill-toggle span {{
    transition: color 0.2s;
  }}
  .pill-toggle span.active {{
    color: var(--text);
  }}
  .pill-switch {{
    position: relative;
    width: 2.6rem;
    height: 1.4rem;
    cursor: pointer;
  }}
  .pill-switch input {{
    opacity: 0;
    width: 0;
    height: 0;
    position: absolute;
  }}
  .pill-track {{
    position: absolute;
    inset: 0;
    background: var(--border);
    border-radius: 1rem;
    transition: background 0.2s;
  }}
  .pill-switch input:checked + .pill-track {{
    background: var(--accent2);
  }}
  .pill-thumb {{
    position: absolute;
    top: 0.2rem;
    left: 0.2rem;
    width: 1rem;
    height: 1rem;
    background: var(--text);
    border-radius: 50%;
    transition: transform 0.2s;
  }}
  .pill-switch input:checked ~ .pill-thumb {{
    transform: translateX(1.2rem);
  }}

  .hint {{
    font-size: 0.7rem;
    color: var(--muted);
    margin-left: auto;
  }}

  .table-wrap {{
    flex: 1;
    overflow: auto;
    padding: 1.5rem 2.5rem 2.5rem;
  }}

  table {{
    border-collapse: collapse;
    font-size: 0.7rem;
  }}

  th, td {{
    width: 2.1rem;
    height: 2.1rem;
    text-align: center;
    vertical-align: middle;
    border: 1px solid #1c1924;
  }}

  /* Row / column header cells */
  th.row-hdr, th.col-hdr {{
    background: var(--surface);
    color: var(--accent);
    font-weight: 500;
    position: sticky;
    z-index: 2;
    white-space: nowrap;
    padding: 0 0.5rem;
    max-width: 7rem;
    overflow: hidden;
    text-overflow: ellipsis;
  }}
  th.row-hdr {{
    left: 0;
    text-align: right;
    border-right: 2px solid var(--border);
  }}
  th.col-hdr {{
    top: 0;
    border-bottom: 2px solid var(--border);
    writing-mode: vertical-rl;
    transform: rotate(180deg);
    height: 6rem;
    width: 2.1rem;
    padding: 0.4rem 0;
  }}
  th.corner {{
    position: sticky;
    top: 0;
    left: 0;
    z-index: 3;
    background: var(--bg);
    border-right: 2px solid var(--border);
    border-bottom: 2px solid var(--border);
  }}

  /* Data cells — colour set by JS */
  td.cell {{
    cursor: default;
    transition: filter 0.1s;
    position: relative;
  }}
  td.cell:hover {{ filter: brightness(1.6); outline: 1px solid var(--accent); z-index: 1; }}
  td.cell.zero {{ background: var(--cell-zero); }}

  /* Highlight classes applied by JS */
  tr.row-hl th.row-hdr,
  tr.row-hl td {{ outline: none; }}
  tr.row-hl td {{ filter: brightness(1.25); }}
  .col-hl {{ filter: brightness(1.25) !important; }}

  /* Tooltip */
  #tip {{
    position: fixed;
    background: #2a2535;
    border: 1px solid var(--accent2);
    border-radius: 6px;
    padding: 0.5rem 0.8rem;
    font-size: 0.75rem;
    line-height: 1.6;
    pointer-events: none;
    z-index: 999;
    display: none;
    white-space: nowrap;
  }}
  #tip .t-pair  {{ color: var(--accent); font-weight: 500; }}
  #tip .t-count {{ color: var(--text); }}
  #tip .t-prob  {{ color: var(--accent2); }}

  /* Hidden rows when filtering */
  tr.hidden {{ display: none; }}
  col.hidden-col {{ visibility: collapse; }}

  .legend {{
    display: flex;
    gap: 0.5rem;
    align-items: center;
    font-size: 0.68rem;
    color: var(--muted);
    margin-left: 1rem;
  }}
  .legend-swatch {{
    width: 1rem;
    height: 1rem;
    border-radius: 2px;
    display: inline-block;
  }}
</style>
</head>
<body>

<header>
  <h1>{title}</h1>
  <span class="meta" id="stats-label"></span>
</header>

<div class="controls">
  <label>FILTER TOKEN</label>
  <input type="text" id="search" placeholder="type a word…" autocomplete="off" spellcheck="false">
  <div class="toggle-group">
    <button class="active" id="btn-both"   onclick="setMode('both')">Row &amp; Col</button>
    <button            id="btn-row"    onclick="setMode('row')">Row only</button>
    <button            id="btn-col"    onclick="setMode('col')">Col only</button>
  </div>
  <div class="pill-toggle">
    <span id="lbl-count" class="active">COUNTS</span>
    <label class="pill-switch">
      <input type="checkbox" id="display-toggle" onchange="setDisplay(this.checked ? 'prob' : 'count')">
      <div class="pill-track"></div>
      <div class="pill-thumb"></div>
    </label>
    <span id="lbl-prob">PROBS</span>
  </div>
  <div class="legend">
    <span class="legend-swatch" style="background:var(--cell-zero)"></span>0
    <span class="legend-swatch" style="background:var(--cell-low)"></span>low
    <span class="legend-swatch" style="background:var(--cell-mid)"></span>mid
    <span class="legend-swatch" style="background:var(--cell-high)"></span>high
    <span class="legend-swatch" style="background:var(--cell-max)"></span>max
  </div>
  <span class="hint" id="hint-label"></span>
</div>

<div class="table-wrap">
  <table id="tbl"></table>
</div>

<div id="tip"></div>

<script>
const VOCAB  = {vocab_json};
const SPARSE = {sparse_json};
const V = VOCAB.length;

// Build dense row sums for probability computation
const rowSums = new Float64Array(V);
for (const [ri, pairs] of Object.entries(SPARSE)) {{
  let s = 0;
  for (const [, c] of pairs) s += c;
  rowSums[ri] = s;
}}

// Find global max count for colour scaling
let globalMax = 0;
for (const pairs of Object.values(SPARSE))
  for (const [, c] of pairs)
    if (c > globalMax) globalMax = c;

function countToColour(c) {{
  if (c === 0) return null;
  const t = c / globalMax;
  if (t < 0.25) return `color-mix(in srgb, var(--cell-low) ${{Math.round(t*400)}}%, var(--cell-zero))`;
  if (t < 0.5)  return `color-mix(in srgb, var(--cell-mid) ${{Math.round((t-0.25)*400)}}%, var(--cell-low))`;
  if (t < 0.75) return `color-mix(in srgb, var(--cell-high) ${{Math.round((t-0.5)*400)}}%, var(--cell-mid))`;
  return `color-mix(in srgb, var(--cell-max) ${{Math.round((t-0.75)*400)}}%, var(--cell-high))`;
}}

// Build lookup: sparse[ri][ci] = count
const lookup = [];
for (let i = 0; i < V; i++) lookup.push({{}});
for (const [ri, pairs] of Object.entries(SPARSE))
  for (const [ci, c] of pairs)
    lookup[ri][ci] = c;

// ── Build table DOM ────────────────────────────────────────────────────────
const tbl = document.getElementById('tbl');
const tip = document.getElementById('tip');
let displayMode = 'count';  // 'count' | 'prob'
let filterMode  = 'both';   // 'both' | 'row' | 'col'

// colgroup for column visibility
const cg = document.createElement('colgroup');
const cornerCol = document.createElement('col');
cg.appendChild(cornerCol);
const colEls = [];
for (let ci = 0; ci < V; ci++) {{
  const col = document.createElement('col');
  col.id = `col-${{ci}}`;
  cg.appendChild(col);
  colEls.push(col);
}}
tbl.appendChild(cg);

// Header row
const thead = tbl.createTHead();
const hdrRow = thead.insertRow();
const corner = document.createElement('th');
corner.className = 'corner';
corner.textContent = 'row ↓  col →';
hdrRow.appendChild(corner);
for (let ci = 0; ci < V; ci++) {{
  const th = document.createElement('th');
  th.className = 'col-hdr';
  th.textContent = VOCAB[ci];
  th.title = VOCAB[ci];
  hdrRow.appendChild(th);
}}

// Body rows — one per vocabulary token
const tbody = tbl.createTBody();
const rows   = [];  // tr elements
const cells  = [];  // cells[ri][ci] = td

for (let ri = 0; ri < V; ri++) {{
  const tr = tbody.insertRow();
  tr.id = `row-${{ri}}`;
  rows.push(tr);
  cells.push([]);

  const rh = document.createElement('th');
  rh.className = 'row-hdr';
  rh.textContent = VOCAB[ri];
  rh.title = VOCAB[ri];
  tr.appendChild(rh);

  const rowPairs = SPARSE[ri] || [];
  // Build per-row count map for fast lookup
  const rowMap = {{}};
  for (const [ci, c] of rowPairs) rowMap[ci] = c;

  for (let ci = 0; ci < V; ci++) {{
    const td = document.createElement('td');
    td.className = 'cell';
    const c = rowMap[ci] || 0;
    td.dataset.count = c;
    td.dataset.ri = ri;
    td.dataset.ci = ci;
    if (c === 0) {{
      td.classList.add('zero');
    }} else {{
      td.style.background = countToColour(c);
      td.textContent = c;
    }}
    td.addEventListener('mouseenter', onCellEnter);
    td.addEventListener('mouseleave', onCellLeave);
    tr.appendChild(td);
    cells[ri].push(td);
  }}
}}

document.getElementById('stats-label').textContent =
  `${{V}} tokens · ${{Object.values(SPARSE).reduce((s,p)=>s+p.length,0).toLocaleString()}} attested bigrams`;

// ── Tooltip ────────────────────────────────────────────────────────────────
function onCellEnter(e) {{
  const td = e.currentTarget;
  const ri = +td.dataset.ri, ci = +td.dataset.ci, c = +td.dataset.count;
  const prob = rowSums[ri] > 0 ? (c / rowSums[ri]) : 0;
  tip.innerHTML =
    `<div class="t-pair">"${{VOCAB[ri]}}" → "${{VOCAB[ci]}}"</div>` +
    `<div class="t-count">count: ${{c}}</div>` +
    `<div class="t-prob">P(col|row): ${{prob.toFixed(4)}}</div>`;
  tip.style.display = 'block';
  document.addEventListener('mousemove', moveTip);
}}
function onCellLeave() {{
  tip.style.display = 'none';
  document.removeEventListener('mousemove', moveTip);
}}
function moveTip(e) {{
  tip.style.left = (e.clientX + 14) + 'px';
  tip.style.top  = (e.clientY + 14) + 'px';
}}

// ── Display mode (count vs prob) ───────────────────────────────────────────
function setDisplay(mode) {{
  displayMode = mode;
  document.getElementById('lbl-count').classList.toggle('active', mode === 'count');
  document.getElementById('lbl-prob').classList.toggle('active',  mode === 'prob');
  // Update cell text for currently visible non-zero cells
  for (let ri = 0; ri < V; ri++) {{
    for (let ci = 0; ci < V; ci++) {{
      const td = cells[ri][ci];
      const c = +td.dataset.count;
      if (c > 0) {{
        if (mode === 'prob') {{
          const p = rowSums[ri] > 0 ? c / rowSums[ri] : 0;
          td.textContent = p > 0 ? p.toFixed(2) : '';
        }} else {{
          td.textContent = c;
        }}
      }}
    }}
  }}
}}

// ── Filter ─────────────────────────────────────────────────────────────────
function setMode(m) {{
  filterMode = m;
  ['both','row','col'].forEach(x =>
    document.getElementById('btn-'+x).classList.toggle('active', x === m));
  applyFilter(document.getElementById('search').value.trim().toLowerCase());
}}

function applyFilter(q) {{
  if (!q) {{
    // Show everything
    for (const tr of rows) tr.classList.remove('hidden');
    for (const col of colEls) col.classList.remove('hidden-col');
    document.getElementById('hint-label').textContent = '';
    return;
  }}

  const matchedRows = [], matchedCols = [];
  for (let i = 0; i < V; i++) {{
    if (VOCAB[i].includes(q)) {{ matchedRows.push(i); matchedCols.push(i); }}
  }}

  const showRow = new Set(filterMode !== 'col' ? matchedRows : Array.from({{length:V}},(_,i)=>i));
  const showCol = new Set(filterMode !== 'row' ? matchedCols : Array.from({{length:V}},(_,i)=>i));

  for (let ri = 0; ri < V; ri++)
    rows[ri].classList.toggle('hidden', !showRow.has(ri));
  for (let ci = 0; ci < V; ci++)
    colEls[ci].classList.toggle('hidden-col', !showCol.has(ci));

  const total = matchedRows.length;
  document.getElementById('hint-label').textContent =
    total === 0 ? 'no matches' : `${{total}} match${{total===1?'':'es'}}`;
}}

document.getElementById('search').addEventListener('input', e =>
  applyFilter(e.target.value.trim().toLowerCase()));
</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)

    return out_path


# ── CLI helpers ───────────────────────────────────────────────────────────────

def validate_prompt(raw, token_to_idx):
    tokens = tokenise(raw)
    if not tokens:
        print("  x Prompt is empty after tokenisation.")
        return None
    unknown = [t for t in tokens if t not in token_to_idx]
    if unknown:
        print(f"  x Unknown token(s): {unknown}")
        return None
    return tokens


def pretty_print(prompt_tokens, generated):
    prompt_str = " ".join(prompt_tokens)
    gen_str = " ".join("<EOS>" if t == EOS else t for t in generated)
    full = prompt_tokens + generated
    readable = " ".join(t for t in full)
    print("\n" + "-" * 60)
    print(f"PROMPT   : {prompt_str}")
    print(f"GENERATED: {gen_str}")
    print(f"FULL TEXT: {readable}")
    print("-" * 60 + "\n")


NO_LIMIT = 10_000
F1 = '\x1bOP'


def ask_token_limit():
    while True:
        raw = input("Max tokens to generate (Enter = EOS only, 'q' to quit): ").strip()
        if raw.lower() == 'q':
            print("Goodbye.")
            sys.exit(0)
        if raw == '':
            print("  -> No token limit; generation stops at EOS.")
            return NO_LIMIT
        if raw.isdigit() and int(raw) > 0:
            print(f"  -> Token limit set to {int(raw)}.")
            return int(raw)
        print("  x Please enter a positive integer, or press Enter for no limit.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python bigram_model.py <text_file>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        with open(path, encoding="utf-8") as fh:
            raw_text = fh.read()
    except FileNotFoundError:
        print(f"Error: file '{path}' not found.")
        sys.exit(1)

    print(f"\nReading '{path}' ...")
    tokens = tokenise(raw_text)
    print(f"  Total tokens   : {len(tokens):,}")

    token_to_idx, idx_to_token = build_vocab(tokens)
    V = len(token_to_idx)
    print(f"  Vocabulary size: {V:,}  (including EOS)")

    bigram_counts = build_bigram_counts(tokens, token_to_idx)
    print("  Bigram table built.")

    # ── Export HTML visualisation ────────────────────────────────────────────
    html_path = export_html(path, idx_to_token, bigram_counts)
    print(f"  Visualisation  : {html_path}\n")

    # ── Ask for token limit once up front ────────────────────────────────────
    max_new = ask_token_limit()
    print("  (Press F1 at the prompt to change the token limit.)\n")

    def limit_label():
        return "EOS only" if max_new == NO_LIMIT else str(max_new)

    # ── Generation loop ───────────────────────────────────────────────────────
    while True:
        raw_prompt = input(f"Prompt [limit={limit_label()}] ('q' to quit): ").strip()

        if raw_prompt.lower() == 'q':
            print("Goodbye.")
            return

        if raw_prompt == F1:
            max_new = ask_token_limit()
            continue

        if not raw_prompt:
            print("  x Prompt cannot be empty. (F1 to change limit, 'q' to quit.)")
            continue

        prompt_tokens = validate_prompt(raw_prompt, token_to_idx)
        if prompt_tokens is None:
            print(f"  Hint: {random.sample(idx_to_token[1:min(20, V)], min(8, V - 1))}")
            continue

        generated = generate(prompt_tokens, max_new, token_to_idx, idx_to_token, bigram_counts)
        pretty_print(prompt_tokens, generated)


if __name__ == "__main__":
    main()
