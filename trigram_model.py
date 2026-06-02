#!/usr/bin/env python3
"""
Word-level trigram language model with stupid backoff.

Usage:
    python trigram_model.py <text_file>

Generation uses the trigram distribution P(w | w-2, w-1) when the context
has been seen, backing off to the bigram P(w | w-1) scaled by alpha=0.4,
and further to a uniform distribution if the unigram context is also unseen.

At the prompt:
    Enter           generate (using current limit)
    q               quit
    F1              change the token limit

A prompt of one word is accepted; generation begins in bigram-backoff mode
until a second context token is available.

After building the tables, an interactive HTML visualisation of the bigram
table is written to <text_file_stem>_bigram_table.html in the same directory.
"""

import sys
import re
import math
import random
import os
import json
from collections import defaultdict


# ── Constants ─────────────────────────────────────────────────────────────────

EOS        = "__eos__"
ALPHA      = 0.4          # stupid-backoff discount
NO_LIMIT   = 10_000       # sentinel: EOS-only termination
F1         = '\x1bOP'     # F1 escape sequence (most terminals)


# ── Tokenisation ─────────────────────────────────────────────────────────────

def tokenise(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r'\n+', f' {EOS} ', text)
    text = re.sub(r'([.!?])', r' \1 ', text)
    text = re.sub(r"[^\w\s'.!?]", ' ', text)
    return text.split()


# ── Vocabulary ────────────────────────────────────────────────────────────────

def build_vocab(tokens: list[str]) -> tuple[dict[str, int], list[str]]:
    unique = [EOS] + sorted({t for t in tokens if t != EOS})
    token_to_idx = {t: i for i, t in enumerate(unique)}
    return token_to_idx, unique


# ── Count tables ──────────────────────────────────────────────────────────────

def build_bigram_counts(tokens: list[str],
                        token_to_idx: dict[str, int]) -> list[list[int]]:
    """Dense V×V bigram count matrix."""
    V = len(token_to_idx)
    counts = [[0] * V for _ in range(V)]
    for a, b in zip(tokens, tokens[1:]):
        counts[token_to_idx[a]][token_to_idx[b]] += 1
    return counts


def build_trigram_counts(tokens: list[str],
                         token_to_idx: dict[str, int]) -> dict:
    """
    Sparse trigram counts.
    trigram_counts[(i, j)][k] = number of times token_k followed (token_i, token_j).
    Uses defaultdict so unseen contexts return an empty dict naturally.
    """
    counts: dict[tuple[int, int], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for a, b, c in zip(tokens, tokens[1:], tokens[2:]):
        i = token_to_idx[a]
        j = token_to_idx[b]
        k = token_to_idx[c]
        counts[(i, j)][k] += 1
    return counts


# ── Probability / sampling ────────────────────────────────────────────────────

def softmax_row(row: list[int]) -> list[float]:
    """
    Softmax over a bigram count row.
    Zero entries are masked so only attested transitions are sampled.
    Falls back to uniform if all counts are zero.
    """
    if not any(row):
        n = len(row)
        return [1.0 / n] * n
    logits = [float(v) if v > 0 else float('-inf') for v in row]
    max_val = max(v for v in logits if v != float('-inf'))
    exps = [math.exp(v - max_val) if v != float('-inf') else 0.0 for v in logits]
    total = sum(exps)
    return [e / total for e in exps]


def backoff_probs(prev_idx: int | None,
                  cur_idx: int,
                  V: int,
                  bigram_counts: list[list[int]],
                  trigram_counts: dict) -> list[float]:
    """
    Stupid-backoff probability vector over the full vocabulary.

    If prev_idx is not None and the trigram context (prev, cur) has been seen,
    use normalised trigram counts.  Otherwise scale bigram probs by ALPHA.
    If the bigram context is also unseen, fall back to uniform scaled by ALPHA².
    """
    # ── Try trigram ──────────────────────────────────────────────────────────
    if prev_idx is not None:
        tri_row = trigram_counts.get((prev_idx, cur_idx))
        if tri_row:
            total = sum(tri_row.values())
            probs = [0.0] * V
            for k, c in tri_row.items():
                probs[k] = c / total
            return probs

    # ── Back off to bigram ───────────────────────────────────────────────────
    bi_row = bigram_counts[cur_idx]
    if any(bi_row):
        bi_probs = softmax_row(bi_row)
        scale = ALPHA if prev_idx is not None else 1.0
        return [p * scale for p in bi_probs]

    # ── Back off to uniform ──────────────────────────────────────────────────
    scale = (ALPHA ** 2) if prev_idx is not None else ALPHA
    return [scale / V] * V


def sample_next(probs: list[float], idx_to_token: list[str]) -> str:
    r = random.random()
    cumulative = 0.0
    for i, p in enumerate(probs):
        cumulative += p
        if r < cumulative:
            return idx_to_token[i]
    return idx_to_token[-1]


# ── Generation ────────────────────────────────────────────────────────────────

def generate(prompt_tokens: list[str],
             max_new_tokens: int,
             token_to_idx: dict[str, int],
             idx_to_token: list[str],
             bigram_counts: list[list[int]],
             trigram_counts: dict) -> list[str]:
    """
    Generate up to max_new_tokens tokens, stopping early on EOS.
    Uses a rolling two-token context; backs off when the context is unseen.
    """
    V = len(idx_to_token)
    generated: list[str] = []

    # Seed context from the prompt
    prev_idx = token_to_idx[prompt_tokens[-2]] if len(prompt_tokens) >= 2 else None
    cur_idx  = token_to_idx[prompt_tokens[-1]]

    for _ in range(max_new_tokens):
        probs      = backoff_probs(prev_idx, cur_idx, V, bigram_counts, trigram_counts)
        next_token = sample_next(probs, idx_to_token)
        generated.append(next_token)
        if next_token == EOS:
            break
        prev_idx = cur_idx
        cur_idx  = token_to_idx[next_token]

    return generated


# ── HTML export (bigram table) ─────────────────────────────────────────────────

def export_html(path: str, idx_to_token: list[str], bigram_counts: list[list[int]], trigram_counts: dict) -> str:
    stem     = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(os.path.dirname(os.path.abspath(path)),
                            f"{stem}_ngram_table.html")

    # Bigram sparse: { rowIdx: [[colIdx, count], ...] }
    bi_sparse = {}
    for i, row in enumerate(bigram_counts):
        nonzero = [[j, c] for j, c in enumerate(row) if c > 0]
        if nonzero:
            bi_sparse[i] = nonzero

    # Trigram sparse: { "prev_idx,cur_idx": [[nextIdx, count], ...] }
    tri_sparse = {}
    for (pi, ci), next_counts in trigram_counts.items():
        tri_sparse[f"{pi},{ci}"] = [[k, c] for k, c in sorted(next_counts.items())]

    display_tokens  = ["&lt;EOS&gt;" if t == EOS else t for t in idx_to_token]
    vocab_json      = json.dumps(display_tokens)
    bi_sparse_json  = json.dumps(bi_sparse)
    tri_sparse_json = json.dumps(tri_sparse)
    title           = stem.replace("_", " ").title()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — N-gram Table</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
  :root {{
    --bg:#0f0e11;--surface:#1a1820;--border:#2e2b38;--accent:#c8a96e;
    --accent2:#7c6fcd;--text:#e8e4dc;--muted:#6b6578;
    --cell-zero:#0f0e11;--cell-low:#1e1a2e;--cell-mid:#3d2f6b;
    --cell-high:#7c6fcd;--cell-max:#c8a96e;
  }}
  *,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'DM Mono',monospace;min-height:100vh;display:flex;flex-direction:column}}
  header{{padding:2rem 2.5rem 1.5rem;border-bottom:1px solid var(--border);display:flex;align-items:baseline;gap:1.5rem;flex-wrap:wrap}}
  header h1{{font-family:'Playfair Display',serif;font-size:1.6rem;color:var(--accent);letter-spacing:.02em;white-space:nowrap}}
  .meta{{font-size:.72rem;color:var(--muted);letter-spacing:.05em}}
  .controls{{padding:1rem 2.5rem;display:flex;gap:1rem;align-items:center;flex-wrap:wrap;border-bottom:1px solid var(--border);background:var(--surface)}}
  .controls label{{font-size:.72rem;color:var(--muted);letter-spacing:.08em;text-transform:uppercase}}
  input[type=text]{{background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--text);font-family:'DM Mono',monospace;font-size:.82rem;padding:.4rem .75rem;width:18rem;outline:none;transition:border-color .2s}}
  input[type=text]:focus{{border-color:var(--accent2)}}
  .toggle-group{{display:flex;gap:.25rem}}
  .toggle-group button{{background:var(--bg);border:1px solid var(--border);border-radius:4px;color:var(--muted);font-family:'DM Mono',monospace;font-size:.72rem;padding:.35rem .7rem;cursor:pointer;letter-spacing:.05em;transition:all .15s}}
  .toggle-group button.active{{background:var(--accent2);border-color:var(--accent2);color:#fff}}
  .pill-toggle{{display:flex;align-items:center;gap:.6rem;font-size:.72rem;color:var(--muted);letter-spacing:.05em}}
  .pill-toggle span{{transition:color .2s}}
  .pill-toggle span.active{{color:var(--text)}}
  .pill-switch{{position:relative;width:2.6rem;height:1.4rem;cursor:pointer}}
  .pill-switch input{{opacity:0;width:0;height:0;position:absolute}}
  .pill-track{{position:absolute;inset:0;background:var(--border);border-radius:1rem;transition:background .2s}}
  .pill-switch input:checked+.pill-track{{background:var(--accent2)}}
  .pill-thumb{{position:absolute;top:.2rem;left:.2rem;width:1rem;height:1rem;background:var(--text);border-radius:50%;transition:transform .2s}}
  .pill-switch input:checked~.pill-thumb{{transform:translateX(1.2rem)}}
  .hint{{font-size:.7rem;color:var(--muted);margin-left:auto}}
  .table-wrap{{flex:1;overflow:auto;padding:1.5rem 2.5rem 2.5rem}}
  table{{border-collapse:collapse;font-size:.7rem}}
  th,td{{width:2.1rem;height:2.1rem;text-align:center;vertical-align:middle;border:1px solid #1c1924}}
  th.row-hdr,th.col-hdr{{background:var(--surface);color:var(--accent);font-weight:500;position:sticky;z-index:2;white-space:nowrap;padding:0 .5rem;max-width:7rem;overflow:hidden;text-overflow:ellipsis}}
  th.row-hdr{{left:0;text-align:right;border-right:2px solid var(--border)}}
  th.col-hdr{{top:0;border-bottom:2px solid var(--border);writing-mode:vertical-rl;transform:rotate(180deg);height:6rem;width:2.1rem;padding:.4rem 0}}
  th.corner{{position:sticky;top:0;left:0;z-index:3;background:var(--bg);border-right:2px solid var(--border);border-bottom:2px solid var(--border)}}
  td.cell{{cursor:default;transition:filter .1s;position:relative}}
  td.cell:hover{{filter:brightness(1.6);outline:1px solid var(--accent);z-index:1}}
  td.cell.zero{{background:var(--cell-zero)}}
  #tip{{position:fixed;background:#2a2535;border:1px solid var(--accent2);border-radius:6px;padding:.5rem .8rem;font-size:.75rem;line-height:1.6;pointer-events:none;z-index:999;display:none;white-space:nowrap}}
  #tip .t-pair{{color:var(--accent);font-weight:500}}
  #tip .t-prob{{color:var(--accent2)}}
  tr.hidden{{display:none}}
  col.hidden-col{{visibility:collapse}}
  .tri-hdr{{color:var(--accent2) !important;font-style:italic}}
  .section-divider{{
    background:var(--surface);color:var(--muted);font-size:.65rem;
    letter-spacing:.12em;text-align:left;padding:.5rem 1rem;
    border-top:2px solid var(--border);border-bottom:2px solid var(--border);
    position:sticky;left:0;
  }}
  tr.divider-row td,tr.divider-row th{{border:none}}
  .legend{{display:flex;gap:.5rem;align-items:center;font-size:.68rem;color:var(--muted);margin-left:1rem}}
  .legend-swatch{{width:1rem;height:1rem;border-radius:2px;display:inline-block}}
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
    <button class="active" id="btn-both" onclick="setMode('both')">Row &amp; Col</button>
    <button id="btn-row" onclick="setMode('row')">Row only</button>
    <button id="btn-col" onclick="setMode('col')">Col only</button>
  </div>
  <div class="pill-toggle">
    <span id="lbl-count" class="active">COUNTS</span>
    <label class="pill-switch">
      <input type="checkbox" id="display-toggle" onchange="setDisplay(this.checked?'prob':'count')">
      <div class="pill-track"></div><div class="pill-thumb"></div>
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
<div class="table-wrap"><table id="tbl"></table></div>
<div id="tip"></div>
<script>
const VOCAB={vocab_json},BI_SPARSE={bi_sparse_json},TRI_SPARSE={tri_sparse_json},V=VOCAB.length;
const rowSums=new Float64Array(V);
for(const[ri,pairs]of Object.entries(BI_SPARSE)){{let s=0;for(const[,c]of pairs)s+=c;rowSums[ri]=s;}}
let globalMax=0;
for(const pairs of Object.values(BI_SPARSE))for(const[,c]of pairs)if(c>globalMax)globalMax=c;
for(const pairs of Object.values(TRI_SPARSE))for(const[,c]of pairs)if(c>globalMax)globalMax=c;
function countToColour(c){{
  if(c===0)return null;const t=c/globalMax;
  if(t<.25)return`color-mix(in srgb,var(--cell-low) ${{Math.round(t*400)}}%,var(--cell-zero))`;
  if(t<.5) return`color-mix(in srgb,var(--cell-mid) ${{Math.round((t-.25)*400)}}%,var(--cell-low))`;
  if(t<.75)return`color-mix(in srgb,var(--cell-high) ${{Math.round((t-.5)*400)}}%,var(--cell-mid))`;
  return`color-mix(in srgb,var(--cell-max) ${{Math.round((t-.75)*400)}}%,var(--cell-high))`;
}}
const tbl=document.getElementById('tbl'),tip=document.getElementById('tip');
let displayMode='count',filterMode='both';
const cg=document.createElement('colgroup');
cg.appendChild(document.createElement('col'));
const colEls=[];
for(let ci=0;ci<V;ci++){{const col=document.createElement('col');col.id=`col-${{ci}}`;cg.appendChild(col);colEls.push(col);}}
tbl.appendChild(cg);
const thead=tbl.createTHead(),hdrRow=thead.insertRow();
const corner=document.createElement('th');corner.className='corner';corner.textContent='row ↓  col →';hdrRow.appendChild(corner);
for(let ci=0;ci<V;ci++){{const th=document.createElement('th');th.className='col-hdr';th.textContent=VOCAB[ci];th.title=VOCAB[ci];hdrRow.appendChild(th);}}
const tbody=tbl.createTBody(),rows=[],cells=[];
for(let ri=0;ri<V;ri++){{
  const tr=tbody.insertRow();tr.id=`row-${{ri}}`;rows.push(tr);cells.push([]);
  const rh=document.createElement('th');rh.className='row-hdr';rh.textContent=VOCAB[ri];rh.title=VOCAB[ri];tr.appendChild(rh);
  const rowMap={{}};for(const[ci,c]of(BI_SPARSE[ri]||[]))rowMap[ci]=c;
  for(let ci=0;ci<V;ci++){{
    const td=document.createElement('td');td.className='cell';
    const c=rowMap[ci]||0;td.dataset.count=c;td.dataset.ri=ri;td.dataset.ci=ci;
    if(c===0){{td.classList.add('zero');}}else{{td.style.background=countToColour(c);td.textContent=c;}}
    td.addEventListener('mouseenter',onCellEnter);td.addEventListener('mouseleave',onCellLeave);
    tr.appendChild(td);cells[ri].push(td);
  }}
}}

// ── Trigram context rows ──────────────────────────────────────────────────
// Divider row
const divTr = tbody.insertRow();
divTr.id = 'tri-divider';
divTr.className = 'divider-row';
const divTh = document.createElement('th');
divTh.colSpan = V + 1;
divTh.className = 'section-divider';
divTh.textContent = 'BIGRAM CONTEXTS  (trigram distributions)';
divTr.appendChild(divTh);

// One row per observed (prev, cur) bigram context
const triRows = [];   // {{key, tr, label}}
for (const [key, pairs] of Object.entries(TRI_SPARSE)) {{
  const [pi, ci] = key.split(',').map(Number);
  const tr = tbody.insertRow();
  tr.id = `tri-${{key}}`;
  tr.dataset.triKey = key;
  triRows.push({{key, tr, pi, ci}});

  const rh = document.createElement('th');
  rh.className = 'row-hdr tri-hdr';
  const label = VOCAB[pi] + ' ' + VOCAB[ci];
  rh.textContent = label;
  rh.title = label + ' →';
  tr.appendChild(rh);

  const rowTotal = pairs.reduce((s,[,c])=>s+c, 0);
  const rowMap = {{}};
  for (const [k, c] of pairs) rowMap[k] = c;

  for (let ci2 = 0; ci2 < V; ci2++) {{
    const td = document.createElement('td');
    td.className = 'cell';
    const c = rowMap[ci2] || 0;
    td.dataset.count = c;
    td.dataset.triKey = key;
    td.dataset.ci = ci2;
    td.dataset.rowTotal = rowTotal;
    if (c === 0) {{
      td.classList.add('zero');
    }} else {{
      td.style.background = countToColour(c);
      td.textContent = c;
    }}
    td.addEventListener('mouseenter', onTriCellEnter);
    td.addEventListener('mouseleave', onCellLeave);
    tr.appendChild(td);
  }}
}}

document.getElementById('stats-label').textContent=`${{V}} tokens · ${{Object.values(BI_SPARSE).reduce((s,p)=>s+p.length,0).toLocaleString()}} bigrams · ${{Object.values(TRI_SPARSE).reduce((s,p)=>s+p.length,0).toLocaleString()}} trigrams`;
function onCellEnter(e){{
  const td=e.currentTarget,ri=+td.dataset.ri,ci=+td.dataset.ci,c=+td.dataset.count;
  const prob=rowSums[ri]>0?(c/rowSums[ri]):0;
  tip.innerHTML=`<div class="t-pair">"${{VOCAB[ri]}}" → "${{VOCAB[ci]}}"</div><div>count: ${{c}}</div><div class="t-prob">P(col|row): ${{prob.toFixed(4)}}</div>`;
  tip.style.display='block';document.addEventListener('mousemove',moveTip);
}}
function onCellLeave(){{tip.style.display='none';document.removeEventListener('mousemove',moveTip);}}
function moveTip(e){{tip.style.left=(e.clientX+14)+'px';tip.style.top=(e.clientY+14)+'px';}}
function setDisplay(mode){{
  displayMode=mode;
  document.getElementById('lbl-count').classList.toggle('active',mode==='count');
  document.getElementById('lbl-prob').classList.toggle('active',mode==='prob');
  // Unigram rows (bigram cells)
  for(let ri=0;ri<V;ri++)for(let ci=0;ci<V;ci++){{
    const td=cells[ri][ci],c=+td.dataset.count;
    if(c>0)td.textContent=mode==='prob'?(rowSums[ri]>0?(c/rowSums[ri]).toFixed(2):''):c;
  }}
  // Trigram rows
  for(const {{tr}} of triRows){{
    for(const td of tr.querySelectorAll('td.cell')){{
      const c=+td.dataset.count;
      if(c>0){{
        const total=+td.dataset.rowTotal;
        td.textContent=mode==='prob'?(total>0?(c/total).toFixed(2):''):c;
      }}
    }}
  }}
}}
function setMode(m){{
  filterMode=m;['both','row','col'].forEach(x=>document.getElementById('btn-'+x).classList.toggle('active',x===m));
  applyFilter(document.getElementById('search').value.trim().toLowerCase());
}}
function applyFilter(q){{
  if(!q){{
    for(const tr of rows)tr.classList.remove('hidden');
    for(const col of colEls)col.classList.remove('hidden-col');
    for(const {{tr}} of triRows)tr.classList.remove('hidden');
    document.getElementById('tri-divider').classList.remove('hidden');
    document.getElementById('hint-label').textContent='';
    return;
  }}
  const mr=[],mc=[];
  for(let i=0;i<V;i++)if(VOCAB[i].includes(q)){{mr.push(i);mc.push(i);}}
  const matchSet=new Set(mr);
  const sr=new Set(filterMode!=='col'?mr:Array.from({{length:V}},(_,i)=>i));
  const sc=new Set(filterMode!=='row'?mc:Array.from({{length:V}},(_,i)=>i));
  for(let ri=0;ri<V;ri++)rows[ri].classList.toggle('hidden',!sr.has(ri));
  for(let ci=0;ci<V;ci++)colEls[ci].classList.toggle('hidden-col',!sc.has(ci));
  // Trigram rows: show if either context token matches (row filter) or any col matches (col filter)
  let triVisible = 0;
  for(const {{pi, ci, tr}} of triRows){{
    const rowMatch = matchSet.has(pi)||matchSet.has(ci);
    const show = filterMode==='col' ? true : rowMatch;
    tr.classList.toggle('hidden', !show);
    if(show) triVisible++;
  }}
  document.getElementById('tri-divider').classList.toggle('hidden', triVisible===0);
  const total = mr.length + triVisible;
  document.getElementById('hint-label').textContent=total===0?'no matches':`${{mr.length}} unigram${{mr.length===1?'':'s'}}, ${{triVisible}} bigram context${{triVisible===1?'':'s'}}`;
}}
document.getElementById('search').addEventListener('input',e=>applyFilter(e.target.value.trim().toLowerCase()));
function onTriCellEnter(e) {{
  const td = e.currentTarget;
  const [pi, ci] = td.dataset.triKey.split(',').map(Number);
  const ci2 = +td.dataset.ci;
  const c = +td.dataset.count;
  const total = +td.dataset.rowTotal;
  const prob = total > 0 ? c / total : 0;
  tip.innerHTML =
    `<div class="t-pair">"${{VOCAB[pi]}} ${{VOCAB[ci]}}" → "${{VOCAB[ci2]}}"</div>` +
    `<div>count: ${{c}}</div>` +
    `<div class="t-prob">P(col|context): ${{prob.toFixed(4)}}</div>`;
  tip.style.display = 'block';
  document.addEventListener('mousemove', moveTip);
}}

</script>
</body>
</html>"""

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    return out_path


# ── CLI helpers ───────────────────────────────────────────────────────────────

def validate_prompt(raw: str, token_to_idx: dict[str, int]) -> list[str] | None:
    tokens = tokenise(raw)
    if not tokens:
        print("  x Prompt is empty after tokenisation.")
        return None
    unknown = [t for t in tokens if t not in token_to_idx]
    if unknown:
        print(f"  x Unknown token(s): {unknown}")
        return None
    return tokens


def pretty_print(prompt_tokens: list[str], generated: list[str]) -> None:
    prompt_str = " ".join(prompt_tokens)
    gen_str    = " ".join("<EOS>" if t == EOS else t for t in generated)
    full       = prompt_tokens + generated
    readable   = " ".join("." if t == EOS else t for t in full)
    print("\n" + "-" * 60)
    print(f"PROMPT   : {prompt_str}")
    print(f"GENERATED: {gen_str}")
    print(f"FULL TEXT: {readable}")
    print("-" * 60 + "\n")


def ask_token_limit() -> int:
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

def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python trigram_model.py <text_file>")
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
    print(f"  Total tokens      : {len(tokens):,}")

    token_to_idx, idx_to_token = build_vocab(tokens)
    V = len(token_to_idx)
    print(f"  Vocabulary size   : {V:,}  (including EOS)")

    bigram_counts  = build_bigram_counts(tokens, token_to_idx)
    trigram_counts = build_trigram_counts(tokens, token_to_idx)

    n_trigrams = sum(len(v) for v in trigram_counts.values())
    print(f"  Bigram contexts   : {sum(1 for r in bigram_counts if any(r)):,}")
    print(f"  Trigram contexts  : {len(trigram_counts):,}  ({n_trigrams:,} attested trigrams)")
    print(f"  Backoff alpha     : {ALPHA}")

    html_path = export_html(path, idx_to_token, bigram_counts, trigram_counts)
    print(f"  Visualisation     : {html_path}\n")

    max_new = ask_token_limit()
    print("  (Press F1 at the prompt to change the token limit.)\n")

    def limit_label() -> str:
        return "EOS only" if max_new == NO_LIMIT else str(max_new)

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

        if len(prompt_tokens) == 1:
            print("  ! Single-token prompt: starting in bigram-backoff mode.")

        generated = generate(
            prompt_tokens, max_new,
            token_to_idx, idx_to_token,
            bigram_counts, trigram_counts
        )
        pretty_print(prompt_tokens, generated)


if __name__ == "__main__":
    main()
