"""
Веб-разметчик текста на авторскую (A) и прямую (D) речь.

Поток работы:
  1. Вставить текст или загрузить .txt
  2. Черновая разметка по пунктуации («…», тире-диалоги) — или моделью
  3. Поправить руками: выделить слова → кнопка A / D / Очистить (или клавиши a/d/o)
  4. Сохранить прямо в exports/<имя>_annotated.jsonl → готово для preprocess.py

Запуск:  venv/bin/python annotator.py  →  http://127.0.0.1:5002
"""

import json
import re
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)
EXPORTS = Path("exports")

DASHES = {"–", "—", "-", "―"}
WORD_RE = re.compile(r"\S+")

# Модель грузится лениво — только если нажмут «предразметить моделью»
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from predict import load_model
        _MODEL = load_model("models/rubert-speech/best")
    return _MODEL


def char_labels_heuristic(line: str) -> list[str]:
    """Черновые посимвольные метки: A по умолчанию, D для прямой речи."""
    n = len(line)
    lab = ["A"] * n

    # 1) Всё внутри кавычек «…» или „…" → прямая речь
    in_quote = False
    for i, ch in enumerate(line):
        if ch in "«„":
            in_quote = True
            lab[i] = "D"
        elif ch in "»" and in_quote:
            lab[i] = "D"
            in_quote = False
        elif in_quote:
            lab[i] = "D"

    # 2) Диалог через тире: только если строка начинается с тире.
    #    Реплика = D, авторская вставка между « – … – » = A (тоггл на " – ").
    stripped = line.lstrip()
    offset = len(line) - len(stripped)
    if stripped[:1] in DASHES:
        state = "D"
        for i in range(offset, n):
            ch = line[i]
            is_marker = ch in DASHES and (i == offset or line[i - 1] == " ")
            if is_marker:
                if i != offset:
                    state = "A" if state == "D" else "D"
                lab[i] = "A"  # само тире — служебный символ
                continue
            if lab[i] != "D":  # не затираем кавычки
                lab[i] = state
    return lab


def words_with_labels(line: str, char_lab: list[str]) -> list[dict]:
    """Слова + метка каждого по большинству символов."""
    words = []
    for m in WORD_RE.finditer(line):
        seg = char_lab[m.start():m.end()]
        counts = {"A": seg.count("A"), "D": seg.count("D"), "O": seg.count("O")}
        label = max(counts, key=counts.get)
        words.append({"w": m.group(), "l": label})
    return words


def prelabel_heuristic(text: str) -> list[list[dict]]:
    out = []
    for line in text.splitlines():
        if line.strip():
            out.append(words_with_labels(line, char_labels_heuristic(line)))
        else:
            out.append([])  # пустая строка
    return out


def prelabel_model(text: str) -> list[list[dict]]:
    from predict import predict
    tokenizer, model, device = get_model()
    cat2lab = {"AUTHOR": "A", "DIRECT": "D", "O": "O"}
    out = []
    for line in text.splitlines():
        if line.strip():
            pairs = predict(line, tokenizer, model, device)
            out.append([{"w": w, "l": cat2lab.get(c, "O")} for w, c in pairs])
        else:
            out.append([])
    return out


PAGE = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметчик речи</title>
<style>
  :root {
    --author:#1d4ed8; --author-bg:#dbeafe;
    --direct:#b45309; --direct-bg:#fef3c7;
    --ink:#1a1a1a; --muted:#6b7280; --line:#e5e7eb;
  }
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    color:var(--ink);background:#fafaf9;line-height:1.6}
  .wrap{max-width:920px;margin:0 auto;padding:40px 24px 100px}
  h1{font-size:26px;font-weight:650;letter-spacing:-.02em;margin:0 0 4px}
  .sub{color:var(--muted);margin:0 0 24px;font-size:14px}
  textarea{width:100%;min-height:140px;padding:14px 16px;font-size:15px;
    font-family:inherit;line-height:1.55;border:1px solid var(--line);
    border-radius:10px;resize:vertical;background:#fff}
  textarea:focus{outline:2px solid var(--author);border-color:transparent}
  .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:12px}
  button{font-family:inherit;font-size:14px;font-weight:550;border-radius:9px;
    cursor:pointer;border:1px solid var(--line);background:#fff;padding:10px 16px}
  button:hover{border-color:var(--ink)}
  button.primary{background:var(--ink);color:#fff;border-color:var(--ink)}
  button.primary:hover{background:#000}
  .file-label{font-size:14px;color:var(--muted);cursor:pointer}
  .file-label input{display:none}
  .file-label b{color:var(--author);text-decoration:underline}

  .toolbar{position:sticky;top:0;background:#fafaf9ee;backdrop-filter:blur(6px);
    padding:14px 0;margin-top:24px;border-bottom:1px solid var(--line);
    display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:5}
  .toolbar .lab{padding:8px 14px}
  .toolbar .lab.a{color:var(--author);background:var(--author-bg);border-color:var(--author)}
  .toolbar .lab.d{color:var(--direct);background:var(--direct-bg);border-color:var(--direct)}
  .stat{margin-left:auto;font-size:13px;color:var(--muted)}
  .hint{font-size:13px;color:var(--muted);margin:10px 0 0}
  kbd{background:#fff;border:1px solid var(--line);border-bottom-width:2px;
    border-radius:5px;padding:1px 6px;font-family:ui-monospace,monospace;font-size:12px}

  #editor{margin-top:18px;padding:24px 26px;background:#fff;border:1px solid var(--line);
    border-radius:12px;font-family:"Iowan Old Style","Palatino",Georgia,serif;
    font-size:17px;line-height:2.05;user-select:none}
  .ln{margin:0 0 2px}
  .gap{height:14px}
  .w{padding:1px 2px;border-radius:3px;cursor:pointer}
  .w.A{color:var(--author);background:var(--author-bg)}
  .w.D{color:var(--direct);background:var(--direct-bg)}
  .w.O{color:var(--muted)}
  .w.sel{outline:2px solid var(--ink);outline-offset:1px}

  .save{margin-top:22px;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
  .save input[type=text]{padding:9px 12px;border:1px solid var(--line);border-radius:8px;
    font-size:14px;font-family:inherit;min-width:240px}
  .msg{font-size:14px;color:#15803d}
</style>
</head>
<body>
<div class="wrap">
  <h1>Разметчик речи</h1>
  <p class="sub">Чёрновая разметка по правилам или моделью → правка мышкой → сохранение в <code>exports/</code> для обучения.</p>

  <textarea id="input" placeholder="Вставьте текст…"></textarea>
  <div class="row">
    <button class="primary" onclick="prelabel('heuristic')">Предразметить (правила)</button>
    <button onclick="prelabel('model')">Предразметить (модель)</button>
    <label class="file-label">
      <input type="file" accept=".txt" id="file">
      или <b>загрузить .txt</b>
    </label>
  </div>

  <div id="work" style="display:none">
    <div class="toolbar">
      <button class="lab a" onclick="apply('A')">Авторская <kbd>A</kbd></button>
      <button class="lab d" onclick="apply('D')">Прямая <kbd>D</kbd></button>
      <button class="lab" onclick="apply('O')">Очистить <kbd>O</kbd></button>
      <span class="stat" id="stat"></span>
    </div>
    <p class="hint">Выделите слова мышью (клик или протяжка), затем нажмите кнопку или клавишу. Выделение остаётся для повторной правки.</p>
    <div id="editor"></div>

    <div class="save">
      <input type="text" id="fname" placeholder="имя файла (без расширения)">
      <button class="primary" onclick="save()">Сохранить в exports/</button>
      <button onclick="download()">Скачать .jsonl</button>
      <span class="msg" id="msg"></span>
    </div>
  </div>
</div>

<script>
let DOC = [];          // [[{w,l},...], ...]  по строкам
let FLAT = [];         // плоский список ссылок на слова + (li,wi)
let sel = new Set();   // выбранные глобальные индексы
let anchor = null, dragging = false;

document.getElementById('file').addEventListener('change', async e => {
  const f = e.target.files[0]; if(!f) return;
  document.getElementById('input').value = await f.text();
  const base = f.name.replace(/\.[^.]+$/, '');
  document.getElementById('fname').value = base + '_annotated';
  prelabel('heuristic');
});

async function prelabel(mode){
  const text = document.getElementById('input').value;
  if(!text.trim()) return;
  const r = await fetch('/prelabel', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text, mode})
  });
  const data = await r.json();
  if(data.error){ alert(data.error); return; }
  DOC = data.lines;
  if(!document.getElementById('fname').value)
    document.getElementById('fname').value = 'annotated';
  render();
  document.getElementById('work').style.display = 'block';
}

function render(){
  const ed = document.getElementById('editor');
  ed.innerHTML = '';
  FLAT = []; sel.clear();
  DOC.forEach((line, li) => {
    if(line.length === 0){
      const g = document.createElement('div'); g.className='gap'; ed.appendChild(g); return;
    }
    const p = document.createElement('p'); p.className='ln';
    line.forEach((tok, wi) => {
      const gi = FLAT.length;
      const s = document.createElement('span');
      s.className = 'w ' + tok.l;
      s.textContent = tok.w;
      s.dataset.gi = gi;
      s.addEventListener('mousedown', ev => { ev.preventDefault(); startSel(gi); });
      s.addEventListener('mouseenter', () => { if(dragging) extendSel(gi); });
      p.appendChild(s);
      p.appendChild(document.createTextNode(' '));
      FLAT.push({li, wi, el:s});
    });
    ed.appendChild(p);
  });
  updateStat();
}

function startSel(gi){ dragging=true; anchor=gi; sel.clear(); sel.add(gi); paintSel(); }
function extendSel(gi){
  sel.clear();
  const [a,b] = [Math.min(anchor,gi), Math.max(anchor,gi)];
  for(let i=a;i<=b;i++) sel.add(i);
  paintSel();
}
document.addEventListener('mouseup', () => { dragging=false; });

function paintSel(){
  FLAT.forEach((t,i)=> t.el.classList.toggle('sel', sel.has(i)));
}

function apply(label){
  if(sel.size===0) return;
  sel.forEach(gi => {
    const {li,wi,el} = FLAT[gi];
    DOC[li][wi].l = label;
    el.className = 'w ' + label + ' sel';
  });
  updateStat();
}

function updateStat(){
  let a=0,d=0,o=0;
  DOC.forEach(l=>l.forEach(t=>{ t.l==='A'?a++:t.l==='D'?d++:o++; }));
  document.getElementById('stat').textContent =
    `Авторская: ${a} · Прямая: ${d} · Нейтрально: ${o}`;
}

document.addEventListener('keydown', e => {
  if(e.target.tagName === 'TEXTAREA' || e.target.tagName === 'INPUT') return;
  const k = e.key.toLowerCase();
  if(k==='a'||k==='ф') apply('A');
  else if(k==='d'||k==='в') apply('D');
  else if(k==='o'||k==='щ'||k==='x'||k==='ч') apply('O');
});

function buildJSONL(){
  const lines = [];
  DOC.forEach(line => {
    if(line.length===0) return;               // пустые строки пропускаем
    const runs = [];
    line.forEach(({w,l}) => {
      const last = runs[runs.length-1];
      if(last && last.l===l) last.ws.push(w);
      else runs.push({l, ws:[w]});
    });
    const tagged = runs.map(r => {
      const t = r.ws.join(' ');
      return r.l==='A' ? `<A>${t}</A>` : r.l==='D' ? `<D>${t}</D>` : t;
    }).join(' ');
    lines.push(JSON.stringify({text: tagged}));
  });
  return lines.join('\n') + '\n';
}

async function save(){
  let name = document.getElementById('fname').value.trim() || 'annotated';
  const r = await fetch('/save', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({filename:name, content:buildJSONL()})
  });
  const data = await r.json();
  document.getElementById('msg').textContent =
    data.error ? '✗ ' + data.error : '✓ Сохранено: ' + data.path;
}

function download(){
  let name = (document.getElementById('fname').value.trim() || 'annotated') + '.jsonl';
  const blob = new Blob([buildJSONL()], {type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = name; a.click();
  URL.revokeObjectURL(a.href);
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(PAGE)


@app.route("/prelabel", methods=["POST"])
def prelabel():
    data = request.get_json(force=True)
    text = data.get("text", "")
    mode = data.get("mode", "heuristic")
    try:
        if mode == "model":
            lines = prelabel_model(text)
        else:
            lines = prelabel_heuristic(text)
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})
    return jsonify({"lines": lines})


@app.route("/save", methods=["POST"])
def save():
    data = request.get_json(force=True)
    name = re.sub(r"[^\w.-]", "_", data.get("filename", "annotated")) or "annotated"
    if not name.endswith(".jsonl"):
        name += ".jsonl"
    content = data.get("content", "")
    if not content.strip():
        return jsonify({"error": "пустой документ"})
    EXPORTS.mkdir(exist_ok=True)
    path = EXPORTS / name
    path.write_text(content, encoding="utf-8")
    return jsonify({"path": str(path)})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=False)
