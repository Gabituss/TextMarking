"""
Веб-визуализация кластеризации реплик прямой речи по персонажам.

Выбираешь размеченный .jsonl из exports/ (или загружаешь), крутишь k/alpha/window —
и видишь 2D-карту реплик (точки по персонажам) + списки реплик с атрибуцией.

Запуск:  venv/bin/python speaker_web.py  →  http://127.0.0.1:5003
"""

import json
import os
import tempfile
from pathlib import Path

# Скачивание через xet на этой сети зависает на первых мегабайтах —
# качаем обычным HTTPS (env нужно выставить до импорта huggingface_hub)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
from huggingface_hub import list_repo_files, snapshot_download
# Нужен сам модуль huggingface_hub.utils.tqdm, а не одноимённый класс,
# которым пакет utils перекрывает атрибут — обычный import отдал бы класс
import importlib
hf_tqdm_module = importlib.import_module("huggingface_hub.utils.tqdm")

from speaker_cluster import (
    parse_lines, nearest_attribution, context_embeddings, choose_k, MODEL_NAME,
)
from speaker_encoder import (
    build_input, load_meta, pick_device, MAX_SEQ_LENGTH,
    BASE_MODEL as RAW_MODEL, OUTPUT_DIR as ENCODER_DIR,
)
from predict import (
    MODEL_DIR as TAGGER_DIR, load_model as load_tagger,
    predict as tag_words, format_output,
)

app = Flask(__name__)
EXPORTS = Path("exports")

print(f"Загрузка модели {MODEL_NAME}...")
MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
print("Модель загружена")

# Текущий этап долгой операции — фронтенд опрашивает /progress раз в полсекунды.
PROGRESS = {"stage": None, "percent": None}


def set_progress(stage, percent=None):
    PROGRESS["stage"] = stage
    PROGRESS["percent"] = percent


class _WebTqdm(hf_tqdm_module.tqdm):
    """tqdm, дублирующий байтовый прогресс скачивания в PROGRESS.

    Байты считаем сами из n: вне tty hub создаёт бар с disable=True,
    и tqdm.update() тогда выходит сразу, не наращивая self.n.
    """

    def update(self, n=1):
        super().update(n)
        # Только крупные файлы (веса): конфиги и словарь мелькают мгновенно
        if self.total and self.total > 10_000_000:
            self._done_bytes = getattr(self, "_done_bytes", 0) + (n or 0)
            set_progress("скачивание модели", round(100 * self._done_bytes / self.total))


def download_with_progress(repo_id: str):
    """snapshot_download с трансляцией прогресса: hf создаёт байтовые бары
    через huggingface_hub.utils.tqdm.tqdm — подменяем его на время скачивания."""
    # Не тянем дубликаты весов (.bin при наличии safetensors) и onnx/openvino —
    # SentenceTransformer их не использует, а это гигабайты
    ignore = ["*.onnx", "onnx/*", "openvino/*", "*.h5", "*.tflite"]
    if any(f.endswith(".safetensors") for f in list_repo_files(repo_id)):
        ignore.append("*.bin")
    orig = hf_tqdm_module.tqdm
    hf_tqdm_module.tqdm = _WebTqdm
    try:
        snapshot_download(repo_id, ignore_patterns=ignore)
    finally:
        hf_tqdm_module.tqdm = orig


def encode_with_progress(model, texts, stage="кодирование реплик"):
    chunks, bs = [], 8
    for s in range(0, len(texts), bs):
        set_progress(stage, round(100 * s / len(texts)))
        chunks.append(model.encode(texts[s:s + bs], normalize_embeddings=True))
    set_progress(stage, 100)
    return np.vstack(chunks)


# Сырой (без дообучения) сентенс-энкодер — лениво, при первом запросе
_RAW = None


def get_raw():
    global _RAW
    if _RAW is None:
        set_progress(f"скачивание модели {RAW_MODEL}")
        download_with_progress(RAW_MODEL)  # если в кэше — вернётся мгновенно
        set_progress("загрузка модели в память")
        _RAW = SentenceTransformer(RAW_MODEL, device=pick_device())
        _RAW.max_seq_length = MAX_SEQ_LENGTH
    return _RAW


# Модель разметки авторская/прямая речь (train.py) — лениво, при первом запросе
_TAGGER = None


def get_tagger():
    global _TAGGER
    if _TAGGER is None:
        if not Path(TAGGER_DIR).exists():
            raise FileNotFoundError(
                f"Нет модели разметки речи в {TAGGER_DIR}. Сначала: venv/bin/python train.py")
        set_progress("загрузка модели разметки речи")
        _TAGGER = load_tagger(TAGGER_DIR)
    return _TAGGER


def annotate_to_jsonl(text: str) -> Path:
    """Свой текст без разметки → временный .jsonl с тегами <A>/<D> от модели."""
    tokenizer, model, device = get_tagger()
    lines = [l for l in text.splitlines() if l.strip()]
    rows = []
    for n, line in enumerate(lines):
        set_progress("разметка прямой речи", round(100 * n / len(lines)))
        rows.append(json.dumps(
            {"text": format_output(tag_words(line, tokenizer, model, device))},
            ensure_ascii=False))
    path = Path(tempfile.gettempdir()) / "speaker_web_pasted.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    return path


# Обученный эмбеддер (speaker_encoder.py train) — лениво, при первом запросе.
# Перезагружаем, если модель на диске переобучили: иначе долго живущий сервер
# молча отдаёт результаты старой модели.
_TRAINED = None
_TRAINED_MTIME = None


def get_trained():
    global _TRAINED, _TRAINED_MTIME
    model_dir = Path(ENCODER_DIR)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"Нет обученной модели в {ENCODER_DIR}. Сначала: "
            f"venv/bin/python speaker_encoder.py train data/speakers.jsonl"
        )
    mtime = max(f.stat().st_mtime for f in model_dir.iterdir() if f.is_file())
    if _TRAINED is None or mtime != _TRAINED_MTIME:
        if _TRAINED is not None:
            print("Модель на диске обновилась — перезагружаю speaker-encoder")
        _TRAINED = SentenceTransformer(str(model_dir), device="cpu")
        _TRAINED.max_seq_length = MAX_SEQ_LENGTH
        _TRAINED_MTIME = mtime
    return _TRAINED


def run_clustering(path: Path, k, window, alpha, encoder="heuristic"):
    lines = parse_lines(path)
    # Плоская последовательность + структура по абзацам с глобальными индексами
    seq, doc = [], []
    for row in lines:
        drow = []
        for lab, text in row:
            drow.append({"gi": len(seq), "type": lab, "text": text})
            seq.append((lab, text))
        doc.append(drow)

    d_idx = [i for i, (lab, _) in enumerate(seq) if lab == "D"]
    if len(d_idx) < 4:
        return {"error": "Слишком мало реплик прямой речи (нужно ≥4)."}

    if encoder == "trained":
        # Контекст — внутри входного текста; окно и alpha из UI не участвуют:
        # окно обязано совпадать с обучением, поэтому берём его из меты модели
        w = load_meta(ENCODER_DIR).get("window", 2)
        vecs = encode_with_progress(
            get_trained(), [build_input(seq, i, w) for i in d_idx])
    elif encoder == "raw":
        # Сырой sbert: тот же входной текст «реплика [SEP] контекст»,
        # окно свободное (модель не привязана к обучению) — берём из UI
        vecs = encode_with_progress(
            get_raw(), [build_input(seq, i, window) for i in d_idx])
    else:
        set_progress("кодирование реплик (эвристика)")
        vecs = context_embeddings(seq, d_idx, MODEL, window=window, alpha=alpha)

    set_progress("кластеризация")
    silh = None
    if not k:
        k, silh = choose_k(vecs)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vecs)

    # 2D-проекция для карты
    xy = PCA(n_components=2, random_state=42).fit_transform(vecs)
    xy = (xy - xy.min(0)) / (np.ptp(xy, 0) + 1e-9)  # нормируем в [0,1]

    d_cluster = {i: int(labels[n]) for n, i in enumerate(d_idx)}

    points, clusters = [], {}
    for n, i in enumerate(d_idx):
        item = {
            "speech": seq[i][1],
            "attribution": nearest_attribution(seq, i),
            "cluster": int(labels[n]),
            "x": float(xy[n, 0]),
            "y": float(xy[n, 1]),
        }
        points.append(item)
        clusters.setdefault(int(labels[n]), []).append(item)

    # Документ по абзацам: к каждому сегменту-реплике подставляем кластер
    document = []
    for drow in doc:
        document.append([
            {"type": s["type"], "text": s["text"], "cluster": d_cluster.get(s["gi"])}
            for s in drow
        ])

    sizes = {str(c): len(v) for c, v in clusters.items()}
    return {
        "k": int(k),
        "silhouette": round(float(silh), 3) if silh is not None else None,
        "n_replicas": len(d_idx),
        "points": points,
        "sizes": sizes,
        "document": document,
    }


PAGE = r"""
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Персонажи · визуализация</title>
<style>
  :root{--ink:#1a1a1a;--muted:#6b7280;--line:#e5e7eb;--accent:#1d4ed8}
  *{box-sizing:border-box}
  body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    color:var(--ink);background:#fafaf9;line-height:1.55}
  .wrap{max-width:1080px;margin:0 auto;padding:36px 24px 90px}
  h1{font-size:25px;font-weight:650;letter-spacing:-.02em;margin:0 0 4px}
  .sub{color:var(--muted);margin:0 0 24px;font-size:14px}
  .panel{display:flex;gap:18px;align-items:flex-end;flex-wrap:wrap;
    padding:18px;background:#fff;border:1px solid var(--line);border-radius:12px}
  .ctl{display:flex;flex-direction:column;gap:5px}
  .ctl label{font-size:12px;color:var(--muted);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
  select,input[type=number]{padding:9px 11px;border:1px solid var(--line);border-radius:8px;
    font-size:14px;font-family:inherit;background:#fff}
  input[type=range]{width:150px}
  .val{font-size:13px;color:var(--ink);font-weight:600}
  button{font-family:inherit;font-size:14px;font-weight:600;border-radius:9px;cursor:pointer;
    border:0;background:var(--ink);color:#fff;padding:11px 22px}
  button:hover{background:#000}
  button:disabled{opacity:.5;cursor:default}

  .meta{margin:22px 0 10px;font-size:14px;color:var(--muted)}
  .layout{display:grid;grid-template-columns:1fr 1fr;gap:22px}
  @media(max-width:840px){.layout{grid-template-columns:1fr}}
  .card{background:#fff;border:1px solid var(--line);border-radius:12px;padding:18px}
  .card h2{font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted);
    font-weight:600;margin:0 0 14px}
  #map{width:100%;height:auto;display:block;border-radius:8px}
  .dot{cursor:pointer;transition:r .1s}
  .dot:hover{stroke:#000;stroke-width:1.5}
  .legend{display:flex;flex-wrap:wrap;gap:12px;margin-top:14px;font-size:13px}
  .legend span{display:inline-flex;align-items:center;gap:6px;cursor:pointer;padding:3px 6px;border-radius:6px}
  .legend span.off{opacity:.35}
  .legend i{width:12px;height:12px;border-radius:50%}

  .clusters{display:flex;flex-direction:column;gap:16px;max-height:560px;overflow:auto}
  .cl h3{margin:0 0 6px;font-size:14px;display:flex;align-items:center;gap:8px}
  .cl h3 i{width:12px;height:12px;border-radius:50%}
  .cl .rep{font-size:14px;font-family:"Iowan Old Style",Georgia,serif;
    padding:5px 9px;border-radius:6px;margin-bottom:4px}
  .cl .rep .attr{color:var(--muted);font-style:italic;font-size:12px;font-family:-apple-system,sans-serif}
  .reader{font-family:"Iowan Old Style",Georgia,serif;font-size:16px;line-height:1.85;
    max-height:620px;overflow:auto;padding-right:6px}
  .reader p{margin:0 0 12px}
  .reader .a{color:#374151}
  .reader .d{border-radius:4px;padding:1px 4px;cursor:default}
  #tip{position:fixed;pointer-events:none;background:#111;color:#fff;padding:8px 11px;
    border-radius:8px;font-size:13px;max-width:320px;display:none;z-index:20;line-height:1.4}
  #tip .a{opacity:.7;font-size:12px}
  .loader{color:var(--muted);font-size:14px}
  .freetext{margin-top:14px}
  .freetext textarea{width:100%;padding:11px 13px;border:1px solid var(--line);
    border-radius:12px;font-family:inherit;font-size:14px;line-height:1.5;
    background:#fff;resize:vertical;color:var(--ink)}
  .freetext textarea::placeholder{color:var(--muted)}
</style>
</head>
<body>
<div class="wrap">
  <h1>Реплики по персонажам</h1>
  <p class="sub">Эмбеддинг каждой реплики считается с учётом соседних сегментов (авторская атрибуция). Карта — PCA-проекция, цвет — кластер.</p>

  <div class="panel">
    <div class="ctl">
      <label>Файл</label>
      <select id="file">{{ options | safe }}</select>
    </div>
    <div class="ctl">
      <label>Эмбеддер</label>
      <select id="encoder"
        onchange="const v=this.value; alpha.disabled=(v!=='heuristic'); document.getElementById('window').disabled=(v==='trained')">
        <option value="heuristic">эвристика (база + α·соседи)</option>
        <option value="raw">{{ raw_name }} (без дообучения)</option>
        <option value="trained" {{ trained_attr | safe }}>обученный (speaker-encoder)</option>
      </select>
    </div>
    <div class="ctl">
      <label>Персонажей (k)</label>
      <input type="number" id="k" min="2" max="12" placeholder="авто" style="width:90px">
    </div>
    <div class="ctl">
      <label>Контекст соседей α · <span class="val" id="av">0.5</span></label>
      <input type="range" id="alpha" min="0" max="1.5" step="0.1" value="0.5"
        oninput="av.textContent=this.value">
    </div>
    <div class="ctl">
      <label>Окно · <span class="val" id="wv">1</span></label>
      <input type="range" id="window" min="0" max="4" step="1" value="1"
        oninput="wv.textContent=this.value">
    </div>
    <button id="go" onclick="run()">Кластеризовать</button>
  </div>

  <div class="freetext">
    <textarea id="freetext" rows="5"
      placeholder="Или вставьте сюда свой текст без разметки — прямая речь будет найдена обученной моделью (rubert-speech), реплики кластеризованы по персонажам. Пока поле не пустое, файл выше игнорируется."></textarea>
  </div>

  <div class="meta" id="meta"></div>

  <div class="layout">
    <div class="card">
      <h2>Карта реплик</h2>
      <svg id="map" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"></svg>
      <div class="legend" id="legend"></div>
    </div>
    <div class="card">
      <h2>Кластеры</h2>
      <div class="clusters" id="clusters"><span class="loader">Нажмите «Кластеризовать».</span></div>
    </div>
  </div>

  <div class="card" style="margin-top:22px">
    <h2>Текст · подсветка по персонажам</h2>
    <div class="reader" id="reader"><span class="loader">Здесь появится текст с цветами персонажей.</span></div>
  </div>
</div>
<div id="tip"></div>

<script>
const PALETTE = ["#1d4ed8","#b45309","#15803d","#be123c","#7c3aed","#0891b2",
                 "#ca8a04","#db2777","#4d7c0f","#9333ea","#0d9488","#dc2626"];
let HIDDEN = new Set();
let LAST = null;

async function run(){
  const go = document.getElementById('go');
  go.disabled = true; go.textContent = 'Считаю…';
  document.getElementById('meta').textContent = 'Считаю эмбеддинги и кластеризую…';
  document.getElementById('clusters').innerHTML = '<span class="loader">Эмбеддинги и кластеризация…</span>';
  // Прогресс долгих этапов (скачивание модели, кодирование) — с сервера
  let polling = true;
  const poll = setInterval(async () => {
    try{
      const p = await (await fetch('/progress')).json();
      if(polling && p.stage){
        document.getElementById('meta').textContent =
          p.stage + (p.percent != null ? ` · ${p.percent}%` : '') + '…';
      }
    } catch(e){}
  }, 500);
  try{
    const body = {
      file: document.getElementById('file').value,
      text: document.getElementById('freetext').value,
      encoder: document.getElementById('encoder').value,
      k: document.getElementById('k').value,
      alpha: document.getElementById('alpha').value,
      window: document.getElementById('window').value,
    };
    const r = await fetch('/cluster', {method:'POST',
      headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)});
    polling = false; clearInterval(poll);
    let data;
    try { data = await r.json(); }
    catch(e){ throw new Error('Сервер вернул не-JSON (код ' + r.status + ')'); }
    if(!r.ok || data.error){ throw new Error(data.error || ('Ошибка сервера ' + r.status)); }
    LAST = data; HIDDEN.clear();
    draw(data);
  } catch(err){
    document.getElementById('meta').textContent = '✗ ' + err.message;
    document.getElementById('clusters').innerHTML =
      '<span class="loader">Не удалось получить результат.</span>';
  } finally {
    polling = false; clearInterval(poll);
    go.disabled = false; go.textContent = 'Кластеризовать';
  }
}

function color(c){ return PALETTE[c % PALETTE.length]; }

function draw(data){
  const sil = data.silhouette!==null ? `, silhouette ${data.silhouette}` : '';
  document.getElementById('meta').textContent =
    `Реплик: ${data.n_replicas} · персонажей: ${data.k}${sil}`;

  // карта
  const svg = document.getElementById('map');
  svg.innerHTML = '';
  data.points.forEach(p => {
    if(HIDDEN.has(p.cluster)) return;
    const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
    c.setAttribute('cx', (4 + p.x*92).toFixed(2));
    c.setAttribute('cy', (96 - p.y*92).toFixed(2));
    c.setAttribute('r', 1.5);
    c.setAttribute('fill', color(p.cluster));
    c.setAttribute('fill-opacity', .75);
    c.setAttribute('class','dot');
    c.addEventListener('mousemove', e => showTip(e, p));
    c.addEventListener('mouseleave', hideTip);
    svg.appendChild(c);
  });

  // легенда
  const leg = document.getElementById('legend');
  leg.innerHTML = '';
  Object.keys(data.sizes).map(Number).sort((a,b)=>a-b).forEach(c => {
    const s = document.createElement('span');
    s.className = HIDDEN.has(c) ? 'off' : '';
    s.innerHTML = `<i style="background:${color(c)}"></i>Персонаж ${c} · ${data.sizes[c]}`;
    s.onclick = () => { HIDDEN.has(c)?HIDDEN.delete(c):HIDDEN.add(c); draw(LAST); };
    leg.appendChild(s);
  });

  // списки
  const byCluster = {};
  data.points.forEach(p => (byCluster[p.cluster] ??= []).push(p));
  const cont = document.getElementById('clusters');
  cont.innerHTML = '';
  Object.keys(byCluster).map(Number).sort((a,b)=>byCluster[b].length-byCluster[a].length)
    .forEach(c => {
      const div = document.createElement('div'); div.className='cl';
      div.innerHTML = `<h3><i style="background:${color(c)}"></i>Персонаж ${c} · ${byCluster[c].length} реплик</h3>`;
      byCluster[c].slice(0, 8).forEach(p => {
        const r = document.createElement('div'); r.className='rep';
        r.style.background = color(c) + '18';
        r.innerHTML = esc(p.speech.slice(0,140)) +
          (p.attribution ? ` <span class="attr">← ${esc(p.attribution.slice(0,50))}</span>` : '');
        div.appendChild(r);
      });
      if(byCluster[c].length > 8){
        const m = document.createElement('div'); m.className='rep attr';
        m.textContent = `… ещё ${byCluster[c].length-8}`; div.appendChild(m);
      }
      cont.appendChild(div);
    });

  // текст с подсветкой по персонажам
  const reader = document.getElementById('reader');
  reader.innerHTML = '';
  (data.document || []).forEach(row => {
    const p = document.createElement('p');
    row.forEach((seg, idx) => {
      if(idx > 0) p.appendChild(document.createTextNode(' '));
      const sp = document.createElement('span');
      if(seg.type === 'D' && seg.cluster !== null && seg.cluster !== undefined){
        const col = color(seg.cluster);
        sp.className = 'd';
        sp.style.background = col + '20';
        sp.style.color = col;
        sp.style.boxShadow = 'inset 0 -2px 0 ' + col;
        sp.title = 'Персонаж ' + seg.cluster;
        if(HIDDEN.has(seg.cluster)) sp.style.opacity = '.25';
      } else {
        sp.className = 'a';
      }
      sp.textContent = seg.text;
      p.appendChild(sp);
    });
    reader.appendChild(p);
  });
}

function showTip(e, p){
  const t = document.getElementById('tip');
  t.style.display='block'; t.style.left=(e.clientX+14)+'px'; t.style.top=(e.clientY+14)+'px';
  t.innerHTML = esc(p.speech.slice(0,160)) +
    (p.attribution ? `<br><span class="a">← ${esc(p.attribution.slice(0,60))}</span>` : '');
}
function hideTip(){ document.getElementById('tip').style.display='none'; }
function esc(s){ return s.replace(/[&<>]/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
</script>
</body>
</html>
"""


@app.route("/")
def index():
    files = sorted(EXPORTS.glob("*.jsonl"))
    options = "".join(f'<option value="{f.name}">{f.name}</option>' for f in files)
    if not options:
        options = '<option value="">(нет файлов в exports/)</option>'
    trained_attr = "" if Path(ENCODER_DIR).exists() else "disabled"
    return render_template_string(PAGE, options=options, trained_attr=trained_attr,
                                  raw_name=RAW_MODEL.split("/")[-1])


@app.route("/progress")
def progress():
    return jsonify(PROGRESS)


@app.route("/cluster", methods=["POST"])
def cluster():
    data = request.get_json(force=True)
    k = data.get("k")
    k = int(k) if str(k).strip() else None
    try:
        text = (data.get("text") or "").strip()
        if text:
            # Свой текст: сначала размечаем речь моделью, потом кластеризуем
            path = annotate_to_jsonl(text)
        else:
            fname = data.get("file", "")
            path = EXPORTS / fname
            if not fname or not path.exists():
                return jsonify({"error": "Файл не найден в exports/."})
        result = run_clustering(
            path, k=k,
            window=int(data.get("window", 1)),
            alpha=float(data.get("alpha", 0.5)),
            encoder=data.get("encoder", "heuristic"),
        )
    except Exception as e:
        return jsonify({"error": f"{type(e).__name__}: {e}"})
    finally:
        set_progress(None)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5003, debug=False)
