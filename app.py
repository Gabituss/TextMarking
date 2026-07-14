"""
Мини веб-приложение для разметки текста на авторскую и прямую речь.
Запуск:  venv/bin/python app.py   →  http://127.0.0.1:5001
"""

import html
from flask import Flask, request, render_template_string

from predict import load_model, predict

app = Flask(__name__)

# Загружаем модель один раз при старте
print("Загрузка модели...")
TOKENIZER, MODEL, DEVICE = load_model("models/rubert-speech/best")
print(f"Модель загружена (device: {DEVICE})")


def annotate_html(text: str) -> str:
    """Разметить текст построчно и вернуть HTML с цветными span'ами."""
    out_lines = []
    for line in text.splitlines():
        if not line.strip():
            out_lines.append("")  # пустая строка → отступ
            continue
        pairs = predict(line, TOKENIZER, MODEL, DEVICE)
        out_lines.append(render_line(pairs))
    # Каждая строка — отдельный абзац, пустые дают вертикальный отступ
    return "".join(
        f'<p class="line">{l}</p>' if l else '<p class="gap"></p>'
        for l in out_lines
    )


def render_line(pairs):
    """Сгруппировать подряд идущие слова одной категории в span."""
    spans = []
    buf, cur = [], None
    for word, cat in pairs:
        if cat != cur:
            spans.append((cur, buf))
            buf, cur = [], cat
        buf.append(word)
    spans.append((cur, buf))

    html_parts = []
    for cat, words in spans:
        if not words:
            continue
        chunk = html.escape(" ".join(words))
        if cat == "AUTHOR":
            html_parts.append(f'<span class="author">{chunk}</span>')
        elif cat == "DIRECT":
            html_parts.append(f'<span class="direct">{chunk}</span>')
        else:
            html_parts.append(f"<span>{chunk}</span>")
    return " ".join(html_parts)


PAGE = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Разметка речи</title>
<style>
  :root {
    --author: #1d4ed8;
    --author-bg: #dbeafe;
    --direct: #b45309;
    --direct-bg: #fef3c7;
    --ink: #1a1a1a;
    --muted: #6b7280;
    --line: #e5e7eb;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: var(--ink);
    background: #fafaf9;
    line-height: 1.6;
  }
  .wrap { max-width: 860px; margin: 0 auto; padding: 48px 24px 80px; }
  h1 { font-size: 28px; font-weight: 650; letter-spacing: -0.02em; margin: 0 0 4px; }
  .sub { color: var(--muted); margin: 0 0 32px; font-size: 15px; }
  .legend { display: flex; gap: 20px; margin: 0 0 20px; font-size: 14px; }
  .legend span { display: inline-flex; align-items: center; gap: 7px; }
  .dot { width: 13px; height: 13px; border-radius: 3px; }
  .dot.a { background: var(--author-bg); border: 1.5px solid var(--author); }
  .dot.d { background: var(--direct-bg); border: 1.5px solid var(--direct); }
  textarea {
    width: 100%; min-height: 170px; padding: 14px 16px; font-size: 15px;
    font-family: inherit; line-height: 1.55; border: 1px solid var(--line);
    border-radius: 10px; resize: vertical; background: #fff;
  }
  textarea:focus { outline: 2px solid var(--author); border-color: transparent; }
  .controls { display: flex; align-items: center; gap: 16px; margin-top: 14px; flex-wrap: wrap; }
  button {
    background: var(--ink); color: #fff; border: 0; padding: 11px 22px;
    font-size: 15px; font-weight: 550; border-radius: 9px; cursor: pointer;
  }
  button:hover { background: #000; }
  .file-label { font-size: 14px; color: var(--muted); cursor: pointer; }
  .file-label input { display: none; }
  .file-label b { color: var(--author); text-decoration: underline; }
  .result {
    margin-top: 36px; padding: 28px 30px; background: #fff;
    border: 1px solid var(--line); border-radius: 12px;
    font-family: "Iowan Old Style", "Palatino", Georgia, serif;
    font-size: 17px; line-height: 1.75;
  }
  .result h2 { font-family: -apple-system, sans-serif; font-size: 13px;
    text-transform: uppercase; letter-spacing: 0.05em; color: var(--muted);
    font-weight: 600; margin: 0 0 18px; }
  .result .line { margin: 0 0 2px; }
  .result .gap { height: 14px; margin: 0; }
  .author { color: var(--author); background: var(--author-bg);
    border-radius: 3px; padding: 1px 3px; }
  .direct { color: var(--direct); background: var(--direct-bg);
    border-radius: 3px; padding: 1px 3px; }
  .empty { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Разметка речи</h1>
  <p class="sub">Вставьте текст или загрузите файл — модель выделит авторскую и прямую речь.</p>

  <div class="legend">
    <span><i class="dot a"></i> Авторская речь</span>
    <span><i class="dot d"></i> Прямая речь</span>
  </div>

  <form method="post" enctype="multipart/form-data">
    <textarea name="text" placeholder="Например: Солдат пришёл к старухе и говорит: «Дай-ка поесть!»">{{ text }}</textarea>
    <div class="controls">
      <button type="submit">Разметить</button>
      <label class="file-label">
        <input type="file" name="file" accept=".txt" onchange="this.form.submit()">
        или <b>выбрать .txt файл</b>
      </label>
    </div>
  </form>

  {% if result %}
  <div class="result">
    <h2>Результат</h2>
    {{ result | safe }}
  </div>
  {% endif %}
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    text, result = "", None
    if request.method == "POST":
        uploaded = request.files.get("file")
        if uploaded and uploaded.filename:
            text = uploaded.read().decode("utf-8", errors="replace")
        else:
            text = request.form.get("text", "")
        if text.strip():
            result = annotate_html(text)
    return render_template_string(PAGE, text=text, result=result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=False)
