"""
Кластеризация реплик прямой речи по персонажам (говорящим).

Ключевая идея: эмбеддинг реплики считается С УЧЁТОМ СОСЕДНИХ сегментов —
указание на говорящего ("сказала Е.", имена) обычно стоит в соседнем
авторском тексте, а не внутри самой реплики.

  emb(реплика) = norm( e(реплика) + alpha * mean(e(соседи в окне)) )

Источник: размеченный .jsonl (формат exports/, с тегами <A>/<D>).

Примеры:
  venv/bin/python speaker_cluster.py exports/Kuprin_Granatovyy-braslet.bNTGLw.858123_annotated.jsonl --k 4
  venv/bin/python speaker_cluster.py exports/1.jsonl --window 2 --alpha 0.7
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer

TAG_RE = re.compile(r"</?[AD]>")
MODEL_NAME = "cointegrated/rubert-tiny2"


def parse_lines(path: Path) -> list[list[tuple[str, str]]]:
    """Разбор по строкам: список абзацев, каждый — список сегментов (label, text)."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        line = json.loads(raw).get("text", "")
        if not line.strip():
            continue
        row, pos, cur = [], 0, "A"  # текст до первого тега считаем авторским
        for m in TAG_RE.finditer(line):
            txt = line[pos:m.start()].strip()
            if txt:
                row.append((cur, txt))
            tag = m.group()
            if tag == "<A>":
                cur = "A"
            elif tag == "<D>":
                cur = "D"
            else:  # </A> или </D> — возврат к авторской речи
                cur = "A"
            pos = m.end()
        txt = line[pos:].strip()
        if txt:
            row.append((cur, txt))
        if row:
            lines.append(row)
    return lines


def parse_segments(path: Path) -> list[tuple[str, str]]:
    """Плоская последовательность сегментов (label, text) в порядке документа."""
    return [seg for line in parse_lines(path) for seg in line]


def nearest_attribution(seq, i) -> str:
    """Ближайший авторский сегмент к реплике i (сначала справа, потом слева)."""
    for j in (i + 1, i - 1):
        if 0 <= j < len(seq) and seq[j][0] == "A":
            return seq[j][1]
    return ""


def context_embeddings(seq, d_idx, model, window=1, alpha=0.5, progress=False):
    """Эмбеддинг каждой реплики с учётом соседей: e(реплика) + alpha*mean(e(соседи))."""
    seq_emb = model.encode([t for _, t in seq], normalize_embeddings=True,
                           show_progress_bar=progress)
    vecs = []
    for i in d_idx:
        ctx = [seq_emb[j] for j in range(i - window, i + window + 1)
               if j != i and 0 <= j < len(seq)]
        v = seq_emb[i].copy()
        if ctx and alpha > 0:
            v = v + alpha * np.mean(ctx, axis=0)
        v = v / (np.linalg.norm(v) + 1e-9)
        vecs.append(v)
    return np.vstack(vecs)


def choose_k(vecs, k_max=10):
    """Подобрать число кластеров по максимуму silhouette."""
    best_k, best = 2, -1.0
    for kk in range(2, min(k_max, len(vecs) - 1) + 1):
        lab = KMeans(n_clusters=kk, n_init=10, random_state=42).fit_predict(vecs)
        sc = silhouette_score(vecs, lab, metric="cosine")
        if sc > best:
            best_k, best = kk, sc
    return best_k, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="размеченный .jsonl с тегами <A>/<D>")
    ap.add_argument("--k", type=int, default=None, help="число персонажей (по умолчанию авто)")
    ap.add_argument("--window", type=int, default=1, help="сколько соседних сегментов с каждой стороны")
    ap.add_argument("--alpha", type=float, default=0.5, help="вес контекста соседей (0 = без них)")
    ap.add_argument("--examples", type=int, default=5, help="реплик на кластер в выводе")
    ap.add_argument("--out", help="сохранить результат в JSON")
    args = ap.parse_args()

    seq = parse_segments(Path(args.input))
    d_idx = [i for i, (lab, _) in enumerate(seq) if lab == "D"]
    print(f"Сегментов: {len(seq)}, реплик прямой речи: {len(d_idx)}")
    if len(d_idx) < 4:
        print("Слишком мало реплик для кластеризации.")
        return

    print(f"Эмбеддинги ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    vecs = context_embeddings(seq, d_idx, model, args.window, args.alpha, progress=True)

    # Число кластеров
    if args.k:
        k = args.k
    else:
        k, best = choose_k(vecs)
        print(f"Выбрано k={k} (silhouette={best:.3f})")

    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vecs)

    clusters = {}
    for n, i in enumerate(d_idx):
        clusters.setdefault(int(labels[n]), []).append(
            {"speech": seq[i][1], "attribution": nearest_attribution(seq, i)}
        )

    print("\n" + "=" * 72)
    for lab in sorted(clusters, key=lambda c: -len(clusters[c])):
        group = clusters[lab]
        print(f"\nПЕРСОНАЖ / КЛАСТЕР {lab}  ({len(group)} реплик)")
        for item in group[:args.examples]:
            attr = f"   ← {item['attribution'][:45]}" if item["attribution"] else ""
            print(f"  • {item['speech'][:75]}{attr}")
        if len(group) > args.examples:
            print(f"  … ещё {len(group) - args.examples}")

    if args.out:
        out = {str(lab): clusters[lab] for lab in clusters}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()
