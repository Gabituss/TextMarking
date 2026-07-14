"""
Кластеризация предложений по смыслу.

Источник: .jsonl (формат exports/, теги <A>/<D> снимаются) или .txt (по строкам).
Эмбеддинги: sentence-transformers (sergeyzh/rubert-tiny2, русская модель под similarity).
Кластеры: KMeans; число кластеров k подбирается по silhouette, если не задано.

Примеры:
  venv/bin/python cluster.py exports/Kuprin_Granatovyy-braslet.bNTGLw.858123_annotated.jsonl
  venv/bin/python cluster.py mytext.txt --k 5
  venv/bin/python cluster.py exports/1.jsonl --out clusters.json
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


def load_sentences(path: Path) -> list[str]:
    """Прочитать предложения из .jsonl (поле text, без тегов) или .txt (по строкам)."""
    sents = []
    if path.suffix == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            text = json.loads(line).get("text", "")
            text = TAG_RE.sub("", text).strip()
            if text:
                sents.append(text)
    else:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sents.append(line.strip())
    return sents


def pick_k(emb: np.ndarray, k_min=2, k_max=10) -> int:
    """Подобрать число кластеров по максимуму silhouette."""
    n = len(emb)
    k_max = min(k_max, n - 1)
    best_k, best_score = k_min, -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)
        score = silhouette_score(emb, labels, metric="cosine")
        print(f"  k={k}: silhouette={score:.3f}")
        if score > best_score:
            best_k, best_score = k, score
    print(f"Выбрано k={best_k} (silhouette={best_score:.3f})")
    return best_k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="путь к .jsonl или .txt")
    ap.add_argument("--k", type=int, default=None, help="число кластеров (по умолчанию авто)")
    ap.add_argument("--out", help="сохранить результат в JSON")
    ap.add_argument("--examples", type=int, default=4, help="сколько примеров на кластер")
    args = ap.parse_args()

    path = Path(args.input)
    sents = load_sentences(path)
    print(f"Загружено предложений: {len(sents)}")
    if len(sents) < 4:
        print("Слишком мало предложений для кластеризации.")
        return

    print(f"Эмбеддинги ({MODEL_NAME})...")
    # CPU: модель крошечная и быстрая, заодно нет предупреждений MPS-аллокатора
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    emb = model.encode(sents, normalize_embeddings=True, show_progress_bar=True)

    k = args.k or pick_k(emb)
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb)

    # Группируем и сортируем кластеры по размеру
    clusters = {}
    for sent, lab in zip(sents, labels):
        clusters.setdefault(int(lab), []).append(sent)

    print("\n" + "=" * 70)
    for lab in sorted(clusters, key=lambda c: -len(clusters[c])):
        group = clusters[lab]
        print(f"\nКЛАСТЕР {lab}  ({len(group)} предложений)")
        for s in group[:args.examples]:
            print(f"  • {s[:100]}")
        if len(group) > args.examples:
            print(f"  … ещё {len(group) - args.examples}")

    if args.out:
        out = {str(lab): clusters[lab] for lab in clusters}
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"\nСохранено в {args.out}")


if __name__ == "__main__":
    main()
