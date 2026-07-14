"""
Обучение эмбеддера реплик: реплики одного персонажа — близко, разных — далеко.

Вход модели — реплика вместе с соседними сегментами одним текстом:
    "{реплика} [SEP] {соседи слева и справа в окне}"
Атрибуция («сказала Е.», имена) обычно стоит в соседнем авторском тексте,
поэтому энкодер учится смотреть и в контекст. На инференсе строим тот же
текст и кластеризуем эмбеддинги (KMeans, k — по silhouette, если не задан).

Лосс: triplet по косинусу. Позитив — другая реплика того же персонажа,
негатив — реплика другого персонажа ИЗ ТОГО ЖЕ документа: кросс-документные
негативы различимы по теме и учат модель не тому.

Дообучение через LoRA-адаптеры (query/value, ранг 16): обучается ~1-2M
параметров вместо 400M. После тренировки адаптеры вливаются в базовые
веса (merge_and_unload) — модель сохраняется и грузится как обычная
SentenceTransformer, peft на инференсе не нужен.

Формат обучающего датасета — JSONL, одна строка = один сегмент,
строго в порядке следования в документе:
    {"doc": "kasha", "label": "A", "text": "Солдат пришёл к старухе и говорит:"}
    {"doc": "kasha", "label": "D", "text": "Дай-ка поесть!", "speaker": "солдат"}
  - label: "A" (авторский текст) или "D" (прямая речь)
  - speaker: имя/ID персонажа, обязателен для "D" (уникален в пределах doc)
  - doc: имя документа (триплеты и оценка — только внутри документа)

Примеры:
  venv/bin/python speaker_encoder.py train data/speakers.jsonl --val-docs kasha
  venv/bin/python speaker_encoder.py cluster exports/1.jsonl --k 4
  venv/bin/python speaker_encoder.py cluster exports/1.jsonl --model models/speaker-encoder
"""

import os

# Потолки аллокатора MPS — до импорта torch (см. train.py: дефолт 1.7x уводит в своп)
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sentence_transformers import SentenceTransformer, InputExample, losses
from peft import LoraConfig, get_peft_model, TaskType

from speaker_cluster import parse_segments, nearest_attribution, choose_k

BASE_MODEL = "ai-forever/sbert_large_nlu_ru"
OUTPUT_DIR = "models/speaker-encoder"
MAX_SEQ_LENGTH = 256
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1


def pick_device() -> str:
    requested = os.environ.get("SPEECH_DEVICE", "mps").lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------- данные ----------

def load_docs(path: Path) -> dict[str, list[dict]]:
    """JSONL сегментов → {doc: [сегменты в порядке документа]}."""
    docs = defaultdict(list)
    for n, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        seg = json.loads(raw)
        if seg.get("label") not in ("A", "D"):
            raise ValueError(f"{path}:{n}: label должен быть 'A' или 'D'")
        if seg["label"] == "D" and not seg.get("speaker"):
            raise ValueError(f"{path}:{n}: у реплики (label='D') нет speaker")
        docs[seg.get("doc", path.stem)].append(seg)
    return dict(docs)


WORD_CHAR_RE = re.compile(r"\w")


def build_input(seq: list[tuple[str, str]], i: int, window: int) -> str:
    """Текст для энкодера: реплика + соседние сегменты в окне.

    Сегменты без единой буквы/цифры (обрывки пунктуации вроде «». – «»
    из разметки exports/) атрибуции не несут и слоты окна не занимают —
    иначе они вытесняют из контекста реальный авторский текст.
    """
    idx = [j for j in range(len(seq)) if j == i or WORD_CHAR_RE.search(seq[j][1])]
    pos = idx.index(i)
    left = " ".join(seq[j][1] for j in idx[max(0, pos - window):pos])
    right = " ".join(seq[j][1] for j in idx[pos + 1:pos + 1 + window])
    ctx = f"{left} {right}".strip()
    return f"{seq[i][1]} [SEP] {ctx}" if ctx else seq[i][1]


def load_meta(model_dir: str) -> dict:
    """Параметры обучения, сохранённые рядом с моделью (окно контекста и т.п.)."""
    p = Path(model_dir) / "speaker_encoder_meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def doc_to_seq(segs: list[dict]) -> tuple[list[tuple[str, str]], dict[int, str]]:
    """Сегменты документа → (seq как в speaker_cluster, {индекс реплики: персонаж})."""
    seq = [(s["label"], s["text"]) for s in segs]
    speakers = {i: s["speaker"] for i, s in enumerate(segs) if s["label"] == "D"}
    return seq, speakers


def make_triplets(docs: dict[str, list[dict]], window: int,
                  per_anchor: int = 3, seed: int = 42) -> list[InputExample]:
    """(якорь, позитив, негатив) — все из одного документа."""
    rng = random.Random(seed)
    examples = []
    for segs in docs.values():
        seq, speakers = doc_to_seq(segs)
        by_speaker = defaultdict(list)
        for i, sp in speakers.items():
            by_speaker[sp].append(i)

        for i, sp in speakers.items():
            positives = [j for j in by_speaker[sp] if j != i]
            negatives = [j for o, idxs in by_speaker.items() if o != sp for j in idxs]
            if not positives or not negatives:
                continue
            anchor = build_input(seq, i, window)
            for _ in range(per_anchor):
                examples.append(InputExample(texts=[
                    anchor,
                    build_input(seq, rng.choice(positives), window),
                    build_input(seq, rng.choice(negatives), window),
                ]))
    return examples


# ---------- оценка ----------

def eval_docs(model, docs: dict[str, list[dict]], window: int) -> None:
    """KMeans с истинным k на каждом документе → ARI / NMI против разметки."""
    scores = []
    for name, segs in docs.items():
        seq, speakers = doc_to_seq(segs)
        d_idx = sorted(speakers)
        gold = [speakers[i] for i in d_idx]
        k = len(set(gold))
        if k < 2 or len(d_idx) < k + 1:
            print(f"  {name}: пропуск (реплик {len(d_idx)}, персонажей {k})")
            continue
        vecs = model.encode([build_input(seq, i, window) for i in d_idx],
                            normalize_embeddings=True)
        pred = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(vecs)
        ari = adjusted_rand_score(gold, pred)
        nmi = normalized_mutual_info_score(gold, pred)
        scores.append((ari, nmi))
        print(f"  {name}: реплик {len(d_idx)}, персонажей {k} → ARI={ari:.3f}, NMI={nmi:.3f}")
    if scores:
        m = np.mean(scores, axis=0)
        print(f"  Среднее: ARI={m[0]:.3f}, NMI={m[1]:.3f}")


# ---------- команды ----------

def cmd_train(args):
    docs = load_docs(Path(args.data))
    val_names = {n for n in (args.val_docs or "").split(",") if n}
    unknown = val_names - docs.keys()
    if unknown:
        raise SystemExit(f"Нет таких документов в датасете: {', '.join(sorted(unknown))}")
    train_docs = {n: s for n, s in docs.items() if n not in val_names}
    val_docs = {n: s for n, s in docs.items() if n in val_names}

    n_reps = sum(1 for segs in train_docs.values() for s in segs if s["label"] == "D")
    print(f"Документов: {len(train_docs)} train / {len(val_docs)} val, "
          f"реплик в train: {n_reps}")

    triplets = make_triplets(train_docs, args.window, args.per_anchor)
    if not triplets:
        raise SystemExit("Не собралось ни одного триплета: нужны ≥2 реплики одного "
                         "персонажа и ≥1 другого в одном документе.")
    print(f"Триплетов: {len(triplets)}")

    device = pick_device()
    print(f"Устройство: {device} | базовая модель: {args.base}")
    model = SentenceTransformer(args.base, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH

    # LoRA-адаптеры на query/value: обучается ~1-2M параметров вместо 400M.
    # Градиенты и AdamW-состояния — только для адаптеров, память падает в разы.
    peft_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=["query", "value"],
    )
    model[0].auto_model = get_peft_model(model[0].auto_model, peft_cfg)
    model[0].auto_model.print_trainable_parameters()

    if device == "mps":
        # НЕ включать здесь gradient checkpointing: на MPS backward с ним
        # виснет (первый шаг не завершается за 10+ минут). Память экономим
        # только батчем — см. --batch-size.

        # Тот же инвариант, что в train.py: батчи фиксированной формы.
        # Динамический паддинг ST (padding=True → по самому длинному в батче)
        # фрагментирует кэширующий аллокатор MPS и приводит к OOM задолго до
        # реального исчерпания памяти. Паддим всё до MAX_SEQ_LENGTH.
        st_transformer = model[0]
        tokenizer = st_transformer.tokenizer

        def tokenize_fixed(texts, padding=True):
            return tokenizer(texts, padding="max_length", truncation=True,
                             max_length=MAX_SEQ_LENGTH, return_tensors="pt")

        st_transformer.tokenize = tokenize_fixed

    if val_docs:
        print("До обучения:")
        eval_docs(model, val_docs, args.window)
        if device == "mps":
            torch.mps.empty_cache()

    loader = DataLoader(triplets, shuffle=True, batch_size=args.batch_size)
    loss = losses.TripletLoss(
        model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=args.margin,
    )
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=args.epochs,
        warmup_steps=max(1, len(loader) * args.epochs // 10),
        show_progress_bar=True,
    )
    # Вливаем LoRA в базовые веса → автономная модель,
    # грузится через SentenceTransformer(path) без peft
    model[0].auto_model = model[0].auto_model.merge_and_unload()
    model.save(args.out)
    # Окно обязано совпадать на инференсе — сохраняем его рядом с моделью
    meta = {"window": args.window, "base": args.base,
            "margin": args.margin, "epochs": args.epochs}
    (Path(args.out) / "speaker_encoder_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"Модель сохранена в {args.out}")

    if val_docs:
        print("После обучения:")
        eval_docs(SentenceTransformer(args.out, device=device), val_docs, args.window)


def cmd_cluster(args):
    seq = parse_segments(Path(args.input))
    d_idx = [i for i, (lab, _) in enumerate(seq) if lab == "D"]
    print(f"Сегментов: {len(seq)}, реплик прямой речи: {len(d_idx)}")
    if len(d_idx) < 4:
        print("Слишком мало реплик для кластеризации.")
        return

    window = args.window if args.window is not None else load_meta(args.model).get("window", 2)
    device = pick_device()
    print(f"Модель: {args.model} ({device}), окно контекста: {window}")
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = MAX_SEQ_LENGTH
    vecs = model.encode([build_input(seq, i, window) for i in d_idx],
                        normalize_embeddings=True, show_progress_bar=True)

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="обучить эмбеддер на размеченном датасете")
    tr.add_argument("data", help="JSONL сегментов со speaker у реплик")
    tr.add_argument("--val-docs", help="имена документов для оценки, через запятую")
    tr.add_argument("--base", default=BASE_MODEL)
    tr.add_argument("--out", default=OUTPUT_DIR)
    tr.add_argument("--window", type=int, default=2, help="сегментов контекста с каждой стороны")
    tr.add_argument("--per-anchor", type=int, default=3, help="триплетов на реплику")
    tr.add_argument("--epochs", type=int, default=4)
    # TripletLoss кодирует 3 текста на пример → эффективный batch = batch_size × 3.
    # LoRA убирает ~4.8GB (градиенты + AdamW-состояния 400M параметров),
    # остаются активации — при seq=256, 24 слоях BERT-large 4×3=12 последовательностей
    # помещаются в потолок MPS 12.43GB. Поднимайте, если хватает памяти.
    tr.add_argument("--batch-size", type=int, default=4)
    tr.add_argument("--margin", type=float, default=0.25)
    tr.add_argument("--lora-r", type=int, default=LORA_R, help="ранг LoRA-адаптеров")
    tr.add_argument("--lora-alpha", type=int, default=LORA_ALPHA, help="масштаб LoRA")
    tr.set_defaults(func=cmd_train)

    cl = sub.add_parser("cluster", help="кластеризовать реплики обученной моделью")
    cl.add_argument("input", help="размеченный .jsonl с тегами <A>/<D> (формат exports/)")
    cl.add_argument("--model", default=OUTPUT_DIR, help="путь к обученному эмбеддеру")
    cl.add_argument("--k", type=int, default=None, help="число персонажей (по умолчанию авто)")
    cl.add_argument("--window", type=int, default=None,
                    help="по умолчанию берётся из модели (сохраняется при обучении)")
    cl.add_argument("--examples", type=int, default=5)
    cl.add_argument("--out", help="сохранить результат в JSON")
    cl.set_defaults(func=cmd_cluster)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
