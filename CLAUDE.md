# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Russian NLP project: token-level classification of literary text into **author speech** vs **direct speech** (fine-tuned ruBERT), plus embedding-based clustering of direct-speech lines by speaker/character. All UI text, docstrings, and target texts are in Russian.

There is no test suite, linter, or requirements.txt. Python is 3.9 (`venv/`); always run scripts with `venv/bin/python` from the project root (scripts use relative paths like `models/...`, `exports/`, `data/`).

## Data Pipeline

The scripts form a sequential pipeline around a shared tag format — `<A>...</A>` (author) and `<D>...</D>` (direct speech) embedded in JSONL `{"text": ...}` lines:

1. **annotator.py** — Flask web annotation tool (port 5002). Drafts labels via punctuation heuristics («…» quotes, dash-dialogues) or the trained model, user corrects by hand, saves to `exports/<name>_annotated.jsonl`.
2. **preprocess.py** — converts all `exports/*.jsonl` into token-level BIO datasets `data/train.json` / `data/val.json` (90/10 split, seed 42). Tags may open on one line and close on a later one; `load_jsonl` threads open-tag state across lines — preserve this if editing.
3. **train.py** — fine-tunes `DeepPavlov/rubert-base-cased` for token classification (labels `O`, `B/I-AUTHOR`, `B/I-DIRECT`), saves best checkpoint to `models/rubert-speech/best`. Device priority: CUDA → MPS → CPU; controlled by `SPEECH_DEVICE=auto|cuda|mps|cpu` (default `auto`). `fp16` and `pin_memory` auto-enable on CUDA.
4. **predict.py** — CLI inference; also provides `load_model()`/`predict()` imported by the web apps. Annotates line-by-line (matches per-line training distribution) and reconstructs `<A>`/`<D>` tagged output.
5. **app.py** — Flask demo (port 5001) that renders model predictions as colored HTML spans.

Clustering tools (independent of the trained model, use `cointegrated/rubert-tiny2` sentence embeddings on CPU):
- **cluster.py** — KMeans topic clustering of sentences; k auto-picked by silhouette score.
- **speaker_cluster.py** — clusters direct-speech lines by speaker. Key idea: a line's embedding is blended with neighboring segments (`emb + alpha * mean(neighbors)`) because speaker attribution ("сказала Е.") lives in adjacent author text, not the line itself.
- **speaker_web.py** — Flask visualization of speaker clustering (port 5003), imports from speaker_cluster.py.
- **speaker_encoder.py** — supervised upgrade of speaker_cluster.py: fine-tunes a sentence encoder (`BASE_MODEL`, currently `ai-forever/sbert_large_nlu_ru`) with triplet loss on a speaker-labeled dataset so same-speaker lines embed close. Input is a single text `"{реплика} [SEP] {context window}"` — the same representation at train and inference (`--window` must match). Triplets are mined within one document only (cross-document negatives teach topic, not speaker). Dataset format: flat JSONL of ordered segments `{"doc", "label": "A"|"D", "text", "speaker" (required for D)}` — see the module docstring.

The label vocabulary `LABEL2ID` is duplicated in preprocess.py and train.py — keep them in sync.

## Commands

```bash
venv/bin/python preprocess.py                  # rebuild data/ from exports/
venv/bin/python train.py                       # train (CUDA→MPS→CPU auto-pick, ~4.75GB GPU, ~6 min)
venv/bin/python predict.py "Текст..."          # inference; also --file/--output/--model
venv/bin/python app.py                         # demo UI       → :5001
venv/bin/python annotator.py                   # annotation UI → :5002
venv/bin/python speaker_web.py                 # speaker map   → :5003
venv/bin/python speaker_cluster.py exports/<f>.jsonl --k 4 --window 2 --alpha 0.7
venv/bin/python speaker_encoder.py train data/speakers.jsonl --val-docs <doc>   # metric-learning эмбеддера реплик
venv/bin/python speaker_encoder.py cluster exports/<f>.jsonl                    # кластеризация обученной моделью
venv/bin/python cluster.py exports/<f>.jsonl --k 5 --out clusters.json
```

## Training Memory Constraints (MPS)

train.py picks **CUDA → MPS → CPU** (override with `SPEECH_DEVICE=cuda|mps|cpu`). On Apple Silicon (24GB machine, hard no-swap requirement), MPS memory stability rests on invariants that look like ordinary tuning but are load-bearing — do not "optimize" them away:

- **Fixed-shape batches**: every sample is padded to exactly `MAX_LENGTH` (256) and batched with `default_data_collator`. Dynamic padding (variable batch shapes) fragments the MPS caching allocator — it held 10GB while using 4.5GB and `empty_cache()` couldn't reclaim it, OOMing far below physical RAM. This is why the script does NOT use `DataCollatorForTokenClassification`.
- **Watermark env vars** (`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7`, `LOW=0.5`) are set at the top of the file **before `import torch`** — the default high watermark (1.7× recommended max ≈ 30GB) silently swaps instead of failing. Keep the `os.environ` lines above the torch import.
- Gradient checkpointing and periodic `torch.mps.empty_cache()` (`MPSCacheCallback`) keep the peak flat (~4.75GB for the whole run, ~6 min for 10 epochs).

Inference (predict.py) is lightweight and uses MPS freely. Note: transformers ≥4.57 removed `Trainer(tokenizer=...)`; use `processing_class=`.
