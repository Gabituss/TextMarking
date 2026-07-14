"""
Fine-tune ruBERT for Author/Direct speech token classification.
Runs on Apple Silicon GPU (MPS) by default; SPEECH_DEVICE=cpu to override.

Memory strategy for MPS (24GB machine, must never swap):
- Fixed-shape batches: every sample is padded to exactly MAX_LENGTH, so the
  MPS caching allocator reuses the same buffers instead of fragmenting on
  variable-size allocations (the fragmentation is what used to OOM this
  script well below physical RAM).
- Hard watermark on the allocator (set via env BEFORE torch import): the
  default high watermark is 1.7x recommended_max_memory (~30GB here), which
  silently spills into swap instead of failing.
- Gradient checkpointing + periodic empty_cache keep the transient peak flat.
"""

import os

# Must be set before torch initializes the MPS allocator.
# High watermark: hard ceiling as a fraction of recommended_max_memory (~18GB
# on 24GB RAM) — allocations beyond it fail loudly instead of swapping.
# Low watermark: where the allocator starts releasing cached buffers.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.7")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    default_data_collator,
)
import evaluate

MODEL_NAME = "DeepPavlov/rubert-base-cased"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("models/rubert-speech")

# Fixed sequence length for ALL samples (covers p99 of the data; longer ones
# are truncated). Uniform shapes are load-bearing for MPS memory stability —
# do not switch back to dynamic padding.
MAX_LENGTH = 256

# Flush the MPS cached-buffer pool every N optimizer steps so it can't
# balloon toward the watermark.
EMPTY_CACHE_EVERY_N_STEPS = 25

LABEL2ID = {"O": 0, "B-AUTHOR": 1, "I-AUTHOR": 2, "B-DIRECT": 3, "I-DIRECT": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class SpeechDataset(Dataset):
    def __init__(self, path: Path, tokenizer):
        with open(path) as f:
            raw = json.load(f)
        self.encodings = []
        for sample in raw:
            enc = tokenize_and_align(sample["tokens"], sample["labels"], tokenizer)
            if enc:
                self.encodings.append(enc)

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        return self.encodings[idx]


def tokenize_and_align(tokens, labels, tokenizer, max_length=MAX_LENGTH):
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    word_ids = enc.word_ids()
    aligned_labels = []
    prev_word = None
    for wid in word_ids:
        if wid is None:
            aligned_labels.append(-100)
        elif wid != prev_word:
            aligned_labels.append(LABEL2ID[labels[wid]])
        else:
            # subword: convert B- → I- for continuation
            lbl = labels[wid]
            if lbl.startswith("B-"):
                lbl = "I-" + lbl[2:]
            aligned_labels.append(LABEL2ID[lbl])
        prev_word = wid

    enc["labels"] = aligned_labels
    return {k: torch.tensor(v) for k, v in enc.items()}


def compute_metrics(eval_pred):
    metric = evaluate.load("seqeval")
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    true_labels, true_preds = [], []
    for pred_row, label_row in zip(preds, labels):
        true_labels.append([ID2LABEL[l] for l in label_row if l != -100])
        true_preds.append([
            ID2LABEL[p] for p, l in zip(pred_row, label_row) if l != -100
        ])

    result = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": result["overall_precision"],
        "recall": result["overall_recall"],
        "f1": result["overall_f1"],
    }


class MPSCacheCallback(TrainerCallback):
    """Release MPS cached buffers periodically and report driver memory."""

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available() and state.global_step % EMPTY_CACHE_EVERY_N_STEPS == 0:
            torch.mps.empty_cache()
            driver = torch.mps.driver_allocated_memory() / 1e9
            print(f"[mem] step {state.global_step}: driver_allocated={driver:.2f} GB", flush=True)

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def pick_device() -> str:
    requested = os.environ.get("SPEECH_DEVICE", "mps").lower()
    if requested == "mps" and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    device = pick_device()
    print(f"Device: {device}")
    if device == "mps":
        recommended = torch.mps.recommended_max_memory() / 1e9
        high = float(os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"])
        print(
            f"MPS recommended_max_memory: {recommended:.1f} GB | "
            f"allocator capped at {recommended * high:.1f} GB"
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = SpeechDataset(DATA_DIR / "train.json", tokenizer)
    val_ds = SpeechDataset(DATA_DIR / "val.json", tokenizer)
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        fp16=False,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        use_cpu=(device == "cpu"),
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=default_data_collator,
        compute_metrics=compute_metrics,
        callbacks=[MPSCacheCallback()],
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR / "best"))
    tokenizer.save_pretrained(str(OUTPUT_DIR / "best"))
    print(f"\nModel saved to {OUTPUT_DIR}/best")


if __name__ == "__main__":
    main()
