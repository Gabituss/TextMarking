"""
Convert <A>...</A> / <D>...</D> tagged JSONL to token-level BIO dataset.
Outputs data/train.json and data/val.json (90/10 split).
"""

import json
import re
import random
from pathlib import Path

EXPORTS_DIR = Path("exports")
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LABEL2ID = {"O": 0, "B-AUTHOR": 1, "I-AUTHOR": 2, "B-DIRECT": 3, "I-DIRECT": 4}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

TAG_RE = re.compile(r"<(/?)([AD])>")


def parse_spans(text: str) -> list[tuple[int, int, str]]:
    """Return list of (start, end, label) char spans from tagged text."""
    spans = []
    clean_chars = []
    pos = 0
    tag_stack = []

    i = 0
    while i < len(text):
        m = TAG_RE.match(text, i)
        if m:
            closing, tag = m.group(1), m.group(2)
            label = "AUTHOR" if tag == "A" else "DIRECT"
            if not closing:
                tag_stack.append((label, len(clean_chars)))
            else:
                for j in range(len(tag_stack) - 1, -1, -1):
                    if tag_stack[j][0] == label:
                        _, span_start = tag_stack.pop(j)
                        spans.append((span_start, len(clean_chars), label))
                        break
            i = m.end()
        else:
            clean_chars.append(text[i])
            i += 1

    # Close any tags still open at end of text
    for label, span_start in tag_stack:
        spans.append((span_start, len(clean_chars), label))

    clean_text = "".join(clean_chars)
    return clean_text, spans


def tokenize_and_label(clean_text: str, spans: list[tuple[int, int, str]]) -> dict:
    """Split into word tokens and assign BIO labels."""
    # Find word boundaries
    tokens = []
    token_starts = []
    for m in re.finditer(r"\S+", clean_text):
        tokens.append(m.group())
        token_starts.append(m.start())

    if not tokens:
        return None

    # Build char-level label array from spans
    char_labels = ["O"] * len(clean_text)
    for start, end, label in spans:
        for c in range(start, end):
            if c < len(char_labels):
                char_labels[c] = label

    # Assign BIO label to each token by majority of its characters
    bio_labels = []
    for tok, start in zip(tokens, token_starts):
        end = start + len(tok)
        tok_char_labels = char_labels[start:end]
        non_o = [l for l in tok_char_labels if l != "O"]
        if non_o:
            dominant = max(set(non_o), key=non_o.count)
        else:
            dominant = "O"

        if dominant == "O":
            bio_labels.append("O")
        else:
            # B- if previous token had different label
            prev = bio_labels[-1] if bio_labels else "O"
            if prev in (f"B-{dominant}", f"I-{dominant}"):
                bio_labels.append(f"I-{dominant}")
            else:
                bio_labels.append(f"B-{dominant}")

    return {"tokens": tokens, "labels": bio_labels}


def load_jsonl(path: Path) -> list[dict]:
    """
    Tags can span multiple lines (opening tag on one line, closing on another).
    We thread open_tags state across lines so multi-line spans are captured.
    Each JSONL line becomes one sample, but labels are assigned using the
    cumulative tag context carried from previous lines.
    """
    samples = []
    open_tags: list[tuple[str, None]] = []  # tags opened but not yet closed

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if not text:
                continue

            # Inject synthetic opening tags for any tags still open from prev lines
            prefix = "".join(f"<{'A' if lbl == 'AUTHOR' else 'D'}>" for lbl, _ in open_tags)
            augmented = prefix + text

            clean_text, spans = parse_spans(augmented)

            # Update open_tags state by re-scanning original text for tag changes
            for m in TAG_RE.finditer(text):
                closing, tag = m.group(1), m.group(2)
                label = "AUTHOR" if tag == "A" else "DIRECT"
                if not closing:
                    open_tags.append((label, None))
                else:
                    for j in range(len(open_tags) - 1, -1, -1):
                        if open_tags[j][0] == label:
                            open_tags.pop(j)
                            break

            sample = tokenize_and_label(clean_text, spans)
            if sample and any(l != "O" for l in sample["labels"]):
                samples.append(sample)

    return samples


def main():
    all_samples = []
    for path in EXPORTS_DIR.glob("*.jsonl"):
        samples = load_jsonl(path)
        print(f"{path.name}: {len(samples)} labeled sentences")
        all_samples.extend(samples)

    print(f"\nTotal: {len(all_samples)} sentences")

    random.seed(42)
    random.shuffle(all_samples)

    split = int(len(all_samples) * 0.9)
    train, val = all_samples[:split], all_samples[split:]

    for name, data in [("train", train), ("val", val)]:
        out = DATA_DIR / f"{name}.json"
        with open(out, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(data)} samples → {out}")

    # Print label distribution
    from collections import Counter
    counter = Counter()
    for s in all_samples:
        counter.update(s["labels"])
    print("\nLabel distribution:", dict(counter))


if __name__ == "__main__":
    main()
