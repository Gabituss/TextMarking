"""
Inference: разметить текст на авторскую и прямую речь.
Usage: python predict.py "Текст для разметки"
       python predict.py --file input.txt
"""

import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

MODEL_DIR = "models/rubert-speech/best"


def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()
    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    model.to(device)
    return tokenizer, model, device


def predict(text: str, tokenizer, model, device) -> list[tuple[str, str]]:
    words = text.split()
    enc = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    preds = logits.argmax(-1)[0].cpu().tolist()
    word_ids = enc.word_ids()
    id2label = model.config.id2label

    result = []
    seen = set()
    for pred, wid in zip(preds, word_ids):
        if wid is None or wid in seen:
            continue
        seen.add(wid)
        label = id2label[pred]
        # Normalize B-/I- → category
        category = label.split("-")[-1] if label != "O" else "O"
        result.append((words[wid], category))

    return result


def format_output(pairs: list[tuple[str, str]]) -> str:
    """Reconstruct text with XML-style tags."""
    out = []
    prev_cat = None
    for word, cat in pairs:
        if cat != prev_cat:
            if prev_cat and prev_cat != "O":
                tag = "A" if prev_cat == "AUTHOR" else "D"
                out.append(f"</{tag}>")
            if cat != "O":
                tag = "A" if cat == "AUTHOR" else "D"
                out.append(f"<{tag}>")
        out.append(word)
        prev_cat = cat
    if prev_cat and prev_cat != "O":
        tag = "A" if prev_cat == "AUTHOR" else "D"
        out.append(f"</{tag}>")
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", help="Text to annotate")
    parser.add_argument("--file", help="Input text file")
    parser.add_argument("--output", help="Output file (default: stdout)")
    parser.add_argument("--model", default=MODEL_DIR)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    print(f"Loaded model from {args.model} (device: {device})")

    if args.file:
        text = Path(args.file).read_text()
    elif args.text:
        text = args.text
    else:
        print("Enter text (Ctrl+D to finish):")
        text = input()

    # Annotate line by line so newlines are preserved and each line matches
    # the per-line distribution the model was trained on. Blank lines pass through.
    annotated_lines = []
    for line in text.splitlines():
        if line.strip():
            pairs = predict(line, tokenizer, model, device)
            annotated_lines.append(format_output(pairs))
        else:
            annotated_lines.append(line)
    output = "\n".join(annotated_lines)

    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)


if __name__ == "__main__":
    main()
