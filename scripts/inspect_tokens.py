import argparse
import getpass
import http.server
import os
import re
import socket
import socketserver
import threading
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy as sp
import shap
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


MAX_TOKENS = 250
LABEL_NAMES = {0: "FAKE", 1: "TRUE"}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}


@dataclass
class PredictionResult:
    label: int
    confidence: float
    probabilities: np.ndarray


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def clean_token(token: str) -> str:
    text = str(token)
    for marker in ("##", "Ġ", "▁"):
        text = text.replace(marker, " ")
    text = text.strip()
    if not text or text in SPECIAL_TOKENS:
        return ""
    return text


@dataclass
class LoadedSample:
    text: str
    true_label: Optional[int]


class TokenInspector:
    def __init__(self, model_dir: str, batch_size: int):
        self.batch_size = batch_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()

        def predict_fn(texts):
            probs = self.get_probs(texts)
            return sp.special.logit(probs[:, 0])

        self.predict_fn = predict_fn
        self.explainer = shap.Explainer(self.predict_fn, self.tokenizer)

    def get_probs(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(text) for text in texts]
        all_probs = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=MAX_TOKENS,
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
        return np.vstack(all_probs)

    def predict_text(self, text: str) -> PredictionResult:
        probs = self.get_probs([text])[0]
        label = int(np.argmax(probs))
        return PredictionResult(
            label=label,
            confidence=float(probs[label]),
            probabilities=probs,
        )

    def explain_text(self, text: str):
        return self.explainer([text], fixed_context=1)

    def extract_ranked_tokens(self, shap_values) -> List[dict]:
        raw_tokens = shap_values.data[0]
        raw_values = shap_values.values[0]

        if hasattr(raw_tokens, "tolist"):
            raw_tokens = raw_tokens.tolist()
        else:
            raw_tokens = list(raw_tokens)

        if hasattr(raw_values, "tolist"):
            raw_values = raw_values.tolist()

        merged_rows = []
        for token, value in zip(raw_tokens, raw_values):
            token_text = str(token).strip()
            cleaned = clean_token(token_text)
            if not cleaned:
                continue

            numeric_value = float(value)
            is_wordpiece_continuation = token_text.startswith("##")

            if is_wordpiece_continuation and merged_rows:
                merged_rows[-1]["token"] += cleaned
                merged_rows[-1]["value"] += numeric_value
                merged_rows[-1]["abs_value"] = abs(merged_rows[-1]["value"])
                merged_rows[-1]["direction"] = (
                    "-> pushes toward FAKE"
                    if merged_rows[-1]["value"] >= 0
                    else "-> pushes toward TRUE"
                )
                continue

            direction = "-> pushes toward FAKE" if numeric_value >= 0 else "-> pushes toward TRUE"
            merged_rows.append(
                {
                    "token": cleaned,
                    "value": numeric_value,
                    "abs_value": abs(numeric_value),
                    "direction": direction,
                }
            )

        merged_rows.sort(key=lambda row: row["abs_value"], reverse=True)
        return merged_rows


def load_text(args: argparse.Namespace) -> LoadedSample:
    if args.text:
        return LoadedSample(text=normalize_text(args.text), true_label=None)

    df = pd.read_csv(args.test_csv, usecols=["statement", "label"])
    if args.sample_idx < 0 or args.sample_idx >= len(df):
        raise IndexError(f"--sample_idx {args.sample_idx} is out of range for {len(df)} rows.")
    row = df.iloc[args.sample_idx]
    return LoadedSample(text=normalize_text(row["statement"]), true_label=int(row["label"]))


def print_prediction(prediction: PredictionResult, prefix: str):
    print(f"{prefix} prediction: {LABEL_NAMES[prediction.label]} ({prediction.confidence:.4f})")
    print(f"{prefix} probabilities: FAKE={prediction.probabilities[0]:.4f}, TRUE={prediction.probabilities[1]:.4f}")


def compute_text_statistics(
    tokenizer,
    text: str,
) -> Tuple[int, int, int]:
    whitespace_tokens = text.split()
    word_count = sum(1 for token in whitespace_tokens if re.search(r"\w", token))

    wordpiece_tokens = tokenizer.tokenize(text)
    wordpiece_count = len(wordpiece_tokens) + 2  # [CLS] and [SEP]

    subword_splits = 0
    for token in whitespace_tokens:
        if not re.search(r"\w", token):
            continue
        pieces = tokenizer.tokenize(token)
        if any("##" in piece for piece in pieces):
            subword_splits += 1

    return word_count, wordpiece_count, subword_splits


def print_text_statistics(
    tokenizer,
    text: str,
    prediction: PredictionResult,
    true_label: Optional[int],
):
    word_count, wordpiece_count, subword_splits = compute_text_statistics(
        tokenizer=tokenizer,
        text=text,
    )
    true_label_text = LABEL_NAMES[true_label] if true_label is not None else "N/A"

    print("═" * 51)
    print("TEXT STATISTICS")
    print("═" * 51)
    print(f'  Original text     : "{text}"')
    print(f"  Word count        : {word_count}")
    print(f"  WordPiece tokens  : {wordpiece_count}  (incl. [CLS] and [SEP])")
    print(f"  Subword splits    : {subword_splits}   (words split into multiple tokens)")
    print(f"  Prediction        : {LABEL_NAMES[prediction.label]} ({prediction.confidence:.4f})")
    print(f"  True label        : {true_label_text}")
    print("═" * 51)


def print_token_table(rows: List[dict], limit: int = 20):
    print("\nRank  Token                      SHAP Value   Direction")
    print("-" * 72)
    for rank, row in enumerate(rows[:limit], start=1):
        token = f'"{row["token"]}"'
        print(f"{rank:<5} {token:<26} {row['value']:+.3f}      {row['direction']}")


def save_shap_html(shap_values, html_out: str):
    os.makedirs(os.path.dirname(html_out) or ".", exist_ok=True)
    try:
        html_fragment = shap.plots.text(shap_values, display=False)
    except ImportError as exc:
        raise ImportError(
            "Saving the SHAP HTML visualization requires matplotlib. "
            "Install it with `pip install matplotlib` or `pip install -r requirements.txt`."
        ) from exc
    if not isinstance(html_fragment, str):
        html_fragment = str(html_fragment)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SHAP Token Inspection</title>
  {shap.getjs()}
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      background: #ffffff;
      color: #111111;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
    }}
    h1 {{
      font-size: 24px;
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <h1>SHAP Text Explanation</h1>
    {html_fragment}
  </div>
</body>
</html>
"""
    with open(html_out, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"Saved SHAP HTML to {os.path.abspath(html_out)}")


def replace_first_token(text: str, old: str, new: str) -> str:
    pattern = re.compile(rf"\b{re.escape(old)}\b", flags=re.IGNORECASE)
    updated, count = pattern.subn(new, text, count=1)
    if count == 0:
        updated = text.replace(old, new, 1)
    return normalize_text(updated)


def start_html_server(html_out: str, host: str, port: int):
    serve_dir = os.path.abspath(os.path.dirname(html_out) or ".")
    handler = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(  # noqa: E731
        *args,
        directory=serve_dir,
        **kwargs,
    )

    class ReusableTCPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    server = ReusableTCPServer((host, port), handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def print_browser_instructions(html_out: str, host: str, port: int):
    file_name = os.path.basename(html_out)
    remote_url = f"http://{host}:{port}/{file_name}"
    local_url = f"http://localhost:{port}/{file_name}"
    hostname = socket.gethostname()
    user = getpass.getuser()

    print("\nSHAP viewer")
    print("=" * 72)
    print(f"Remote server: {remote_url}")
    print(f"Local browser URL after tunnel: {local_url}")
    print("From your laptop, run:")
    print(f"ssh -N -L {port}:127.0.0.1:{port} {user}@{hostname}")
    print(f"Then open: {local_url}")
    print("Refresh the page after each token replacement to see the updated SHAP view.")


def inspect_once(
    inspector: TokenInspector,
    text: str,
    html_out: str,
    prefix: str,
    true_label: Optional[int],
):
    prediction = inspector.predict_text(text)
    print(f"\nText:\n{text}\n")
    print_prediction(prediction, prefix=prefix)
    shap_values = inspector.explain_text(text)
    ranked_tokens = inspector.extract_ranked_tokens(shap_values)
    print_text_statistics(
        tokenizer=inspector.tokenizer,
        text=text,
        prediction=prediction,
        true_label=true_label,
    )
    print_token_table(ranked_tokens)
    save_shap_html(shap_values, html_out)
    return prediction, shap_values, ranked_tokens


def interactive_flip_loop(
    inspector: TokenInspector,
    original_text: str,
    html_out: str,
    true_label: Optional[int],
):
    current_text = original_text
    history = []

    while True:
        token_to_replace = input("\nEnter a token to replace (or press Enter to skip): ").strip()
        if token_to_replace.lower() == "quit":
            break
        if token_to_replace == "":
            break

        replacement = input("Replace with: ").strip()
        if replacement.lower() == "quit":
            break

        previous_prediction = inspector.predict_text(current_text)
        updated_text = replace_first_token(current_text, token_to_replace, replacement)
        if updated_text == current_text:
            print("No change made; token not found.")
            continue

        current_text = updated_text
        new_prediction, _, _ = inspect_once(
            inspector,
            current_text,
            html_out,
            prefix="Modified",
            true_label=true_label,
        )
        history.append(
            {
                "replace": token_to_replace,
                "with": replacement,
                "before_label": previous_prediction.label,
                "before_conf": previous_prediction.confidence,
                "after_label": new_prediction.label,
                "after_conf": new_prediction.confidence,
                "text": current_text,
            }
        )

        keep_going = input("Flip another token? [y/N or quit]: ").strip().lower()
        if keep_going in {"quit", "n", "no", ""}:
            break

    return current_text, history


def print_summary(history: List[dict], initial_prediction: PredictionResult, final_prediction: PredictionResult):
    print("\nSummary")
    print("=" * 72)
    print(
        f"Initial prediction: {LABEL_NAMES[initial_prediction.label]} ({initial_prediction.confidence:.4f})"
    )
    print(
        f"Final prediction:   {LABEL_NAMES[final_prediction.label]} ({final_prediction.confidence:.4f})"
    )
    if not history:
        print("No replacements were made.")
        return

    for step, item in enumerate(history, start=1):
        before_label = LABEL_NAMES[item["before_label"]]
        after_label = LABEL_NAMES[item["after_label"]]
        print(
            f"{step}. \"{item['replace']}\" -> \"{item['with']}\" | "
            f"{before_label} ({item['before_conf']:.4f}) -> {after_label} ({item['after_conf']:.4f})"
        )


def main(args: argparse.Namespace):
    sample = load_text(args)
    text = sample.text
    inspector = TokenInspector(args.model_dir, args.batch_size)
    server, _ = start_html_server(args.html_out, args.serve_host, args.serve_port)

    print(f"Using device: {inspector.device}")
    print_browser_instructions(args.html_out, args.serve_host, args.serve_port)
    try:
        initial_prediction, _, _ = inspect_once(
            inspector,
            text,
            args.html_out,
            prefix="Original",
            true_label=sample.true_label,
        )
        final_text, history = interactive_flip_loop(
            inspector,
            text,
            args.html_out,
            true_label=sample.true_label,
        )
        final_prediction = inspector.predict_text(final_text)
        print_summary(history, initial_prediction, final_prediction)
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactively inspect SHAP token importance and test manual token flips."
    )
    parser.add_argument("--model_dir", required=True, help="Directory containing the trained model and tokenizer")
    parser.add_argument("--test_csv", required=True, help="CSV with statement and label columns")
    parser.add_argument("--sample_idx", type=int, default=0, help="Row index in the CSV to inspect")
    parser.add_argument("--text", default=None, help="Optional raw text to inspect instead of a CSV sample")
    parser.add_argument(
        "--html_out",
        default="shap_inspection.html",
        help="Path to save the SHAP HTML visualization",
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--serve_host", default="127.0.0.1", help="Host for the local HTML server")
    parser.add_argument("--serve_port", type=int, default=8765, help="Port for the local HTML server")
    main(parser.parse_args())
