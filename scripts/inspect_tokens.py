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
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", ""}


@dataclass
class PredictionResult:
    label: int
    confidence: float
    logit: float
    probabilities: np.ndarray


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", str(text))


def collapse_spaces_around_punct(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?%)\]\}])", r"\1", text)
    text = re.sub(r"([(\[\{])\s+", r"\1", text)
    return text


def clean_token(token: str) -> str:
    text = str(token)
    for marker in ("##", "Ġ", "▁"):
        text = text.replace(marker, " ")
    text = text.strip()
    if not text or text in SPECIAL_TOKENS:
        return ""
    return text


def count_and_mask_boundary_special_tokens(shap_values) -> int:
    raw_tokens = shap_values.data[0]
    if hasattr(raw_tokens, "tolist"):
        raw_tokens = raw_tokens.tolist()
    else:
        raw_tokens = list(raw_tokens)

    masked_values = np.array(shap_values.values, copy=True)
    token_values = masked_values[0] if masked_values.ndim > 1 else masked_values

    masked_count = 0

    if raw_tokens:
        first_token = str(raw_tokens[0]).strip()
        if first_token == "[CLS]":
            token_values[0] = 0.0
            masked_count += 1

    sep_index = None
    for idx in range(len(raw_tokens) - 1, -1, -1):
        token = str(raw_tokens[idx]).strip()
        if token in {"[PAD]", "<pad>", ""}:
            continue
        sep_index = idx
        break

    if sep_index is not None and sep_index != 0:
        last_non_padding_token = str(raw_tokens[sep_index]).strip()
        if last_non_padding_token == "[SEP]":
            token_values[sep_index] = 0.0
            masked_count += 1

    for idx, token in enumerate(raw_tokens):
        if repr(token).strip() in ("''", '""', "' '"):
            token_values[idx] = 0.0
            masked_count += 1
            print(f"  Zeroed ghost token at position {idx}")

    if masked_values.ndim > 1:
        masked_values[0] = token_values
    else:
        masked_values = token_values

    shap_values.values = masked_values
    return masked_count


def debug_print_shap_tokens(shap_values):
    print("\nSHAP token debug")
    print("-" * 72)
    raw_tokens = shap_values.data[0].tolist()
    raw_values = shap_values.values[0]
    if raw_values.ndim == 2:
        vals = raw_values[:, 1]  # TRUE class
    else:
        vals = raw_values

    for i, (tok, val) in enumerate(zip(raw_tokens, vals)):
        print(f"  [{i:3d}] repr={repr(tok):<30} value={float(val):+.6f}")


def debug_print_top_shap_tokens(shap_values):
    raw_tokens = shap_values.data[0].tolist()
    raw_values = shap_values.values[0]
    if raw_values.ndim == 2:
        vals = raw_values[:, 1]
    else:
        vals = raw_values

    sorted_idx = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
    print("TOP 5 HIGHEST SHAP VALUE TOKENS:")
    for i in sorted_idx[:5]:
        tok = raw_tokens[i]
        print(
            f"  [{i}] type={type(tok).__name__} repr={repr(tok)} "
            f"len={len(str(tok))} value={float(vals[i]):+.6f}"
        )


def remove_ghost_tokens(shap_values):
    raw_tokens = list(shap_values.data[0])
    raw_values = shap_values.values[0]

    keep_idx = [i for i, tok in enumerate(raw_tokens) if len(str(tok).strip()) > 0]

    filtered_tokens = np.array([raw_tokens[i] for i in keep_idx])
    filtered_values = raw_values[keep_idx]

    return shap.Explanation(
        values=filtered_values[np.newaxis, :],
        data=np.array([filtered_tokens]),
        feature_names=filtered_tokens,
        base_values=shap_values.base_values,
    )


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
        prob_true = float(probs[1])
        return PredictionResult(
            label=label,
            confidence=float(probs[label]),
            logit=float(sp.special.logit(prob_true)),
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
    print(f"{prefix} confidence: FAKE={prediction.probabilities[0]:.4f}, TRUE={prediction.probabilities[1]:.4f}")


def format_signed(value: float) -> str:
    return f"{value:+.4f}"


def format_flip_flag(flipped: bool) -> str:
    return "YES ✓" if flipped else "NO  ✗"


def print_flip_summary(
    replaced_token: str,
    replacement_token: str,
    before_prediction: PredictionResult,
    after_prediction: PredictionResult,
):
    logit_shift = after_prediction.logit - before_prediction.logit
    flipped = before_prediction.label != after_prediction.label

    print("═" * 48)
    print("FLIP SUMMARY")
    print("═" * 48)
    print(f'  Replacement       : "{replaced_token}" -> "{replacement_token}"')
    print(
        f"  Prediction before : {LABEL_NAMES[before_prediction.label]} "
        f"(logit: {format_signed(before_prediction.logit)})"
    )
    print(
        f"  Prediction after  : {LABEL_NAMES[after_prediction.label]} "
        f"(logit: {format_signed(after_prediction.logit)})"
    )
    print(f"  Logit shift       : {format_signed(logit_shift)}")
    print(f"  Label flipped     : {format_flip_flag(flipped)}")
    print("═" * 48)


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
    print(f"  Prediction        : {LABEL_NAMES[prediction.label]} (confidence: {prediction.confidence:.4f})")
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
    html_doc = f"""<!DOCTYPE html>
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
        handle.write(html_doc)
    print(f"Saved SHAP HTML to {os.path.abspath(html_out)}")


def replace_first_token(text: str, old: str, new: str) -> str:
    pattern = re.compile(rf"\b{re.escape(old)}\b", flags=re.IGNORECASE)
    updated, count = pattern.subn(new, text, count=1)
    if count == 0:
        updated = text.replace(old, new, 1)
    return updated


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
    debug: bool = False,
):
    prediction = inspector.predict_text(text)
    print(f"\nText:\n{text}\n")
    print_prediction(prediction, prefix=prefix)
    shap_values = inspector.explain_text(text)
    shap_values = remove_ghost_tokens(shap_values)
    if debug:
        debug_print_shap_tokens(shap_values)
        debug_print_top_shap_tokens(shap_values)
    masked_count = count_and_mask_boundary_special_tokens(shap_values)
    ranked_tokens = inspector.extract_ranked_tokens(shap_values)
    print_text_statistics(
        tokenizer=inspector.tokenizer,
        text=text,
        prediction=prediction,
        true_label=true_label,
    )
    print(f"Masked {masked_count} special tokens from SHAP output")
    print_token_table(ranked_tokens)
    save_shap_html(shap_values, html_out)
    return prediction, shap_values, ranked_tokens


def interactive_flip_loop(
    inspector: TokenInspector,
    original_text: str,
    html_out: str,
    true_label: Optional[int],
    debug: bool = False,
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

        updated_text = normalize_whitespace(updated_text)
        updated_text = collapse_spaces_around_punct(updated_text)
        updated_text = updated_text.strip()

        if len(updated_text) < 3:
            print("Warning: replacement produced invalid text, skipping.")
            continue

        if debug:
            print(f"Debug text repr: {updated_text!r}")

        if updated_text == current_text:
            print("No change made after whitespace normalization.")
            continue

        current_text = updated_text
        new_prediction, _, _ = inspect_once(
            inspector,
            current_text,
            html_out,
            prefix="Modified",
            true_label=true_label,
            debug=debug,
        )
        flipped = previous_prediction.label != new_prediction.label
        print_flip_summary(token_to_replace, replacement, previous_prediction, new_prediction)
        history.append(
            {
                "replace": token_to_replace,
                "with": replacement,
                "before_label": previous_prediction.label,
                "before_conf": previous_prediction.confidence,
                "before_logit": previous_prediction.logit,
                "after_label": new_prediction.label,
                "after_conf": new_prediction.confidence,
                "after_logit": new_prediction.logit,
                "logit_shift": new_prediction.logit - previous_prediction.logit,
                "flipped": flipped,
                "text": current_text,
            }
        )

        keep_going = input("Flip another token? [y/N or quit]: ").strip().lower()
        if keep_going in {"quit", "n", "no", ""}:
            break

    return current_text, history


def print_summary(history: List[dict], initial_prediction: PredictionResult, final_prediction: PredictionResult):
    print("SESSION SUMMARY")
    print("═" * 48)
    print(
        f"Initial prediction: {LABEL_NAMES[initial_prediction.label]} "
        f"(logit: {format_signed(initial_prediction.logit)})"
    )
    print(
        f"Final prediction:   {LABEL_NAMES[final_prediction.label]} "
        f"(logit: {format_signed(final_prediction.logit)})"
    )
    if not history:
        print("No replacements were made.")
        return

    print()
    print("Step  Replacement                Logit Before  Logit After   Shift    Flipped")
    for step, item in enumerate(history, start=1):
        replacement_text = f'"{item["replace"]}" -> "{item["with"]}"'
        print(
            f"{step:<5} {replacement_text:<26} "
            f"{format_signed(item['before_logit']):<13} "
            f"{format_signed(item['after_logit']):<12} "
            f"{format_signed(item['logit_shift']):<8} "
            f"{format_flip_flag(item['flipped'])}"
        )
    print("═" * 48)


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
            debug=args.debug,
        )
        final_text, history = interactive_flip_loop(
            inspector,
            text,
            args.html_out,
            true_label=sample.true_label,
            debug=args.debug,
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
    parser.add_argument(
        "--test_csv",
        default="/cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_test.csv",
        help="CSV with statement and label columns",
    )
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
    parser.add_argument("--debug", action="store_true", help="Print debug details such as repr(text) after edits")
    main(parser.parse_args())
