"""
LLM-Guided Adversarial Attack on Fake News Classifier
======================================================
Extends Kozik et al. (2023) by replacing rule-based word substitution
with an LLM (Claude API) guided by SHAP token importance scores.

Pipeline per sample:
  1. Get model prediction + confidence
  2. Run SHAP -> extract top tokens by importance
  3. Send original text + top SHAP tokens to Claude API
  4. Claude returns minimally modified text targeting those tokens
  5. Re-run classifier -> flipped? Record metrics and stop
  6. If not flipped -> re-run SHAP on modified text -> repeat
  7. Stop after --max_iter iterations or successful flip
  8. Loop detection: stop if LLM produces a previously seen text

Outputs two CSVs matching the LIAR Experiment Log Google Sheet structure:
  - llm_experiment_log.csv      -> LIAR Experiment Log tab
  - llm_modification_detail.csv -> LIAR Modification Detail tab
"""

import argparse
import os
import re
from datetime import datetime

import time
import anthropic
import numpy as np
import pandas as pd
import scipy as sp
import shap
import torch
from Levenshtein import distance as levenshtein_distance
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


MAX_TOKENS = 250
LABEL_NAMES = {0: "Fake", 1: "Real"}
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>", ""}

# Approximate cost per token (Claude Sonnet 4)
COST_PER_INPUT_TOKEN  = 3.0  / 1_000_000   # $3 per 1M input tokens
COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000   # $15 per 1M output tokens


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

def count_words(text: str) -> int:
    return len([w for w in text.split() if any(c.isalnum() for c in w)])


def count_sentences(text: str) -> int:
    sentences = re.split(r'[.!?]+', text)
    return len([s for s in sentences if s.strip()])


def count_word_diff(text1: str, text2: str) -> int:
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    return len(words1.symmetric_difference(words2))


def count_sentence_diff(text1: str, text2: str) -> int:
    sents1 = set(re.split(r'[.!?]+', text1))
    sents2 = set(re.split(r'[.!?]+', text2))
    return len([s for s in sents1.symmetric_difference(sents2) if s.strip()])


def format_pct(value: float) -> str:
    return f"{value*100:.2f}%"


def format_conf_change(before: float, after: float) -> str:
    change = (after - before) * 100
    sign = "+" if change >= 0 else ""
    return f"{sign}{change:.2f}%"


# ---------------------------------------------------------------------------
# Additional metrics utilities
# ---------------------------------------------------------------------------

def compute_bleu(original: str, modified: str) -> float:
    """Compute sentence-level BLEU score between original and modified text."""
    if not NLTK_AVAILABLE:
        return -1.0
    try:
        ref = original.lower().split()
        hyp = modified.lower().split()
        smoothie = SmoothingFunction().method1
        return round(sentence_bleu([ref], hyp, smoothing_function=smoothie), 4)
    except Exception:
        return -1.0


def find_changed_tokens(original: str, modified: str) -> str:
    """Identify which words changed between original and modified text."""
    orig_words = original.lower().split()
    mod_words  = modified.lower().split()
    changed = []
    for i, (o, m) in enumerate(zip(orig_words, mod_words)):
        if o != m:
            changed.append(f'"{o}"→"{m}"')
    # handle length differences
    if len(orig_words) > len(mod_words):
        for w in orig_words[len(mod_words):]:
            changed.append(f'"{w}"→[removed]')
    elif len(mod_words) > len(orig_words):
        for w in mod_words[len(orig_words):]:
            changed.append(f'[added]→"{w}"')
    return ", ".join(changed) if changed else "no change detected"


def sentence_length_change(original: str, modified: str) -> int:
    """Word count difference between modified and original (positive = longer)."""
    return len(modified.split()) - len(original.split())


# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def load_model(model_dir: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    return tokenizer, model


def get_probs(texts, tokenizer, model, device, batch_size=8):
    if isinstance(texts, np.ndarray):
        texts = texts.tolist()
    texts = [str(t) for t in texts]
    all_probs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=MAX_TOKENS,
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        all_probs.append(probs)
    return np.vstack(all_probs)


def predict(text, tokenizer, model, device):
    probs = get_probs([text], tokenizer, model, device)[0]
    label = int(np.argmax(probs))
    logit = float(sp.special.logit(float(probs[1])))
    return label, float(probs[label]), probs, logit


# ---------------------------------------------------------------------------
# SHAP utilities
# ---------------------------------------------------------------------------

def build_explainer(tokenizer, model, device):
    def predict_fn(texts):
        probs = get_probs(texts, tokenizer, model, device)
        return sp.special.logit(probs[:, 1])
    return shap.Explainer(predict_fn, tokenizer)


def clean_token(token: str) -> str:
    text = str(token)
    for marker in ("##", "Ġ", "▁"):
        text = text.replace(marker, " ")
    text = text.strip()
    if not text or text in SPECIAL_TOKENS:
        return ""
    return text


def get_top_shap_tokens(explainer, text: str, predicted_label: int, top_n: int = 5):
    shap_values = explainer([text], fixed_context=1)
    raw_tokens = list(shap_values.data[0])
    raw_values = shap_values.values[0]

    token_scores = []
    for token, value in zip(raw_tokens, raw_values):
        cleaned = clean_token(str(token))
        if not cleaned:
            continue
        # negative SHAP = pushes toward FAKE, positive SHAP = pushes toward TRUE
        token_scores.append((cleaned, float(value)))

    token_scores.sort(key=lambda x: abs(x[1]), reverse=True)
    return token_scores[:top_n]


def format_top_tokens(top_tokens: list) -> str:
    return ", ".join(
        f"{tok} ({score:+.3f})"
        for tok, score in top_tokens
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

PUNCTUATION_TOKENS = {".", ",", "!", "?", ";", ":", "-", "'", '"', "(", ")", "..."}


def call_claude(client, text: str, top_tokens: list,
                previous_texts: list = None,
                max_retries: int = 5) -> tuple:
    """Returns (modified_text, input_tokens, output_tokens).
    Retries with exponential backoff on 529 overloaded errors."""
    token_list = "\n".join(
        f'  - "{tok}"'
        for tok, _ in top_tokens
    )

    # check if any top tokens are punctuation — if so, give explicit instruction
    # for TRUE predictions: target positive SHAP punct (pushing toward TRUE)
    # for FAKE predictions: target negative SHAP punct (pushing toward FAKE)
    top_token_strings = [tok.strip() for tok, _ in top_tokens]
    punct_tokens = [t for t in top_token_strings if t in PUNCTUATION_TOKENS]
    punct_note = ""
    if punct_tokens:
        punct_list = ", ".join(f'"{p}"' for p in punct_tokens)
        punct_note = (
            f"\nIMPORTANT: The following punctuation marks are among the most "
            f"influential tokens: {punct_list}. "
            f"You MUST try removing or replacing at least one of them. "
            f"For example, remove a trailing period, replace a comma with a semicolon, "
            f"or remove an exclamation mark. This is the most effective modification."
        )

    # build history note to prevent loops
    history_note = ""
    if previous_texts:
        history_note = "\nDo NOT reproduce any of these previously tried versions:\n"
        history_note += "\n".join(f'  - "{t}"' for t in previous_texts)

    prompt = f"""Please rewrite the following sentence with minimal changes.
Focus on finding natural alternatives for these specific words if possible:
{token_list}
{punct_note}
Sentence: "{text}"
{history_note}
Rules:
1. Keep the original meaning completely intact
2. Only change 1-2 words or punctuation marks maximum
3. Ensure the result is grammatically correct and fluent
4. Punctuation changes (removing or replacing periods, commas, etc.) are valid and do not affect meaning
5. Return ONLY the rewritten sentence, nothing else"""

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            result = response.content[0].text.strip()
            result = re.sub(r'^["\']|["\']$', '', result).strip()
            return result, response.usage.input_tokens, response.usage.output_tokens
        except Exception as e:
            if "529" in str(e) or "overloaded" in str(e).lower():
                wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"  API overloaded, retrying in {wait}s "
                      f"(attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait)
            else:
                raise  # re-raise non-overload errors immediately

    raise RuntimeError(f"API still overloaded after {max_retries} retries")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(original_text, original_probs, original_label,
                    modified_text, modified_probs, modified_label):
    ci = levenshtein_distance(original_text, modified_text)
    pi = 100.0 * ci / max(len(original_text), 1)
    target_label = 1 - original_label
    ae = int(modified_label != original_label)
    as_metric = float(modified_probs[target_label] - original_probs[target_label])
    return {"AE": ae, "CI": ci, "PI": pi, "AS": as_metric}


# ---------------------------------------------------------------------------
# Main attack loop per sample
# ---------------------------------------------------------------------------

def attack_sample(
    exp_num: int,
    sample_id: int,
    text: str,
    true_label: int,
    tokenizer,
    model,
    device,
    explainer,
    client,
    max_iter: int,
    top_n: int,
    today: str,
    budget_tracker: dict,
) -> tuple:
    orig_label, orig_conf, orig_probs, orig_logit = predict(
        text, tokenizer, model, device
    )
    current_text = text
    modification_detail_rows = []
    final_label = orig_label
    final_conf = orig_conf
    final_probs = orig_probs
    final_logit = orig_logit

    # track all seen texts to detect loops
    seen_texts = {text.strip()}

    # track best text — closest to decision boundary (logit closest to 0)
    best_text = text
    best_distance_to_boundary = abs(orig_logit)
    best_label = orig_label
    best_conf = orig_conf
    best_probs = orig_probs
    best_logit = orig_logit

    # final_text initialised to original — overwritten on flip or after loop
    final_text = text

    for iteration in range(1, max_iter + 1):
        # check budget before calling API
        if budget_tracker["spent"] >= budget_tracker["limit"]:
            print(f"\n  ⚠ Budget limit ${budget_tracker['limit']:.2f} reached — stopping.")
            break

        # always attack from the best text found so far
        current_text = best_text
        current_label = best_label
        current_logit = best_logit

        # get top SHAP tokens for best text
        top_tokens = get_top_shap_tokens(
            explainer, current_text, current_label, top_n
        )
        impact_before = current_logit

        # call LLM — pass seen texts to prevent loops
        try:
            modified_text, in_tok, out_tok = call_claude(
                client, current_text, top_tokens,
                previous_texts=list(seen_texts)
            )
            # track cost
            cost = (in_tok * COST_PER_INPUT_TOKEN) + (out_tok * COST_PER_OUTPUT_TOKEN)
            budget_tracker["spent"]         += cost
            budget_tracker["input_tokens"]  += in_tok
            budget_tracker["output_tokens"] += out_tok
            budget_tracker["api_calls"]     += 1
        except RuntimeError as e:
            # all retries exhausted — skip iteration but continue
            print(f"  Skipping iteration {iteration} after retries exhausted: {e}")
            continue
        except Exception as e:
            print(f"  Unexpected error at iteration {iteration}: {e}")
            continue

        # stop if no change
        if not modified_text or modified_text.strip() == current_text.strip():
            print(f"  No change from LLM at iteration {iteration}, stopping.")
            break

        # stop if loop detected
        if modified_text.strip() in seen_texts:
            print(f"  Loop detected at iteration {iteration} — stopping.")
            break

        # register new text
        seen_texts.add(modified_text.strip())

        # predict on modified text
        mod_label, mod_conf, mod_probs, mod_logit = predict(
            modified_text, tokenizer, model, device
        )
        impact_after = mod_logit

        # update best text if this one is closer to decision boundary
        distance_to_boundary = abs(mod_logit)
        is_new_best = distance_to_boundary < best_distance_to_boundary
        if is_new_best:
            best_distance_to_boundary = distance_to_boundary
            best_text = modified_text
            best_label = mod_label
            best_conf = mod_conf
            best_probs = mod_probs
            best_logit = mod_logit
            print(f"  ✓ New best at iteration {iteration} "
                  f"(logit: {mod_logit:+.4f}, distance to boundary: {distance_to_boundary:.4f})")

        # record modification detail row
        modification_detail_rows.append({
            "Exp #": exp_num,
            "Article Idx": sample_id,
            "Sent #": iteration,
            "Strategy": "LLM rewrite",
            "Original Sentence": current_text,
            "Modified Sentence": modified_text,
            "Top Impact Words": format_top_tokens(top_tokens),
            "Impact Before": f"{impact_before:.4f}",
            "Impact After": f"{impact_after:.4f}",
            "Notes/observations": "← new best" if is_new_best else f"best=logit {best_logit:+.4f}",
        })

        # check flip immediately after predicting on modified text
        if mod_label != orig_label:
            final_label = mod_label
            final_conf = mod_conf
            final_probs = mod_probs
            final_logit = mod_logit
            final_text = modified_text
            print(f"  🎯 Flip detected at iteration {iteration}!")
            break
    else:
        # no flip — use best text found as final result
        final_label = best_label
        final_conf = best_conf
        final_probs = best_probs
        final_logit = best_logit
        final_text = best_text

    # final metrics — compare original to best/flipped text
    iterations_run = len(modification_detail_rows)
    metrics = compute_metrics(
        text, orig_probs, orig_label,
        final_text, final_probs, final_label
    )
    flipped = final_label != orig_label
    correct_after = final_label == true_label

    # build experiment log row matching Google Sheet columns exactly
    experiment_log_row = {
        "Exp #": exp_num,
        "Date": today,
        "Article Idx": sample_id,
        "True Label": LABEL_NAMES[true_label],
        "# Sentences": count_sentences(text),
        "# Tokens\n(full)": len(tokenizer.tokenize(text)) + 2,
        "# Words (full)": count_words(text),
        "Orig Pred": LABEL_NAMES[orig_label],
        "Orig Conf %": format_pct(orig_conf),
        "Orig P(Real)": format_pct(orig_probs[1]),
        "Orig P(Fake)": format_pct(orig_probs[0]),
        "Strategy": "LLM rewrite",
        "Sent # Modified": iterations_run,
        "# Words changed": count_word_diff(text, final_text),
        "# Sents Changed": count_sentence_diff(text, final_text),
        "New Pred": LABEL_NAMES[final_label],
        "New Conf %": format_pct(final_conf),
        "New P(Real)": format_pct(final_probs[1]),
        "New P(Fake)": format_pct(final_probs[0]),
        "Flipped?": "YES" if flipped else "NO",
        "Correct After?": "YES" if correct_after else "NO",
        "Conf Change %": format_conf_change(orig_conf, final_conf),
        "Orig Logit":          f"{orig_logit:+.4f}",
        "Final Logit":         f"{final_logit:+.4f}",
        "Logit Shift":         f"{final_logit - orig_logit:+.4f}",
        "BLEU Score":          compute_bleu(text, final_text),
        "Changed Token(s)":    find_changed_tokens(text, final_text),
        "Length Change":       sentence_length_change(text, final_text),
        "Orig Correct?":       "YES" if orig_label == true_label else "NO",
        "Notes / Observations": (
            f"LLM attack | {iterations_run} iter | "
            f"AE={metrics['AE']} CI={metrics['CI']} "
            f"PI={metrics['PI']:.1f}% AS={metrics['AS']:.3f}"
        ),
    }

    return experiment_log_row, modification_detail_rows


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(exp_log_df: pd.DataFrame):
    flipped = exp_log_df["Flipped?"].str.upper() == "YES"
    print("\n" + "=" * 70)
    print("LLM ATTACK RESULTS")
    print("=" * 70)
    print(f"  Samples attacked : {len(exp_log_df)}")
    print(f"  Successful flips : {flipped.sum()} ({flipped.mean()*100:.1f}%)")
    print("=" * 70)

    flipped_rows = exp_log_df[flipped].head(3)
    if not flipped_rows.empty:
        print("\nQualitative examples (successful flips):")
        print("-" * 70)
        for _, row in flipped_rows.iterrows():
            print(f"  Article {row['Article Idx']}: "
                  f"{row['Orig Pred']} ({row['Orig Conf %']}) -> "
                  f"{row['New Pred']} ({row['New Conf %']})")
            print(f"  Conf change: {row['Conf Change %']}")
            print("-" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    today = datetime.now().strftime("%d.%m.%Y")

    # load model
    tokenizer, model = load_model(args.model_dir, device)

    # build SHAP explainer
    print("Building SHAP explainer...")
    explainer = build_explainer(tokenizer, model, device)

    # load Anthropic client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "Anthropic API key required. "
            "Pass --api_key or set ANTHROPIC_API_KEY env variable."
        )
    client = anthropic.Anthropic(api_key=api_key)

    # load data
    df = pd.read_csv(args.test_csv, usecols=["statement", "label"])
    df = df.dropna(subset=["statement", "label"]).reset_index(drop=True)
    sample_df = df.sample(
        n=min(args.n_samples, len(df)), random_state=args.seed
    )
    print(
        f"Attacking {len(sample_df)} samples "
        f"(max {args.max_iter} iterations each)\n"
    )
    print(f"Budget limit: ${args.budget_limit:.2f}")

    # initialise budget tracker
    budget_tracker = {
        "limit":         args.budget_limit,
        "spent":         0.0,
        "input_tokens":  0,
        "output_tokens": 0,
        "api_calls":     0,
    }

    # define output paths before loop so incremental save works
    os.makedirs(args.out_dir, exist_ok=True)
    exp_log_path    = os.path.join(args.out_dir, "llm_experiment_log.csv")
    mod_detail_path = os.path.join(args.out_dir, "llm_modification_detail.csv")

    # run attacks — load existing rows first so incremental save preserves them
    exp_log_rows = []
    mod_detail_rows = []

    # auto-detect starting exp number from existing log if it exists
    # and load existing rows so incremental save preserves them
    if os.path.exists(exp_log_path):
        try:
            existing = pd.read_csv(exp_log_path)
            if not existing.empty and "Exp #" in existing.columns:
                exp_num = int(existing["Exp #"].max()) + 1
                exp_log_rows = existing.to_dict("records")
                print(f"Auto-detected start exp num: {exp_num} (continuing from existing log)")
                print(f"Loaded {len(exp_log_rows)} existing experiment rows")
            else:
                exp_num = args.start_exp_num
        except Exception:
            exp_num = args.start_exp_num
    else:
        exp_num = args.start_exp_num

    if os.path.exists(mod_detail_path):
        try:
            existing_mod = pd.read_csv(mod_detail_path)
            if not existing_mod.empty:
                mod_detail_rows = existing_mod.to_dict("records")
                print(f"Loaded {len(mod_detail_rows)} existing modification detail rows")
        except Exception:
            pass

    print(f"Starting from experiment #{exp_num}")

    # skip samples already attacked in previous runs
    if exp_log_rows:
        already_attacked = set(str(row["Article Idx"]) for row in exp_log_rows)
        original_count = len(sample_df)
        sample_df = sample_df[~sample_df.index.astype(str).isin(already_attacked)]
        skipped = original_count - len(sample_df)
        if skipped > 0:
            print(f"Skipping {skipped} already-attacked samples ({len(sample_df)} remaining)")
        # if all samples already attacked, sample new ones from the rest of the dataset
        if len(sample_df) == 0:
            print("All sampled articles already attacked — sampling from remaining unseen articles...")
            remaining_df = df[~df.index.astype(str).isin(already_attacked)]
            sample_df = remaining_df.sample(
                n=min(args.n_samples, len(remaining_df)), random_state=args.seed
            )
            print(f"Sampled {len(sample_df)} new unseen articles")

    for sample_id, row in tqdm(
        sample_df.iterrows(), total=len(sample_df), desc="LLM attack"
    ):
        # stop entire run if budget exhausted
        if budget_tracker["spent"] >= budget_tracker["limit"]:
            print(f"\n⚠ Budget limit ${budget_tracker['limit']:.2f} reached — stopping early.")
            break

        text = str(row["statement"]).strip()
        true_label = int(row["label"])

        exp_row, detail_rows = attack_sample(
            exp_num=exp_num,
            sample_id=sample_id,
            text=text,
            true_label=true_label,
            tokenizer=tokenizer,
            model=model,
            device=device,
            explainer=explainer,
            client=client,
            max_iter=args.max_iter,
            top_n=args.top_n,
            today=today,
            budget_tracker=budget_tracker,
        )

        exp_log_rows.append(exp_row)
        mod_detail_rows.extend(detail_rows)
        exp_num += 1

        # print running cost after each sample
        print(
            f"  💰 Cost so far: ${budget_tracker['spent']:.4f} / "
            f"${budget_tracker['limit']:.2f} "
            f"({budget_tracker['api_calls']} API calls)"
        )

        # save incrementally — exp_log_rows and mod_detail_rows already include existing data
        pd.DataFrame(exp_log_rows).to_csv(exp_log_path, index=False)
        pd.DataFrame(mod_detail_rows).to_csv(mod_detail_path, index=False)

    # build final dataframes
    exp_log_df    = pd.DataFrame(exp_log_rows)
    mod_detail_df = pd.DataFrame(mod_detail_rows)

    print(f"\nExperiment log saved to     : {exp_log_path}")
    print(f"Modification detail saved to : {mod_detail_path}")
    print("\nImport directly into your Google Sheet tabs:")
    print("  llm_experiment_log.csv      -> LIAR Experiment Log tab")
    print("  llm_modification_detail.csv -> LIAR Modification Detail tab")

    # final cost summary
    print("\n" + "=" * 70)
    print("COST SUMMARY")
    print("=" * 70)
    print(f"  API calls      : {budget_tracker['api_calls']}")
    print(f"  Input tokens   : {budget_tracker['input_tokens']:,}")
    print(f"  Output tokens  : {budget_tracker['output_tokens']:,}")
    print(f"  Total spent    : ${budget_tracker['spent']:.4f}")
    print(f"  Budget limit   : ${budget_tracker['limit']:.2f}")
    print(f"  Remaining      : ${budget_tracker['limit'] - budget_tracker['spent']:.4f}")
    print("=" * 70)

    print_summary(exp_log_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "LLM-guided adversarial attack on fake news classifier "
            "using SHAP + Claude API."
        )
    )
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--test_csv", required=True,
        help="Path to test CSV with statement and label columns"
    )
    parser.add_argument(
        "--out_dir", required=True,
        help="Directory to save output CSVs"
    )
    parser.add_argument(
        "--api_key", default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=50,
        help="Number of samples to attack (default: 50)"
    )
    parser.add_argument(
        "--max_iter", type=int, default=10,
        help="Max LLM iterations per sample (default: 10)"
    )
    parser.add_argument(
        "--top_n", type=int, default=5,
        help="Top N SHAP tokens to pass to LLM (default: 5)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8,
        help="Inference batch size (default: 8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sample selection (default: 42)"
    )
    parser.add_argument(
        "--budget_limit", type=float, default=2.0,
        help="Maximum USD to spend on Claude API calls (default: $2.00)"
    )
    parser.add_argument(
        "--start_exp_num", type=int, default=1,
        help="Starting experiment number for log (default: 1)"
    )
    main(parser.parse_args())