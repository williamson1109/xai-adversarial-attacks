import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
import torch
from Levenshtein import distance as levenshtein_distance
from nlpaug.augmenter.word import SynonymAug
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


MAX_TOKENS = 250
LABEL_NAMES = {0: "FAKE", 1: "TRUE"}
WORD_RE = re.compile(r"\b[\w'-]+\b")
SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?|\n+")
SPECIAL_TOKENS = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}


@dataclass
class SentenceSpan:
    text: str
    start: int
    end: int
    score: float


@dataclass
class AttackOutcome:
    modified_text: str
    modified_label: int
    modified_confidence: float
    probabilities: np.ndarray


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, usecols=["statement", "label"])
    df = df.dropna(subset=["statement", "label"]).reset_index(drop=True)
    df["statement"] = df["statement"].astype(str)
    df["label"] = df["label"].astype(int)
    return df


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_token(token: str) -> str:
    if token is None:
        return ""
    text = str(token)
    for marker in ("##", "Ġ", "▁"):
        text = text.replace(marker, " ")
    text = text.strip()
    if text in SPECIAL_TOKENS:
        return ""
    return text


def split_sentences(text: str) -> List[Tuple[str, int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        sent = match.group(0)
        if sent.strip():
            spans.append((sent, match.start(), match.end()))
    if not spans:
        spans.append((text, 0, len(text)))
    return spans


def collapse_spaces_around_punct(text: str) -> str:
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return normalize_whitespace(text)


class AttackRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(self.device)
        self.model.eval()

        self.explainer = shap.Explainer(self.predict_proba, self.tokenizer)
        self.synonym_aug = self._build_synonym_augmenter()

        self.en_de_tokenizer = None
        self.en_de_model = None
        self.de_en_tokenizer = None
        self.de_en_model = None

        self.opposite_word_cache: Dict[int, List[str]] = {}

    def _build_synonym_augmenter(self) -> SynonymAug:
        try:
            import nltk

            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
        except Exception:
            pass
        return SynonymAug(aug_src="wordnet", aug_p=0.3)

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        if isinstance(texts, np.ndarray):
            texts = texts.tolist()
        texts = [str(text) for text in texts]
        all_probs = []
        for start in range(0, len(texts), self.args.batch_size):
            batch = texts[start:start + self.args.batch_size]
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

    def predict_text(self, text: str) -> AttackOutcome:
        probs = self.predict_proba([text])[0]
        label = int(np.argmax(probs))
        confidence = float(probs[label])
        return AttackOutcome(
            modified_text=text,
            modified_label=label,
            modified_confidence=confidence,
            probabilities=probs,
        )

    def explain(self, text: str):
        return self.explainer([text])

    def extract_token_importance(self, text: str, predicted_label: int) -> List[dict]:
        explanation = self.explain(text)
        raw_tokens = explanation.data[0]
        raw_values = explanation.values[0]
        if raw_values.ndim == 2:
            class_values = raw_values[:, predicted_label]
        else:
            class_values = raw_values

        raw_tokens = raw_tokens.tolist() if hasattr(raw_tokens, "tolist") else list(raw_tokens)
        spans = self.align_tokens_to_text(text, raw_tokens)

        token_info = []
        for idx, (token, value, span) in enumerate(zip(raw_tokens, class_values, spans)):
            cleaned = clean_token(token)
            token_info.append(
                {
                    "index": idx,
                    "token": str(token),
                    "cleaned": cleaned,
                    "score": float(value),
                    "span": span,
                }
            )
        return token_info

    def align_tokens_to_text(self, text: str, tokens: Sequence[str]) -> List[Optional[Tuple[int, int]]]:
        lowered = text.lower()
        cursor = 0
        spans: List[Optional[Tuple[int, int]]] = []
        for token in tokens:
            cleaned = clean_token(token)
            if not cleaned:
                spans.append(None)
                continue

            candidates = []
            candidates.append(cleaned)
            candidates.append(cleaned.lstrip())
            candidates.append(cleaned.strip())
            deduped = []
            seen = set()
            for candidate in candidates:
                if candidate and candidate not in seen:
                    seen.add(candidate)
                    deduped.append(candidate)

            found = None
            for candidate in deduped:
                idx = lowered.find(candidate.lower(), cursor)
                if idx != -1:
                    found = (idx, idx + len(candidate))
                    cursor = idx + len(candidate)
                    break
            if found is None:
                for candidate in deduped:
                    idx = lowered.find(candidate.lower())
                    if idx != -1:
                        found = (idx, idx + len(candidate))
                        break
            spans.append(found)
        return spans

    def rank_word_spans(self, text: str, predicted_label: int) -> List[dict]:
        token_info = self.extract_token_importance(text, predicted_label)
        ranked = []
        seen_spans = set()
        for item in token_info:
            span = item["span"]
            cleaned = item["cleaned"]
            if span is None or not cleaned:
                continue
            if item["score"] <= 0:
                continue
            if not WORD_RE.fullmatch(cleaned):
                continue
            if span in seen_spans:
                continue
            seen_spans.add(span)
            ranked.append(item)

        ranked.sort(key=lambda row: (-row["score"], len(row["cleaned"]), row["index"]))
        return ranked

    def rank_sentences(self, text: str, predicted_label: int) -> List[SentenceSpan]:
        sentences = split_sentences(text)
        token_info = self.extract_token_importance(text, predicted_label)

        ranked = []
        for sentence, start, end in sentences:
            score = 0.0
            for token in token_info:
                span = token["span"]
                if span is None:
                    continue
                token_center = (span[0] + span[1]) / 2
                if start <= token_center <= end:
                    score += abs(token["score"])
            ranked.append(SentenceSpan(text=sentence, start=start, end=end, score=score))

        ranked.sort(key=lambda sentence: sentence.score, reverse=True)
        return ranked

    def remove_span(self, text: str, span: Tuple[int, int]) -> str:
        updated = text[:span[0]] + " " + text[span[1]:]
        return collapse_spaces_around_punct(updated)

    def insert_word(self, text: str, position: int, word: str) -> str:
        if not word:
            return text
        if position <= 0:
            updated = f"{word} {text}"
        elif position >= len(text):
            updated = f"{text} {word}"
        else:
            left_needs_space = text[position - 1].isalnum()
            right_needs_space = text[position].isalnum()
            updated = text[:position]
            if left_needs_space:
                updated += " "
            updated += word
            if right_needs_space:
                updated += " "
            updated += text[position:]
        return collapse_spaces_around_punct(updated)

    def replace_sentence(self, text: str, sentence_span: SentenceSpan, replacement: str) -> str:
        updated = text[:sentence_span.start] + replacement + text[sentence_span.end:]
        return collapse_spaces_around_punct(updated)

    def build_opposite_word_pool(self, df: pd.DataFrame, target_label: int) -> List[str]:
        if target_label in self.opposite_word_cache:
            return self.opposite_word_cache[target_label]

        subset = df[df["label"] == target_label].head(self.args.reference_samples)
        scores = defaultdict(float)
        counts = defaultdict(int)
        for text in tqdm(subset["statement"].tolist(), desc=f"Collecting class-{target_label} words", leave=False):
            for token in self.extract_token_importance(text, target_label):
                cleaned = token["cleaned"].lower()
                if token["score"] <= 0 or len(cleaned) < 3:
                    continue
                if not WORD_RE.fullmatch(cleaned):
                    continue
                scores[cleaned] += token["score"]
                counts[cleaned] += 1

        ranked = sorted(
            scores,
            key=lambda word: (-(scores[word] / max(counts[word], 1)), len(word), word),
        )
        pool = ranked[: self.args.injection_candidates]
        self.opposite_word_cache[target_label] = pool
        return pool

    def attack_swr(self, text: str) -> AttackOutcome:
        current_text = text
        original = self.predict_text(text)

        for _ in range(self.args.max_removals):
            current = self.predict_text(current_text)
            if current.modified_label != original.modified_label:
                return current

            ranked_words = self.rank_word_spans(current_text, current.modified_label)
            if not ranked_words:
                return current

            candidate = ranked_words[0]
            current_text = self.remove_span(current_text, candidate["span"])
            if not current_text:
                return self.predict_text(current_text)

        return self.predict_text(current_text)

    def attack_swi(self, text: str, df: pd.DataFrame) -> AttackOutcome:
        original = self.predict_text(text)
        target_label = 1 - original.modified_label
        candidate_words = self.build_opposite_word_pool(df, target_label)
        token_info = self.extract_token_importance(text, original.modified_label)

        positions = {0, len(text)}
        for token in sorted(token_info, key=lambda row: row["score"], reverse=True)[: self.args.position_candidates]:
            span = token["span"]
            if span is not None and token["score"] > 0:
                positions.add(span[0])
                positions.add(span[1])

        best_text = text
        best_probs = original.probabilities
        best_target_score = float(best_probs[target_label])

        for word in candidate_words:
            for position in sorted(positions):
                candidate_text = self.insert_word(text, position, word)
                candidate_outcome = self.predict_text(candidate_text)
                target_score = float(candidate_outcome.probabilities[target_label])
                if target_score > best_target_score:
                    best_text = candidate_text
                    best_probs = candidate_outcome.probabilities
                    best_target_score = target_score
                if candidate_outcome.modified_label == target_label:
                    return candidate_outcome

        return AttackOutcome(
            modified_text=best_text,
            modified_label=int(np.argmax(best_probs)),
            modified_confidence=float(np.max(best_probs)),
            probabilities=best_probs,
        )

    def attack_ss(self, text: str) -> AttackOutcome:
        original = self.predict_text(text)
        current_text = text

        for _ in range(self.args.max_target_sentences):
            current_prediction = self.predict_text(current_text)
            if current_prediction.modified_label != original.modified_label:
                return current_prediction

            ranked_sentences = self.rank_sentences(current_text, current_prediction.modified_label)
            if not ranked_sentences:
                break
            sentence_span = ranked_sentences[0]
            try:
                augmented = self.synonym_aug.augment(sentence_span.text)
            except Exception:
                continue
            if isinstance(augmented, list):
                augmented = augmented[0] if augmented else sentence_span.text
            if not isinstance(augmented, str) or normalize_whitespace(augmented) == normalize_whitespace(sentence_span.text):
                continue

            current_text = self.replace_sentence(current_text, sentence_span, augmented)
            current_outcome = self.predict_text(current_text)
            if current_outcome.modified_label != original.modified_label:
                return current_outcome

        return self.predict_text(current_text)

    def ensure_back_translation_models(self):
        if self.en_de_model is not None:
            return
        self.en_de_tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-en-de")
        self.en_de_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-en-de").to(self.device)
        self.de_en_tokenizer = AutoTokenizer.from_pretrained("facebook/wmt19-de-en")
        self.de_en_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/wmt19-de-en").to(self.device)
        self.en_de_model.eval()
        self.de_en_model.eval()

    def translate(self, text: str, tokenizer, model) -> str:
        encoded = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            generated = model.generate(**encoded, max_length=MAX_TOKENS)
        return tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

    def back_translate(self, text: str) -> str:
        self.ensure_back_translation_models()
        german = self.translate(text, self.en_de_tokenizer, self.en_de_model)
        english = self.translate(german, self.de_en_tokenizer, self.de_en_model)
        return english

    def attack_bt(self, text: str) -> AttackOutcome:
        original = self.predict_text(text)
        current_text = text

        for _ in range(self.args.max_target_sentences):
            current_prediction = self.predict_text(current_text)
            if current_prediction.modified_label != original.modified_label:
                return current_prediction

            ranked_sentences = self.rank_sentences(current_text, current_prediction.modified_label)
            if not ranked_sentences:
                break
            sentence_span = ranked_sentences[0]
            try:
                translated = self.back_translate(sentence_span.text)
            except Exception:
                continue
            if normalize_whitespace(translated) == normalize_whitespace(sentence_span.text):
                continue

            current_text = self.replace_sentence(current_text, sentence_span, translated)
            current_outcome = self.predict_text(current_text)
            if current_outcome.modified_label != original.modified_label:
                return current_outcome

        return self.predict_text(current_text)


def compute_metrics(original_text: str, original_probs: np.ndarray, original_label: int, attacked: AttackOutcome) -> dict:
    ci = levenshtein_distance(original_text, attacked.modified_text)
    pi = 100.0 * ci / max(len(original_text), 1)
    target_label = 1 - original_label
    attack_effective = int(attacked.modified_label != original_label)
    average_shift = float(attacked.probabilities[target_label] - original_probs[target_label])
    return {
        "AE": attack_effective,
        "CI": ci,
        "PI": pi,
        "AS": average_shift,
    }


def print_summary_table(results_df: pd.DataFrame):
    print("\nTable 3-style summary")
    print("=" * 86)
    print(f"{'Attack':<8} {'AE':<10} {'CI':<20} {'PI':<20} {'AS':<10}")
    print("-" * 86)
    for attack_type, group in results_df.groupby("attack_type", sort=False):
        ae_mean = group["AE"].mean()
        ci_mean = group["CI"].mean()
        ci_std = group["CI"].std(ddof=0)
        pi_mean = group["PI"].mean()
        pi_std = group["PI"].std(ddof=0)
        as_mean = group["AS"].mean()
        print(
            f"{attack_type:<8} "
            f"{ae_mean:<10.3f} "
            f"{f'{ci_mean:.2f} ± {ci_std:.2f}':<20} "
            f"{f'{pi_mean:.2f} ± {pi_std:.2f}':<20} "
            f"{as_mean:<10.3f}"
        )
    print("=" * 86)


def print_examples(results_df: pd.DataFrame, max_examples: int):
    print("\nQualitative examples")
    print("=" * 86)
    ranked = results_df.sort_values(["AE", "AS"], ascending=[False, False]).head(max_examples)
    for _, row in ranked.iterrows():
        before_label = LABEL_NAMES[int(row["original_label"])]
        after_label = LABEL_NAMES[int(row["modified_label"])]
        print(f"[{row['attack_type']}] {before_label} ({row['original_confidence']:.3f}) -> {after_label} ({row['modified_confidence']:.3f})")
        print(f"Original: {row['original_text']}")
        print(f"Modified: {row['modified_text']}")
        print("-" * 86)


def main(args: argparse.Namespace):
    df = load_data(args.test_csv)
    sample_df = df.head(args.n_samples).copy()

    runner = AttackRunner(args)
    print(f"Using device: {runner.device}")
    print(f"Loaded {len(df)} samples, attacking first {len(sample_df)}")

    attack_fns = {
        "SWR": lambda text: runner.attack_swr(text),
        "SWI": lambda text: runner.attack_swi(text, df),
        "SS": lambda text: runner.attack_ss(text),
        "BT": lambda text: runner.attack_bt(text),
    }

    rows = []
    iterator = tqdm(sample_df.iterrows(), total=len(sample_df), desc="Attacking samples")
    for sample_id, row in iterator:
        original_text = normalize_whitespace(row["statement"])
        original_prediction = runner.predict_text(original_text)

        for attack_type, attack_fn in attack_fns.items():
            attacked = attack_fn(original_text)
            metrics = compute_metrics(
                original_text=original_text,
                original_probs=original_prediction.probabilities,
                original_label=original_prediction.modified_label,
                attacked=attacked,
            )
            rows.append(
                {
                    "sample_id": sample_id,
                    "original_text": original_text,
                    "modified_text": attacked.modified_text,
                    "original_label": original_prediction.modified_label,
                    "original_confidence": original_prediction.modified_confidence,
                    "modified_label": attacked.modified_label,
                    "modified_confidence": attacked.modified_confidence,
                    "attack_type": attack_type,
                    **metrics,
                }
            )

    results_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    output_columns = [
        "sample_id",
        "original_text",
        "modified_text",
        "original_label",
        "original_confidence",
        "attack_type",
        "AE",
        "CI",
        "PI",
        "AS",
    ]
    results_df[output_columns].to_csv(args.out_path, index=False)
    print(f"\nSaved per-sample results to {args.out_path}")

    print_summary_table(results_df)
    print_examples(results_df, args.example_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run SHAP-guided adversarial attacks against a DistilBERT fake news detector."
    )
    parser.add_argument("--test_csv", required=True, help="CSV with statement and label columns")
    parser.add_argument("--model_dir", required=True, help="Directory containing the trained model and tokenizer")
    parser.add_argument("--n_samples", type=int, default=10, help="Number of samples to attack")
    parser.add_argument("--out_path", required=True, help="Path to save per-sample attack results as CSV")
    parser.add_argument("--batch_size", type=int, default=8, help="Inference batch size")
    parser.add_argument("--reference_samples", type=int, default=50, help="Samples per class used to build SWI word pools")
    parser.add_argument("--injection_candidates", type=int, default=20, help="Candidate opposite-class words for SWI")
    parser.add_argument("--position_candidates", type=int, default=8, help="Important insertion positions to test in SWI")
    parser.add_argument("--max_removals", type=int, default=20, help="Maximum iterative removals for SWR")
    parser.add_argument("--max_target_sentences", type=int, default=3, help="Top influential sentences to perturb for SS and BT")
    parser.add_argument("--example_count", type=int, default=3, help="Number of qualitative examples to print")
    main(parser.parse_args())
