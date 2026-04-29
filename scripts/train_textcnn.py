"""
TextCNN Fake News Classifier — Kim (2014)
=========================================
Trains a TextCNN on the LIAR binary dataset using the same
5x2 repeated k-fold CV setup as the DistilBERT/RoBERTa experiments.

Architecture (Kim 2014):
  - GloVe 100d pretrained embeddings (frozen)
  - Conv1d filters of sizes [3, 4, 5], 100 filters each
  - Max-over-time pooling
  - Dropout (0.5)
  - Fully connected output layer (2 classes)

Usage:
    python scripts/train_textcnn.py \
        --train_csv data/processed/liar_train.csv \
        --glove_path data/glove/glove.6B.100d.txt \
        --out_dir models/textcnn_model \
        --epochs 10 \
        --batch_size 64
"""

import argparse
import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from collections import Counter
import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}

MAX_LEN      = 250   # match DistilBERT/RoBERTa max tokens
EMBED_DIM    = 100   # GloVe 100d
FILTER_SIZES = [3, 4, 5]
NUM_FILTERS  = 100
DROPOUT      = 0.5
MIN_FREQ     = 2     # minimum word frequency to include in vocab

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    """Simple whitespace + punctuation tokenizer."""
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def build_vocab(texts: list, min_freq: int = MIN_FREQ) -> dict:
    """Build word→index vocabulary from list of texts."""
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)
    return vocab


def load_glove(glove_path: str, vocab: dict, embed_dim: int = EMBED_DIM) -> np.ndarray:
    """Load GloVe embeddings for words in vocab. OOV words get random vectors."""
    embeddings = np.random.uniform(-0.25, 0.25, (len(vocab), embed_dim)).astype(np.float32)
    embeddings[0] = 0.0  # PAD = zero vector

    found = 0
    print(f"Loading GloVe from {glove_path}...")
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            if word in vocab:
                embeddings[vocab[word]] = np.array(parts[1:], dtype=np.float32)
                found += 1

    print(f"GloVe coverage: {found}/{len(vocab)} vocab words ({found/len(vocab)*100:.1f}%)")
    return embeddings


def text_to_indices(text: str, vocab: dict, max_len: int = MAX_LEN) -> list:
    """Convert text to list of vocab indices, padded/truncated to max_len."""
    tokens = tokenize(text)[:max_len]
    indices = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    # pad
    indices += [vocab[PAD_TOKEN]] * (max_len - len(indices))
    return indices


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LIARDataset(Dataset):
    def __init__(self, texts: list, labels: list, vocab: dict, max_len: int = MAX_LEN):
        self.indices = [text_to_indices(t, vocab, max_len) for t in texts]
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.indices[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],  dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TextCNN(nn.Module):
    """
    Kim (2014) TextCNN for sentence classification.
    Multiple filter sizes → max-over-time pooling → dropout → FC.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_filters: int,
        filter_sizes: list,
        num_classes: int,
        dropout: float,
        pretrained_embeddings: np.ndarray = None,
        freeze_embeddings: bool = True,
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=num_filters,
                kernel_size=fs,
            )
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)            # (batch, seq_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)    # (batch, embed_dim, seq_len)

        pooled = []
        for conv in self.convs:
            c = torch.relu(conv(embedded))      # (batch, num_filters, seq_len - fs + 1)
            c = c.max(dim=2).values             # (batch, num_filters)
            pooled.append(c)

        cat = torch.cat(pooled, dim=1)          # (batch, num_filters * len(filter_sizes))
        out = self.dropout(cat)
        return self.fc(out)                     # (batch, num_classes)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(labels, preds):
    report = classification_report(
        labels, preds,
        labels=[0, 1],
        target_names=["FAKE", "TRUE"],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        gmean = np.sqrt(sensitivity * specificity)
    else:
        gmean = 0.0

    return {
        "accuracy":          accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision":         report["macro avg"]["precision"],
        "recall":            report["macro avg"]["recall"],
        "f1":                report["macro avg"]["f1-score"],
        "precision_fake":    report["FAKE"]["precision"],
        "recall_fake":       report["FAKE"]["recall"],
        "f1_fake":           report["FAKE"]["f1-score"],
        "precision_true":    report["TRUE"]["precision"],
        "recall_true":       report["TRUE"]["recall"],
        "f1_true":           report["TRUE"]["f1-score"],
        "gmean":             gmean,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def run_fold(
    fold_idx: int,
    train_texts: list, train_labels: list,
    val_texts: list,   val_labels: list,
    vocab: dict,
    embeddings: np.ndarray,
    args,
    device: str,
) -> dict:
    train_ds = LIARDataset(train_texts, train_labels, vocab)
    val_ds   = LIARDataset(val_texts,   val_labels,   vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

    model = TextCNN(
        vocab_size=len(vocab),
        embed_dim=EMBED_DIM,
        num_filters=NUM_FILTERS,
        filter_sizes=FILTER_SIZES,
        num_classes=2,
        dropout=DROPOUT,
        pretrained_embeddings=embeddings,
        freeze_embeddings=True,
    ).to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    best_f1    = -1.0
    best_state = None
    best_metrics = {}

    for epoch in range(1, args.epochs + 1):
        # train
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                logits = model(x)
                preds = torch.argmax(logits, dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(y.tolist())

        metrics = compute_metrics(all_labels, all_preds)
        avg_loss = train_loss / len(train_loader)

        print(f"  Epoch {epoch:02d} | loss={avg_loss:.4f} | "
              f"acc={metrics['accuracy']:.4f} | "
              f"f1={metrics['f1']:.4f} | "
              f"f1_fake={metrics['f1_fake']:.4f}")

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics

    print(f"\n  Fold {fold_idx} best F1: {best_f1:.4f}")
    return best_metrics, best_state


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # load data
    df = pd.read_csv(args.train_csv, usecols=lambda c: c != "original_label")
    df = df[["statement", "label"]].dropna().reset_index(drop=True)
    texts  = df["statement"].tolist()
    labels = df["label"].tolist()
    print(f"Loaded {len(df)} samples.")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}\n")

    # build vocab from training data
    print("Building vocabulary...")
    vocab = build_vocab(texts, min_freq=MIN_FREQ)
    print(f"Vocabulary size: {len(vocab)}\n")

    # load GloVe
    embeddings = load_glove(args.glove_path, vocab, embed_dim=EMBED_DIM)

    # 5×2 repeated k-fold CV
    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)

    all_metrics = []
    best_overall_f1    = -1.0
    best_overall_state = None
    fold_idx = 0

    for train_idx, val_idx in rkf.split(df):
        fold_idx += 1
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx} / 10")
        print(f"{'='*60}")

        train_texts  = [texts[i]  for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_texts    = [texts[i]  for i in val_idx]
        val_labels   = [labels[i] for i in val_idx]

        metrics, state = run_fold(
            fold_idx,
            train_texts, train_labels,
            val_texts,   val_labels,
            vocab, embeddings, args, device,
        )
        all_metrics.append(metrics)

        if metrics["f1"] > best_overall_f1:
            best_overall_f1    = metrics["f1"]
            best_overall_state = state

    # aggregate
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (mean ± std across all folds)")
    print(f"{'='*60}")

    metric_layout = [
        ("Precision", "precision", "precision_fake", "precision_true"),
        ("Recall",    "recall",    "recall_fake",    "recall_true"),
        ("F1",        "f1",        "f1_fake",        "f1_true"),
        ("Accuracy",          "accuracy",          None, None),
        ("Balanced Accuracy", "balanced_accuracy", None, None),
        ("G-mean",            "gmean",             None, None),
    ]

    summary = {}
    for _, avg_key, fake_key, true_key in metric_layout:
        for key in [k for k in [avg_key, fake_key, true_key] if k]:
            values = [m[key] for m in all_metrics]
            summary[key] = (np.mean(values), np.std(values))

    print("  Metric              Average            Label:FAKE         Label:TRUE")
    for name, avg_key, fake_key, true_key in metric_layout:
        avg_s  = f"{summary[avg_key][0]:.3f} ± {summary[avg_key][1]:.3f}"
        fake_s = f"{summary[fake_key][0]:.3f} ± {summary[fake_key][1]:.3f}" if fake_key else "-"
        true_s = f"{summary[true_key][0]:.3f} ± {summary[true_key][1]:.3f}" if true_key else "-"
        print(f"  {name:<19} {avg_s:<18} {fake_s:<18} {true_s}")

    # save results CSV
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for name, avg_key, fake_key, true_key in metric_layout:
        rows.append({
            "metric":       name,
            "average_mean": summary[avg_key][0],
            "average_std":  summary[avg_key][1],
            "fake_mean":    summary[fake_key][0] if fake_key else np.nan,
            "fake_std":     summary[fake_key][1] if fake_key else np.nan,
            "true_mean":    summary[true_key][0] if true_key else np.nan,
            "true_std":     summary[true_key][1] if true_key else np.nan,
        })
    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "cv_results.csv"), index=False)

    # save best model
    best_dir = os.path.join(args.out_dir, "best_model")
    os.makedirs(best_dir, exist_ok=True)

    # save model weights
    torch.save(best_overall_state, os.path.join(best_dir, "textcnn_weights.pt"))

    # save vocab
    with open(os.path.join(best_dir, "vocab.json"), "w") as f:
        json.dump(vocab, f)

    # save embeddings
    np.save(os.path.join(best_dir, "embeddings.npy"), embeddings)

    # save config
    config = {
        "vocab_size":    len(vocab),
        "embed_dim":     EMBED_DIM,
        "num_filters":   NUM_FILTERS,
        "filter_sizes":  FILTER_SIZES,
        "num_classes":   2,
        "dropout":       DROPOUT,
        "max_len":       MAX_LEN,
        "id2label":      ID2LABEL,
        "label2id":      LABEL2ID,
    }
    with open(os.path.join(best_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nBest model saved to {best_dir}")
    print(f"  - textcnn_weights.pt")
    print(f"  - vocab.json")
    print(f"  - embeddings.npy")
    print(f"  - config.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TextCNN (Kim 2014) on LIAR binary dataset."
    )
    parser.add_argument(
        "--train_csv",
        default="/cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_train.csv",
    )
    parser.add_argument(
        "--glove_path",
        default="/cluster/home/williasf/xai-adversarial-attacks/data/glove/glove.6B.100d.txt",
        help="Path to GloVe embeddings file",
    )
    parser.add_argument(
        "--out_dir",
        default="/cluster/home/williasf/xai-adversarial-attacks/models/textcnn_model",
    )
    parser.add_argument("--epochs",     type=int,   default=10)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-3)
    args = parser.parse_args()
    main(args)