"""
TextCNN Test Script
===================
Evaluates trained TextCNN on the held-out LIAR test set.

Usage:
    python scripts/test_textcnn.py \
        --model_dir models/textcnn_model/best_model \
        --test_csv data/processed/liar_test.csv
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants (must match train_textcnn.py)
# ---------------------------------------------------------------------------

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


# ---------------------------------------------------------------------------
# Utilities (duplicated from train_textcnn.py for standalone use)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def text_to_indices(text: str, vocab: dict, max_len: int) -> list:
    tokens = tokenize(text)[:max_len]
    indices = [vocab.get(t, vocab[UNK_TOKEN]) for t in tokens]
    indices += [vocab[PAD_TOKEN]] * (max_len - len(indices))
    return indices


class LIARDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len):
        self.indices = [text_to_indices(t, vocab, max_len) for t in texts]
        self.labels  = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.indices[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],  dtype=torch.long),
        )


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes,
                 num_classes, dropout, pretrained_embeddings=None,
                 freeze_embeddings=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        embedded = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.relu(conv(embedded)).max(dim=2).values
                  for conv in self.convs]
        return self.fc(self.dropout(torch.cat(pooled, dim=1)))


def g_mean(labels, preds):
    cm = confusion_matrix(labels, preds)
    per_class_recall = cm.diagonal() / cm.sum(axis=1)
    return float(np.prod(per_class_recall) ** (1.0 / len(per_class_recall)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # load config
    with open(os.path.join(args.model_dir, "config.json")) as f:
        config = json.load(f)

    # load vocab
    with open(os.path.join(args.model_dir, "vocab.json")) as f:
        vocab = json.load(f)

    # load embeddings
    embeddings = np.load(os.path.join(args.model_dir, "embeddings.npy"))

    # build model and load weights
    model = TextCNN(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        num_filters=config["num_filters"],
        filter_sizes=config["filter_sizes"],
        num_classes=config["num_classes"],
        dropout=config["dropout"],
        pretrained_embeddings=embeddings,
        freeze_embeddings=True,
    ).to(device)

    state = torch.load(
        os.path.join(args.model_dir, "textcnn_weights.pt"),
        map_location=device
    )
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # load test data
    df = pd.read_csv(args.test_csv, usecols=["statement", "label"])
    df = df.dropna().reset_index(drop=True)
    texts  = df["statement"].tolist()
    labels = df["label"].tolist()
    print(f"Loaded {len(df)} test samples.")

    # dataset & loader
    ds = LIARDataset(texts, labels, vocab, config["max_len"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    # inference
    all_preds = []
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Testing"):
            x = x.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).cpu().tolist()
            all_preds.extend(preds)

    # metrics
    acc     = accuracy_score(labels, all_preds)
    bal_acc = balanced_accuracy_score(labels, all_preds)
    gmean   = g_mean(labels, all_preds)

    print(f"\n{'='*50}")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"G-mean:            {gmean:.4f}")
    print(f"{'='*50}\n")
    print(classification_report(labels, all_preds,
                                 target_names=["FAKE", "TRUE"], digits=4))

    # save predictions
    df["predicted"] = all_preds
    out_path = os.path.join(args.model_dir, "test_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir", required=True,
        help="Path to best_model directory containing config, vocab, weights"
    )
    parser.add_argument(
        "--test_csv",
        default="/cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_test.csv",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
