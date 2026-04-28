import argparse
import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report,
)
from sklearn.metrics import confusion_matrix
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Label mapping (confirmed from raw LIAR dataset):
#   0 = false  →  binary FAKE (0)
#   5 = pants-fire  →  binary FAKE (0)
#   2 = mostly-true  →  binary REAL (1)
#   3 = true  →  binary REAL (1)
#   1 = half-true  →  DROPPED
#   4 = barely-true  →  DROPPED
# ---------------------------------------------------------------------------

ID2LABEL = {0: "FAKE", 1: "REAL"}
LABEL2ID = {"FAKE": 0, "REAL": 1}
MAX_TOKENS = 250   # Match Kozik et al. (they used 250 tokens)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load preprocessed binary LIAR CSV, dropping original_label if present."""
    df = pd.read_csv(csv_path, usecols=lambda c: c != "original_label")
    assert "statement" in df.columns, "Expected a 'statement' column in the CSV."
    assert "label" in df.columns, "Expected a 'label' column in the CSV."
    print(f"Loaded {len(df)} samples.")
    print(f"Class distribution:\n{df['label'].value_counts().to_string()}\n")
    return df[["statement", "label"]].reset_index(drop=True)


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples["statement"],
        truncation=True,
        padding="max_length",
        max_length=MAX_TOKENS,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        labels=[0, 1],
        target_names=["FAKE", "TRUE"],
        output_dict=True,
        zero_division=0,
    )

    # G-mean for binary: sqrt(sensitivity * specificity)
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


def build_model(model_name: str, device: str):
    """
    Build a sequence classification model.
    - DistilBERT: encoder is FROZEN, only classification head is trained
      (matches Kozik et al. transfer-learning setup)
    - RoBERTa (and others): full fine-tuning — all parameters trainable.
      RoBERTa requires encoder gradients to converge on this task;
      freezing the encoder causes class collapse (predicts TRUE only).
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    is_distilbert = "distilbert" in model_name.lower()

    if is_distilbert:
        # Freeze encoder — train classification head only
        HEAD_KEYWORDS = ("classifier", "pre_classifier")
        for name, param in model.named_parameters():
            if not any(kw in name for kw in HEAD_KEYWORDS):
                param.requires_grad = False
        print("DistilBERT: encoder frozen, training classification head only.")
    else:
        # Full fine-tuning for RoBERTa and other models
        print(f"{model_name}: full fine-tuning (all parameters trainable).")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    return model.to(device)


def run_fold(
    fold_idx: int,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    tokenizer,
    args,
    device: str,
) -> tuple:
    """Train and evaluate one fold. Returns metrics and model/trainer."""
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds   = val_ds.map(lambda x: tokenize_function(x, tokenizer),   batched=True)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format(  type="torch", columns=["input_ids", "attention_mask", "label"])

    fold_out = os.path.join(args.out_dir, f"fold_{fold_idx}")
    os.makedirs(fold_out, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=fold_out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=os.path.join(fold_out, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        report_to="none",
    )

    model = build_model(args.model, device)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    results = trainer.evaluate()
    print(f"\n--- Fold {fold_idx} results ---")
    fold_metric_keys = [
        "eval_precision", "eval_precision_fake", "eval_precision_true",
        "eval_recall", "eval_recall_fake", "eval_recall_true",
        "eval_f1", "eval_f1_fake", "eval_f1_true",
        "eval_accuracy", "eval_balanced_accuracy", "eval_gmean",
    ]
    for k in fold_metric_keys:
        v = results.get(k)
        if v is None:
            continue
        print(f"  {k}: {v:.4f}")
    return results, model, tokenizer


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    print(f"Model: {args.model}\n")

    df = load_data(args.train_csv)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # 5x2-fold repeated cross-validation (n_splits=2, n_repeats=5)
    rkf = RepeatedKFold(n_splits=2, n_repeats=5, random_state=42)

    all_metrics: list[dict] = []
    best_f1 = -1
    best_model = None
    best_tokenizer = None
    fold_idx = 0

    for train_idx, val_idx in rkf.split(df):
        fold_idx += 1
        print(f"\n{'='*60}")
        print(f"  Fold {fold_idx} / 10")
        print(f"{'='*60}")

        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        results, model, tok = run_fold(fold_idx, train_df, val_df, tokenizer, args, device)
        all_metrics.append(results)
        # Use F1 to select best model
        f1 = results.get("eval_f1", results.get("f1", 0.0))
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_tokenizer = tok

    # Aggregate results across all folds (mean ± std)
    print(f"\n{'='*60}")
    print("  FINAL RESULTS (mean ± std across all folds)")
    print(f"{'='*60}")
    metric_layout = [
        ("Precision", "eval_precision", "eval_precision_fake", "eval_precision_true"),
        ("Recall", "eval_recall", "eval_recall_fake", "eval_recall_true"),
        ("F1", "eval_f1", "eval_f1_fake", "eval_f1_true"),
        ("Accuracy", "eval_accuracy", None, None),
        ("Balanced Accuracy", "eval_balanced_accuracy", None, None),
        ("G-mean", "eval_gmean", None, None),
    ]
    summary = {}
    for _, avg_key, fake_key, true_key in metric_layout:
        keys = [k for k in [avg_key, fake_key, true_key] if k is not None]
        for key in keys:
            values = [m.get(key, 0.0) for m in all_metrics]
            summary[key] = (np.mean(values), np.std(values))

    def fmt_metric(mean_std):
        return f"{mean_std[0]:.3f} ± {mean_std[1]:.3f}"

    print("  Metric              Average            Label:FAKE         Label:TRUE")
    for metric_name, avg_key, fake_key, true_key in metric_layout:
        avg_str = fmt_metric(summary[avg_key])
        fake_str = fmt_metric(summary[fake_key]) if fake_key else "-"
        true_str = fmt_metric(summary[true_key]) if true_key else "-"
        print(f"  {metric_name:<19} {avg_str:<18} {fake_str:<18} {true_str}")

    # Save summary CSV
    summary_path = os.path.join(args.out_dir, "cv_results.csv")
    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for metric_name, avg_key, fake_key, true_key in metric_layout:
        avg_mean, avg_std = summary[avg_key]
        fake_mean, fake_std = summary[fake_key] if fake_key else (np.nan, np.nan)
        true_mean, true_std = summary[true_key] if true_key else (np.nan, np.nan)
        rows.append(
            {
                "metric": metric_name,
                "average_mean": avg_mean,
                "average_std": avg_std,
                "fake_mean": fake_mean,
                "fake_std": fake_std,
                "true_mean": true_mean,
                "true_std": true_std,
            }
        )
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")

    # Save best model by validation F1
    if best_model is not None:
        best_model.save_pretrained(os.path.join(args.out_dir, "best_model"))
        best_tokenizer.save_pretrained(os.path.join(args.out_dir, "best_model"))
        print(f"\nBest model (by validation F1) saved to {os.path.join(args.out_dir, 'best_model')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a fake-news classifier on LIAR (binary), "
                    "replicating Kozik et al. (2023) methodology. "
                    "Supports DistilBERT and RoBERTa via --model argument."
    )
    parser.add_argument(
        "--train_csv",
        default="/cluster/home/williasf/xai-adversarial-attacks/data/processed/liar_train.csv",
        help="Path to preprocessed LIAR training CSV",
    )
    parser.add_argument(
        "--out_dir",
        default="/cluster/home/williasf/xai-adversarial-attacks/models/liar_model",
        help="Output directory for model checkpoints and results",
    )
    parser.add_argument(
        "--model",
        default="distilbert-base-uncased",
        help="HuggingFace model name (default: distilbert-base-uncased). "
             "Use 'roberta-base' for RoBERTa.",
    )
    parser.add_argument("--epochs",     type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (default: 16)")
    args = parser.parse_args()
    main(args)