import argparse
import pandas as pd
import os
from pathlib import Path

# Actual LIAR dataset label encoding (verified against dataset distribution):
# 0 = false
# 1 = half-true
# 2 = mostly-true
# 3 = true
# 4 = barely-true
# 5 = pants-fire

FAKE_LABELS  = {0, 5}   # false, pants-fire  → binary 0
TRUE_LABELS  = {2, 3}   # mostly-true, true  → binary 1
DROP_LABELS  = {1, 4}   # half-true, barely-true → excluded

LABEL_NAMES = {
    0: "false",
    1: "half-true",
    2: "mostly-true",
    3: "true",
    4: "barely-true",
    5: "pants-fire",
}


def fetch_liar_from_hf(split="train"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' package: pip install datasets")
    ds = load_dataset("liar")
    df = pd.DataFrame(ds[split])
    return df


def resolve_columns(df):
    """Ensure 'label' and 'statement' columns exist, renaming if needed."""
    col_lower = {c.lower(): c for c in df.columns}

    if "label" not in df.columns:
        for candidate in ["original_label", "label_id"]:
            if candidate in col_lower:
                df = df.rename(columns={col_lower[candidate]: "label"})
                break
        else:
            raise ValueError("Could not find a label column in the dataset.")

    if "statement" not in df.columns:
        for candidate in ["statement", "text", "claim", "sentence"]:
            if candidate in col_lower:
                df = df.rename(columns={col_lower[candidate]: "statement"})
                break
        else:
            raise ValueError("Could not find a statement/text column in the dataset.")

    return df


def validate_labels(df):
    """Check that all original labels in the dataset are within the expected 0-5 range."""
    unexpected = set(df["original_label"].unique()) - set(LABEL_NAMES.keys())
    if unexpected:
        raise ValueError(f"Unexpected label values found: {unexpected}")


def binary_map(orig_label):
    if orig_label in FAKE_LABELS:
        return 0
    if orig_label in TRUE_LABELS:
        return 1
    return None  # should never reach here after filtering


def preprocess_liar(input_path, out_dir, from_hf=False, split="train"):
    os.makedirs(out_dir, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    if from_hf or not Path(input_path).is_file():
        print(f"Fetching LIAR '{split}' split from HuggingFace...")
        df = fetch_liar_from_hf(split)
    else:
        df = pd.read_csv(input_path, sep=None, engine="python")
        print(f"Loaded {len(df)} rows from {input_path}")

    # ── Normalise columns ─────────────────────────────────────────────────────
    df = resolve_columns(df)
    df["original_label"] = df["label"]          # preserve raw label
    df["original_label"] = df["original_label"].astype(int)

    validate_labels(df)

    # ── Show full distribution before filtering ───────────────────────────────
    print("\nOriginal label distribution:")
    dist = df["original_label"].value_counts().sort_index()
    for idx, count in dist.items():
        tag = "(DROPPED)" if idx in DROP_LABELS else ""
        print(f"  {idx} ({LABEL_NAMES[idx]:>12s}): {count:>5}  {tag}")

    # ── Drop ambiguous labels (half-true=1, barely-true=4) ────────────────────
    n_before = len(df)
    df = df[df["original_label"].isin(FAKE_LABELS | TRUE_LABELS)].copy()
    n_dropped = n_before - len(df)
    print(f"\nDropped {n_dropped} rows with ambiguous labels (half-true, barely-true)")

    # ── Map to binary ─────────────────────────────────────────────────────────
    df["label"] = df["original_label"].apply(binary_map)

    # Sanity check
    assert df["label"].isin([0, 1]).all(), "Unexpected None values in binary label column!"
    assert df[df["label"] == 0]["original_label"].isin(FAKE_LABELS).all(), "Fake mapping error!"
    assert df[df["label"] == 1]["original_label"].isin(TRUE_LABELS).all(), "True mapping error!"

    # ── Report final distribution ─────────────────────────────────────────────
    print("\nBinary label distribution after filtering:")
    print(pd.crosstab(
        df["original_label"].map(LABEL_NAMES),
        df["label"],
        colnames=["binary_label (0=fake, 1=true)"]
    ))
    counts = df["label"].value_counts()
    print(f"\n  Total fake  (0): {counts.get(0, 0)}")
    print(f"  Total true  (1): {counts.get(1, 0)}")
    print(f"  Total rows    : {len(df)}")

    balance = counts.get(0, 0) / len(df)
    if not (0.35 <= balance <= 0.65):
        print(f"\n  ⚠ Warning: class imbalance detected ({balance:.1%} fake). "
              "Consider oversampling or weighted loss.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(out_dir, "liar_binary.csv")
    df[["label", "statement", "original_label"]].to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess LIAR dataset for binary fake news classification.")
    parser.add_argument("--input",    required=True,  help="Path to raw LIAR CSV/TSV (ignored if --from_hf)")
    parser.add_argument("--out_dir",  required=True,  help="Output directory")
    parser.add_argument("--from_hf",  action="store_true", help="Fetch from HuggingFace instead of local file")
    parser.add_argument("--split",    default="train", choices=["train", "validation", "test"],
                        help="Dataset split to use when fetching from HuggingFace (default: train)")
    args = parser.parse_args()
    preprocess_liar(args.input, args.out_dir, from_hf=args.from_hf, split=args.split)