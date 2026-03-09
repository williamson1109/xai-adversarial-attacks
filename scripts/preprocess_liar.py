
import argparse
import pandas as pd
import os
from pathlib import Path

def fetch_liar_from_hf(split="train"):
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install the 'datasets' package: pip install datasets")
    ds = load_dataset("liar")
    data = ds[split]
    df = pd.DataFrame(data)
    return df

def map_label(label):
    # Accepts string or int
    true_labels = {"true", "mostly-true", 1, 2, "1", "2"}
    false_labels = {"false", "pants-fire", 0, 5, "0", "5"}
    if label in true_labels:
        return 1
    if label in false_labels:
        return 0
    return None


def preprocess_liar(input_path, out_dir, from_hf=False):
    os.makedirs(out_dir, exist_ok=True)
    df = None
    if from_hf or not Path(input_path).is_file():
        print("Fetching LIAR dataset from HuggingFace...")
        df = fetch_liar_from_hf("train")
    else:
        df = pd.read_csv(input_path, sep=None, engine='python')
    if 'label' not in df.columns:
        # Try to infer label column
        for col in df.columns:
            if 'label' in col.lower():
                df.rename(columns={col: 'label'}, inplace=True)
    if 'statement' not in df.columns:
        for col in df.columns:
            if 'statement' in col.lower() or 'text' in col.lower():
                df.rename(columns={col: 'statement'}, inplace=True)
    # Always derive binary label from original_label
    # Try to get original_label column (0-6) from df
    if 'original_label' in df.columns:
        df['original_label'] = df['original_label']
    elif 'label' in df.columns:
        df['original_label'] = df['label']
    else:
        raise ValueError("No original_label or label column found in input data.")

    # Only keep rows with original_label in [0,1,4,5]
    df = df[df['original_label'].isin([0,1,4,5])].copy()

    # Compute binary label
    def liar_binary_label(orig):
        if orig in [0,1]:
            return 1
        elif orig in [4,5]:
            return 0
        else:
            return None
    df['label'] = df['original_label'].apply(liar_binary_label)

    # Remove any rows with label None (shouldn't happen)
    df = df[df['label'].isin([0,1])]

    # Validation: label==1 only for original_label in [0,1], label==0 only for [4,5]
    valid_1 = df[df['label']==1]['original_label'].isin([0,1]).all()
    valid_0 = df[df['label']==0]['original_label'].isin([4,5]).all()
    if not (valid_1 and valid_0):
        raise ValueError("Inconsistent binary label mapping detected!")

    # Diagnostic printout
    print("Label vs. Original Label crosstab:")
    print(pd.crosstab(df["label"], df["original_label"]))

    # Output CSV: label,statement,original_label
    out_path = os.path.join(out_dir, 'liar_binary.csv')
    df_out = df[['label', 'statement', 'original_label']]
    df_out.to_csv(out_path, index=False)
    print(f"Saved binary LIAR data to {out_path} ({len(df_out)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Raw LIAR CSV/TSV file (or dummy if using --from_hf)')
    parser.add_argument('--out_dir', required=True, help='Output directory for processed CSV')
    parser.add_argument('--from_hf', action='store_true', help='Fetch LIAR from HuggingFace instead of local file')
    args = parser.parse_args()
    preprocess_liar(args.input, args.out_dir, from_hf=args.from_hf)
