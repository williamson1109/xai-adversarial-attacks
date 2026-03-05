import argparse
import pandas as pd
import os

def map_label(label):
    # Accepts string or int
    true_labels = {"true", "mostly-true", 1, 2, "1", "2"}
    false_labels = {"false", "pants-fire", 0, 5, "0", "5"}
    if label in true_labels:
        return 1
    if label in false_labels:
        return 0
    return None

def preprocess_liar(input_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
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
    df['label_bin'] = df['label'].apply(map_label)
    df = df[df['label_bin'].isin([0,1])]
    df = df[['label_bin', 'statement']].rename(columns={'label_bin': 'label'})
    out_path = os.path.join(out_dir, 'liar_binary.csv')
    df.to_csv(out_path, index=False)
    print(f"Saved binary LIAR data to {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Raw LIAR CSV/TSV file')
    parser.add_argument('--out_dir', required=True, help='Output directory for processed CSV')
    args = parser.parse_args()
    preprocess_liar(args.input, args.out_dir)
