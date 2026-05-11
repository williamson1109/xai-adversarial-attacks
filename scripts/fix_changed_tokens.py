"""
fix_changed_tokens.py
=====================
Recomputes 'Changed Token(s)' using the modification detail CSV directly.

For each experiment:
  - Original = Original Sentence from the first iteration row
  - Final    = Modified Sentence from the last "← new best" row
                (or last row if no new best)

Usage:
    python scripts/fix_changed_tokens.py \
        --log results/llm_attack/llm_experiment_log.csv \
        --mod_detail results/llm_attack/llm_modification_detail.csv
"""

import argparse
import difflib
import pandas as pd


def find_changed_tokens(original: str, modified: str) -> str:
    orig_words = str(original).split()
    mod_words  = str(modified).split()
    matcher = difflib.SequenceMatcher(None, orig_words, mod_words)
    changed = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            oc, mc = orig_words[i1:i2], mod_words[j1:j2]
            for o, m in zip(oc, mc):
                changed.append(f'"{o}"→"{m}"')
            for o in oc[len(mc):]:
                changed.append(f'"{o}"→[removed]')
            for m in mc[len(oc):]:
                changed.append(f'[added]→"{m}"')
        elif tag == 'delete':
            for o in orig_words[i1:i2]:
                changed.append(f'"{o}"→[removed]')
        elif tag == 'insert':
            for m in mod_words[j1:j2]:
                changed.append(f'[added]→"{m}"')
    return ", ".join(changed) if changed else "no change detected"


def main(args):
    exp_df    = pd.read_csv(args.log)
    detail_df = pd.read_csv(args.mod_detail)

    print(f"Experiment log:      {len(exp_df)} rows")
    print(f"Modification detail: {len(detail_df)} rows")

    # build lookups per exp_num
    orig_lookup  = {}   # exp_num -> original sentence (first iteration)
    final_lookup = {}   # exp_num -> final modified sentence (last new best)

    for exp_num, group in detail_df.groupby('Exp #'):
        group = group.sort_values('Sent #')

        # original = Original Sentence from first row
        orig_lookup[exp_num] = str(group.iloc[0]['Original Sentence'])

        # final = Modified Sentence from last "← new best" row
        new_best = group[group['Notes/observations'].str.contains('new best', na=False)]
        if len(new_best) > 0:
            final_lookup[exp_num] = str(new_best.iloc[-1]['Modified Sentence'])
        else:
            # no improvement — final = last modified sentence
            final_lookup[exp_num] = str(group.iloc[-1]['Modified Sentence'])

    # recompute Changed Token(s) for each experiment
    fixed = []
    missing = 0
    for _, row in exp_df.iterrows():
        exp_num = int(row['Exp #'])
        orig  = orig_lookup.get(exp_num, '')
        final = final_lookup.get(exp_num, '')

        if not orig or not final:
            fixed.append('no change detected')
            missing += 1
            continue

        fixed.append(find_changed_tokens(orig, final))

    exp_df['Changed Token(s)'] = fixed

    if missing > 0:
        print(f"Warning: {missing} experiments had no matching detail rows")

    out_path = args.log.replace('.csv', '_fixed.csv')
    exp_df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    changed = exp_df[exp_df['Changed Token(s)'] != 'no change detected']
    print(f"Rows with changes: {len(changed)} / {len(exp_df)}")
    print("\nSample fixes:")
    for _, row in changed.head(5).iterrows():
        print(f"  Exp #{row['Exp #']} (Article {row['Article Idx']}): "
              f"{str(row['Changed Token(s)'])[:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--log',        required=True,
                        help='Path to llm_experiment_log CSV')
    parser.add_argument('--mod_detail', required=True,
                        help='Path to llm_modification_detail CSV')
    args = parser.parse_args()
    main(args)