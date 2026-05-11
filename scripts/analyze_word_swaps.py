"""
analyze_word_swaps.py
=====================
Extracts meaningful word substitutions from the modification detail CSV
by diffing Original Sentence vs Modified Sentence per iteration.

Filters out:
  - Stopwords
  - Punctuation-only changes
  - Cases where more than 3 words changed (too noisy)

Usage:
    python scripts/analyze_word_swaps.py \
        --detail results/llm_attack/llm_modification_detail.csv \
        --out_dir results/analysis/

    python scripts/analyze_word_swaps.py \
        --detail results/llm_attack_roberta/llm_modification_detail_roberta_model.csv \
        --out_dir results/analysis/

    python scripts/analyze_word_swaps.py \
        --detail results/llm_attack_textcnn/llm_modification_detail_textcnn_model.csv \
        --out_dir results/analysis/
"""

import argparse
import difflib
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

STOPWORDS = {
    'the','a','an','of','in','to','and','is','it','that','for','on','are',
    'was','as','at','be','by','this','with','from','or','but','not','have',
    'had','he','she','they','we','you','his','her','its','our','their','all',
    'been','has','will','do','did','does','than','then','there','so','if',
    'up','out','my','your','no','more','about','also','into','after','when',
    'who','what','which','can','could','would','should','may','might','shall',
    'must','just','now','like','over','some','any','most','other','such',
    'how','these','those','us','very','an','were','me','him','them','too',
    'i','s','t','re','ve','ll','d'
}

PUNCT_RE = re.compile(r'^[^\w]+$')


def is_meaningful(word: str) -> bool:
    """Return True if word is worth tracking."""
    w = word.lower().strip('.,!?;:\'"()-')
    if not w:
        return False
    if w in STOPWORDS:
        return False
    if PUNCT_RE.match(w):
        return False
    if len(w) <= 1:
        return False
    return True


def get_swaps(original: str, modified: str):
    """Extract (old_word, new_word) pairs from a single iteration diff.
    Only returns substitutions, not insertions or deletions.
    Skips if more than 3 words changed (too noisy / sentence restructured)."""
    orig_words = str(original).split()
    mod_words  = str(modified).split()

    matcher = difflib.SequenceMatcher(None, orig_words, mod_words)
    swaps = []
    total_changes = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            continue
        total_changes += max(i2 - i1, j2 - j1)
        if tag == 'replace':
            oc, mc = orig_words[i1:i2], mod_words[j1:j2]
            for o, m in zip(oc, mc):
                swaps.append((o.lower().strip('.,!?;:\'"()-'),
                              m.lower().strip('.,!?;:\'"()-')))

    # skip noisy iterations where too much changed
    if total_changes > 3:
        return []

    return [
        (o, m) for o, m in swaps
        if is_meaningful(o) and is_meaningful(m) and o != m
    ]


def main(args):
    df = pd.read_csv(args.detail)
    exp_df = pd.read_csv(args.exp_log)
    print(f"Loaded {len(df)} modification detail rows")
    print(f"Loaded {len(exp_df)} experiment log rows")

    model_tag = os.path.splitext(os.path.basename(args.detail))[0]

    # build flip direction lookup: exp_num -> (flipped, orig_label, new_label)
    flip_lookup = {}
    for _, row in exp_df.iterrows():
        exp_num   = int(row['Exp #'])
        flipped   = str(row['Flipped?']).upper() == 'YES'
        orig_pred = str(row['Orig Pred']).strip()
        new_pred  = str(row['New Pred']).strip()
        flip_lookup[exp_num] = (flipped, orig_pred, new_pred)

    # collect swaps overall and by flip direction
    all_swaps        = Counter()
    fake_to_true     = Counter()  # FAKE→TRUE flips
    true_to_fake     = Counter()  # TRUE→FAKE flips
    no_flip_swaps    = Counter()  # not flipped

    for _, row in df.iterrows():
        orig = str(row['Original Sentence'])
        mod  = str(row['Modified Sentence'])
        exp_num = int(row['Exp #'])

        if orig == mod or 'no change' in mod.lower():
            continue

        swaps = get_swaps(orig, mod)
        if not swaps:
            continue

        flipped, orig_pred, new_pred = flip_lookup.get(exp_num, (False, '', ''))

        for old, new in swaps:
            all_swaps[(old, new)] += 1
            if flipped:
                if orig_pred.lower() == 'fake' and new_pred.lower() == 'real':
                    fake_to_true[(old, new)] += 1
                elif orig_pred.lower() == 'real' and new_pred.lower() == 'fake':
                    true_to_fake[(old, new)] += 1
            else:
                no_flip_swaps[(old, new)] += 1

    # print results
    print(f"\nTotal unique swap pairs: {len(all_swaps)}")

    print(f"\n── FAKE→TRUE flips ({sum(fake_to_true.values())} total swaps) ──")
    for (o, n), c in fake_to_true.most_common(15):
        print(f"  '{o}' → '{n}': {c}x")

    print(f"\n── TRUE→FAKE flips ({sum(true_to_fake.values())} total swaps) ──")
    for (o, n), c in true_to_fake.most_common(15):
        print(f"  '{o}' → '{n}': {c}x")

    print(f"\n── Not flipped ({sum(no_flip_swaps.values())} total swaps) ──")
    for (o, n), c in no_flip_swaps.most_common(10):
        print(f"  '{o}' → '{n}': {c}x")

    # ── Chart: side by side FAKE→TRUE vs TRUE→FAKE ──
    os.makedirs(args.out_dir, exist_ok=True)

    def make_chart(counter, title, color, filename, top_n=15):
        top = counter.most_common(top_n)
        if not top:
            print(f"No data for {title}")
            return
        labels = [f'"{o}" → "{n}"'.replace('$', r'\$') for (o, n), _ in top][::-1]
        counts = [c for _, c in top][::-1]

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FAFAFA')
        bars = ax.barh(labels, counts, color=color, edgecolor='none', height=0.65)
        for bar, count in zip(bars, counts):
            ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                    str(count), va='center', ha='left', fontsize=10,
                    fontweight='bold', color='#333333')
        ax.set_xlabel('Frequency', fontsize=12, color='#555555', labelpad=10)
        ax.set_title(title, fontsize=13, fontweight='bold', color='#2C3E50', pad=16)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#DDDDDD')
        ax.tick_params(axis='y', labelsize=10, colors='#333333')
        ax.tick_params(axis='x', colors='#888888')
        ax.xaxis.grid(True, color='#EEEEEE', linewidth=0.8, linestyle='--')
        ax.set_axisbelow(True)
        ax.set_xlim(0, max(counts) * 1.25)
        plt.tight_layout()
        path = os.path.join(args.out_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to {path}")
        plt.close()

    make_chart(fake_to_true, f'Word Swaps in FAKE→TRUE Flips\n({model_tag})',
               '#7DB8FF', f'swaps_fake_to_true_{model_tag}.png')
    make_chart(true_to_fake, f'Word Swaps in TRUE→FAKE Flips\n({model_tag})',
               '#FF9999', f'swaps_true_to_fake_{model_tag}.png')
    make_chart(all_swaps,    f'All Word Swaps (Top 20)\n({model_tag})',
               '#95D5B2', f'swaps_all_{model_tag}.png', top_n=20)

    # save CSV
    rows = []
    for (o, n), c in all_swaps.most_common():
        rows.append({
            'Original Word': o, 'Replacement Word': n, 'Count': c,
            'FAKE→TRUE': fake_to_true.get((o, n), 0),
            'TRUE→FAKE': true_to_fake.get((o, n), 0),
            'No Flip':   no_flip_swaps.get((o, n), 0),
        })
    csv_path = os.path.join(args.out_dir, f'word_swaps_{model_tag}.csv')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Swap data saved to {csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--detail',  required=True,
                        help='Path to llm_modification_detail CSV')
    parser.add_argument('--exp_log', required=True,
                        help='Path to llm_experiment_log CSV (for flip direction)')
    parser.add_argument('--out_dir', default='results/analysis/',
                        help='Output directory for charts and CSV')
    args = parser.parse_args()
    main(args)