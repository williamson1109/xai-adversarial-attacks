"""
POS Analysis of Flipped vs Not-Flipped Sentences
=================================================
Run this on IDUN where NLTK corpora are available.
Produces per-model charts AND a combined cross-model comparison.

Requires: nltk, pandas, matplotlib

Usage:
    python scripts/pos_analysis.py \
        --distilbert_log    results/llm_attack/llm_experiment_log_fixed.csv \
        --distilbert_detail results/llm_attack/llm_modification_detail.csv \
        --roberta_log       results/llm_attack_roberta/llm_experiment_log_roberta_model_fixed.csv \
        --roberta_detail    results/llm_attack_roberta/llm_modification_detail_roberta_model.csv \
        --textcnn_log       results/llm_attack_textcnn/llm_experiment_log_textcnn_model_fixed.csv \
        --textcnn_detail    results/llm_attack_textcnn/llm_modification_detail_textcnn_model.csv \
        --out_dir           results/analysis/
"""

import argparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nltk
nltk.download("averaged_perceptron_tagger_eng", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("punkt", quiet=True)

from nltk import pos_tag, word_tokenize

NOUN_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
ADJ_TAGS  = {"JJ", "JJR", "JJS"}
ADV_TAGS  = {"RB", "RBR", "RBS"}
STAT_RE   = re.compile(r'\b\d+[\.,]?\d*\b|%|\$')


def get_pos_counts(text):
    try:
        tokens = word_tokenize(str(text))
        tags   = pos_tag(tokens)
        nouns  = sum(1 for _, t in tags if t in NOUN_TAGS)
        verbs  = sum(1 for _, t in tags if t in VERB_TAGS)
        adjs   = sum(1 for _, t in tags if t in ADJ_TAGS)
        advs   = sum(1 for _, t in tags if t in ADV_TAGS)
        stats  = len(STAT_RE.findall(str(text)))
        total  = len([w for w in tokens if w.isalpha()])
        denom  = max(total, 1)
        return {
            "nouns": nouns, "verbs": verbs, "adjs": adjs,
            "advs": advs, "stats": stats, "total": total,
            "noun_ratio": nouns / denom,
            "verb_ratio": verbs / denom,
            "adj_ratio":  adjs  / denom,
            "adv_ratio":  advs  / denom,
            "stat_ratio": stats / denom,
        }
    except Exception:
        return None


def load_sentences(exp_log_path, detail_path):
    exp_df    = pd.read_csv(exp_log_path)
    detail_df = pd.read_csv(detail_path)
    exp_df["flipped"] = exp_df["Flipped?"].str.upper() == "YES"
    orig_sentences = (
        detail_df.sort_values("Sent #")
        .groupby("Exp #").first()["Original Sentence"].to_dict()
    )
    exp_df["orig_sentence"] = exp_df["Exp #"].map(orig_sentences)
    return exp_df


def run_pos_analysis(exp_df, model_name):
    print(f"\nRunning POS tagging for {model_name} ({len(exp_df)} sentences)...")
    results = []
    for _, row in exp_df.iterrows():
        sent = str(row.get("orig_sentence", ""))
        if not sent.strip():
            continue
        counts = get_pos_counts(sent)
        if counts is None or counts["total"] == 0:
            continue
        counts["exp"]     = row["Exp #"]
        counts["flipped"] = row["flipped"]
        counts["model"]   = model_name
        results.append(counts)
    df = pd.DataFrame(results)
    print(f"  Processed {len(df)} | Flipped: {df['flipped'].sum()} | Not: {(~df['flipped']).sum()}")
    print(f"\n  Mean POS ratios:")
    for col, label in [("noun_ratio","Nouns"),("verb_ratio","Verbs"),
                       ("adj_ratio","Adjectives"),("adv_ratio","Adverbs"),
                       ("stat_ratio","Statistics")]:
        f  = df[df["flipped"]][col].mean()
        nf = df[~df["flipped"]][col].mean()
        print(f"    {label:14s}: Flipped={f:.3f}  Not={nf:.3f}  Diff={f-nf:+.3f}")
    return df


def plot_per_model(df, model_name, color_flipped, color_not, out_dir):
    pos_types  = ["Nouns", "Verbs", "Adjectives", "Adverbs", "Statistics"]
    ratio_cols = ["noun_ratio","verb_ratio","adj_ratio","adv_ratio","stat_ratio"]
    f_means  = [df[df["flipped"]][c].mean() * 100 for c in ratio_cols]
    nf_means = [df[~df["flipped"]][c].mean() * 100 for c in ratio_cols]
    x = np.arange(len(pos_types))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    bars1 = ax.bar(x - width/2, f_means,  width, color=color_flipped, edgecolor="none",
                   label=f"Flipped (n={df['flipped'].sum()})")
    bars2 = ax.bar(x + width/2, nf_means, width, color=color_not,     edgecolor="none",
                   label=f"Not Flipped (n={(~df['flipped']).sum()})")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                    f"{bar.get_height():.1f}%", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(pos_types, fontsize=12)
    ax.set_ylabel("Mean % of Words in Sentence", fontsize=11, color="#555555")
    ax.set_title(f"POS Composition by Flip Outcome — {model_name}",
                 fontsize=13, fontweight="bold", color="#2C3E50", pad=16)
    ax.legend(fontsize=11, framealpha=0.7, edgecolor="#DDDDDD")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#DDDDDD")
    ax.spines["bottom"].set_color("#DDDDDD")
    ax.tick_params(colors="#888888")
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(max(f_means), max(nf_means)) * 1.3)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pos_flip_{model_name.lower()}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close()


def plot_combined(all_dfs, model_names, model_colors, out_dir):
    pos_types  = ["Nouns", "Verbs", "Adjectives", "Adverbs", "Statistics"]
    ratio_cols = ["noun_ratio","verb_ratio","adj_ratio","adv_ratio","stat_ratio"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle("POS Composition by Flip Outcome Across Models",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.02)
    x = np.arange(len(pos_types))
    width = 0.35
    for ax, df, model_name, (cf, cn) in zip(axes, all_dfs, model_names, model_colors):
        ax.set_facecolor("#FAFAFA")
        f_means  = [df[df["flipped"]][c].mean() * 100 for c in ratio_cols]
        nf_means = [df[~df["flipped"]][c].mean() * 100 for c in ratio_cols]
        bars1 = ax.bar(x - width/2, f_means,  width, color=cf, edgecolor="none",
                       label=f"Flipped (n={df['flipped'].sum()})")
        bars2 = ax.bar(x + width/2, nf_means, width, color=cn, edgecolor="none",
                       label="Not Flipped")
        for bars in [bars1, bars2]:
            for bar in bars:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                        f"{bar.get_height():.1f}%", ha="center", va="bottom",
                        fontsize=7.5, fontweight="bold", color="#333333")
        ax.set_title(model_name, fontsize=13, fontweight="bold", color="#2C3E50")
        ax.set_xticks(x)
        ax.set_xticklabels(pos_types, fontsize=9, rotation=15, ha="right")
        ax.set_ylabel("Mean % of Words in Sentence", fontsize=10, color="#555555")
        ax.legend(fontsize=9, framealpha=0.7, edgecolor="#DDDDDD")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")
        ax.tick_params(colors="#888888")
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, linestyle="--")
        ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(out_dir, "pos_flip_combined.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"\nCombined chart saved: {path}")
    plt.close()


def plot_flip_rate_by_pos(df, model_name, out_dir):
    """Flip rate by POS ratio bins — one subplot per POS type."""
    pos_configs = [
        ("verb_ratio",  "Verb Ratio",      "#7DB8FF"),
        ("noun_ratio",  "Noun Ratio",      "#FFB347"),
        ("adj_ratio",   "Adjective Ratio", "#95D5B2"),
        ("stat_ratio",  "Statistic Ratio", "#FF9999"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(f"Flip Rate by POS Composition — {model_name}",
                 fontsize=14, fontweight="bold", color="#2C3E50", y=1.02)

    bins       = [0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 1.0]
    bin_labels = ["0-5%","5-10%","10-15%","15-20%","20-25%","25-35%","35%+"]
    overall_ae = df["flipped"].mean() * 100

    for ax, (col, label, color) in zip(axes, pos_configs):
        ax.set_facecolor("#FAFAFA")
        df["_bin"] = pd.cut(df[col], bins=bins, labels=bin_labels)
        stats = df.groupby("_bin", observed=True).agg(
            total=("flipped", "count"),
            flipped=("flipped", "sum")
        ).reset_index()
        stats["rate"] = (stats["flipped"] / stats["total"] * 100).fillna(0)

        bars = ax.bar(range(len(stats)), stats["rate"],
                      color=color, edgecolor="none", width=0.6)
        ax.axhline(y=overall_ae, color="#333333", linewidth=1.5,
                   linestyle="--", alpha=0.7,
                   label=f"Overall AE ({overall_ae:.1f}%)")

        for bar, (_, srow) in zip(bars, stats.iterrows()):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.8,
                    f"{srow['rate']:.0f}%\n(n={int(srow['total'])})",
                    ha="center", va="bottom", fontsize=8, color="#333333")

        ax.set_xticks(range(len(stats)))
        ax.set_xticklabels(list(stats["_bin"].astype(str)),
                           fontsize=8, rotation=30, ha="right")
        ax.set_ylabel("Flip Rate (%)", fontsize=10, color="#555555")
        ax.set_title(label, fontsize=11, fontweight="bold", color="#2C3E50")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#DDDDDD")
        ax.spines["bottom"].set_color("#DDDDDD")
        ax.yaxis.grid(True, color="#EEEEEE", linewidth=0.8, linestyle="--")
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(stats["rate"].max(), overall_ae) * 1.4)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"flip_rate_by_pos_{model_name.lower()}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Flip rate chart saved: {path}")
    plt.close()
    df.drop(columns=["_bin"], inplace=True, errors="ignore")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    configs = [
        ("DistilBERT", args.distilbert_log, args.distilbert_detail, "#7DB8FF", "#FFB3B3"),
        ("RoBERTa",    args.roberta_log,    args.roberta_detail,    "#7DBF7D", "#FFD9B3"),
        ("TextCNN",    args.textcnn_log,    args.textcnn_detail,    "#FF7D7D", "#D9B3FF"),
    ]
    all_dfs, model_names, model_colors = [], [], []
    for model_name, exp_log, detail, cf, cn in configs:
        print(f"\n{'='*60}\n  {model_name}\n{'='*60}")
        exp_df = load_sentences(exp_log, detail)
        df     = run_pos_analysis(exp_df, model_name)
        plot_per_model(df, model_name, cf, cn, args.out_dir)
        plot_flip_rate_by_pos(df, model_name, args.out_dir)
        csv_path = os.path.join(args.out_dir, f"pos_analysis_{model_name.lower()}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  CSV saved: {csv_path}")
        all_dfs.append(df)
        model_names.append(model_name)
        model_colors.append((cf, cn))
    plot_combined(all_dfs, model_names, model_colors, args.out_dir)
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilbert_log",    required=True)
    parser.add_argument("--distilbert_detail", required=True)
    parser.add_argument("--roberta_log",       required=True)
    parser.add_argument("--roberta_detail",    required=True)
    parser.add_argument("--textcnn_log",       required=True)
    parser.add_argument("--textcnn_detail",    required=True)
    parser.add_argument("--out_dir",           default="results/analysis/")
    main(parser.parse_args())
