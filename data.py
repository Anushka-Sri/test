"""
data.py — Dataset loading and splitting for both models.

Model 1 uses Jigsaw Toxic Comment Classification Challenge (train.csv).
Model 2 uses HateXplain (dataset.json + post_id_divisions.json).

Download:
  Jigsaw:     https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data
              -> put train.csv in data/jigsaw/
  HateXplain: https://github.com/hate-alert/HateXplain/tree/master/Data
              -> put dataset.json and post_id_divisions.json in data/hatexplain/

If the files are missing, every script can run with --demo, which uses a tiny
synthetic dataset just to prove the code runs end-to-end (the scores it
produces mean nothing).
"""
import json
import os
import random
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

JIGSAW_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
HATEXPLAIN_CLASSES = ["hate", "offensive", "normal"]   # index 0, 1, 2


# --------------------------------------------------------------------------- #
# Light normalization (report §3 step 2): fix Unicode and whitespace but keep
# punctuation and casing — those carry signal ("YOU idiot!!!" vs "you idiot").
# --------------------------------------------------------------------------- #
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


# --------------------------------------------------------------------------- #
# Model 1 — Jigsaw
# --------------------------------------------------------------------------- #
def load_jigsaw(path="data/jigsaw/train.csv", val_size=0.1, test_size=0.1):
    """Return (train_df, val_df, test_df), each with 'text' + six label columns.

    The official Kaggle test labels are released separately, so we carve a
    deterministic train/val/test split out of train.csv (report §7.1).
    """
    df = pd.read_csv(path)
    df["text"] = df["comment_text"].map(normalize)

    # Data cleaning rules (documented, per report §7.1)
    n0 = len(df)
    df = df[df["text"].str.len() > 0]                  # drop empty comments
    df = df.drop_duplicates(subset="text")             # drop exact duplicate texts
    print(f"[jigsaw] {n0} rows -> {len(df)} after dropping empty/duplicate text")

    df = df[["text"] + JIGSAW_LABELS].reset_index(drop=True)

    # Multi-label data can't be stratified on all 6 labels at once with plain
    # sklearn, so stratify on "any toxic label" to keep the positive rate stable.
    strat = (df[JIGSAW_LABELS].sum(axis=1) > 0).astype(int)
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, random_state=SEED, stratify=strat)
    strat_tmp = (temp_df[JIGSAW_LABELS].sum(axis=1) > 0).astype(int)
    val_df, test_df = train_test_split(
        temp_df, test_size=test_size / (val_size + test_size),
        random_state=SEED, stratify=strat_tmp)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def describe_jigsaw(df):
    """Label frequency + co-occurrence — inspect before training (report §7.1)."""
    print("\nLabel frequency (% of comments):")
    print((df[JIGSAW_LABELS].mean() * 100).round(2).to_string())
    print("\nLabel co-occurrence counts:")
    Y = df[JIGSAW_LABELS].values
    print(pd.DataFrame(Y.T @ Y, index=JIGSAW_LABELS, columns=JIGSAW_LABELS).to_string())


# --------------------------------------------------------------------------- #
# Model 2 — HateXplain
# --------------------------------------------------------------------------- #
_HX_MAP = {"hatespeech": "hate", "offensive": "offensive", "normal": "normal"}


def load_hatexplain(folder="data/hatexplain"):
    """Return (train_df, val_df, test_df) with columns 'text', 'label' (0/1/2).

    Each HateXplain post has 3 annotator labels. The gold label is the
    majority vote; posts where all 3 annotators disagree have no majority and
    are dropped — the same rule the HateXplain authors use. We use the
    official train/val/test post-id division shipped with the dataset.
    """
    with open(os.path.join(folder, "dataset.json")) as f:
        data = json.load(f)
    with open(os.path.join(folder, "post_id_divisions.json")) as f:
        divisions = json.load(f)

    rows, dropped = {}, 0
    for post_id, post in data.items():
        votes = Counter(_HX_MAP[a["label"]] for a in post["annotators"])
        label, count = votes.most_common(1)[0]
        if count < 2:          # no majority
            dropped += 1
            continue
        rows[post_id] = {"text": normalize(" ".join(post["post_tokens"])),
                         "label": HATEXPLAIN_CLASSES.index(label)}
    print(f"[hatexplain] kept {len(rows)} posts, dropped {dropped} with no majority label")

    def split(name):
        return pd.DataFrame([rows[i] for i in divisions[name] if i in rows])

    return split("train"), split("val"), split("test")


def describe_hatexplain(df):
    counts = df["label"].value_counts().sort_index()
    print("\nClass distribution:")
    for i, n in counts.items():
        print(f"  {HATEXPLAIN_CLASSES[i]:<10} {n:>6}  ({n / len(df):.1%})")


# --------------------------------------------------------------------------- #
# Synthetic demo data — ONLY for checking that the pipeline runs.
# Placeholder tokens stand in for real abusive vocabulary.
# --------------------------------------------------------------------------- #
_NEUTRAL = ("the article was edited yesterday please add a source for this claim "
            "thanks for the update i think the section needs more detail "
            "good point see the talk page for the discussion").split()
_CUES = {
    "toxic":         ["nastyword", "awfulword"],
    "severe_toxic":  ["extremeword"],
    "obscene":       ["swearword", "crudeword"],
    "threat":        ["harmword"],
    "insult":        ["insultword", "mockword"],
    "identity_hate": ["groupslur"],
}


def _sentence(rng, cue_words=()):
    words = rng.choices(_NEUTRAL, k=rng.randint(6, 18)) + list(cue_words)
    rng.shuffle(words)
    return " ".join(words)


def demo_jigsaw(n=3000):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        y = {lab: 0 for lab in JIGSAW_LABELS}
        if rng.random() < 0.25:                      # ~25% toxic
            y["toxic"] = 1
            for lab, p in [("obscene", .5), ("insult", .5), ("severe_toxic", .1),
                           ("threat", .05), ("identity_hate", .08)]:
                y[lab] = int(rng.random() < p)
        cues = [rng.choice(_CUES[lab]) for lab in JIGSAW_LABELS if y[lab]]
        rows.append({"text": _sentence(rng, cues), **y})
    df = pd.DataFrame(rows)
    tr, tmp = train_test_split(df, test_size=0.2, random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED)
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def demo_hatexplain(n=3000):
    rng = random.Random(SEED)
    rows = []
    for _ in range(n):
        r = rng.random()
        if r < 0.30:
            label, cues = 0, ["groupslur", rng.choice(["nastyword", "insultword"])]
        elif r < 0.60:
            label, cues = 1, [rng.choice(["swearword", "insultword", "nastyword"])]
        else:
            label, cues = 2, []
        rows.append({"text": _sentence(rng, cues), "label": label})
    df = pd.DataFrame(rows)
    tr, tmp = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
    va, te = train_test_split(tmp, test_size=0.5, random_state=SEED, stratify=tmp["label"])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)
