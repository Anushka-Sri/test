"""
model1_tfidf_lr.py — Model 1 baseline: TF-IDF + Logistic Regression,
multi-label toxicity on Jigsaw (report §4.2).

One independent binary logistic regression per label = six sigmoid outputs.
A comment can be toxic AND insult AND obscene at the same time.

Run:
    python model1_tfidf_lr.py              # real Jigsaw data
    python model1_tfidf_lr.py --demo       # synthetic smoke test
    python model1_tfidf_lr.py --balanced   # class-weighted variant (report §4.8)
"""
import argparse
import os

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from data import JIGSAW_LABELS, demo_jigsaw, describe_jigsaw, load_jigsaw
from metrics import multilabel_report, tune_thresholds


class TfidfFeatures:
    """Word 1-2 grams + character 2-5 grams, stacked side by side.

    Char n-grams catch obfuscated spellings ("st*pid", "idiooot") that a
    word vocabulary misses. fit() is called on TRAIN ONLY to avoid leakage
    (report §7.1); val/test only call transform().
    """

    def __init__(self, max_word=50_000, max_char=50_000):
        self.word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                    max_features=max_word, sublinear_tf=True,
                                    lowercase=True, strip_accents="unicode")
        self.char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                    max_features=max_char, sublinear_tf=True,
                                    lowercase=True)

    def fit_transform(self, texts):
        return hstack([self.word.fit_transform(texts), self.char.fit_transform(texts)]).tocsr()

    def transform(self, texts):
        return hstack([self.word.transform(texts), self.char.transform(texts)]).tocsr()


class MultiLabelLogReg:
    """Six separate binary classifiers, one per Jigsaw label."""

    def __init__(self, C=4.0, balanced=False):
        self.models = {
            lab: LogisticRegression(C=C, solver="liblinear", max_iter=1000,
                                    class_weight="balanced" if balanced else None)
            for lab in JIGSAW_LABELS
        }

    def fit(self, X, Y):
        for j, lab in enumerate(JIGSAW_LABELS):
            self.models[lab].fit(X, Y[:, j])
        return self

    def predict_proba(self, X):
        # column j = P(label j = 1 | text); columns are independent, rows need not sum to 1
        return np.column_stack([self.models[lab].predict_proba(X)[:, 1] for lab in JIGSAW_LABELS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--balanced", action="store_true")
    ap.add_argument("--data", default="data/jigsaw/train.csv")
    ap.add_argument("--out", default="artifacts/model1_tfidf_lr.joblib")
    args = ap.parse_args()

    train, val, test = demo_jigsaw() if args.demo else load_jigsaw(args.data)
    print(f"train={len(train)}  val={len(val)}  test={len(test)}")
    describe_jigsaw(train)

    # Import the classes via the module name (not __main__) so the saved
    # pickle can be loaded later from predict.py.
    from model1_tfidf_lr import MultiLabelLogReg, TfidfFeatures
    feats = TfidfFeatures()
    X_tr = feats.fit_transform(train["text"])
    X_va, X_te = feats.transform(val["text"]), feats.transform(test["text"])
    Y_tr, Y_va, Y_te = (d[JIGSAW_LABELS].values for d in (train, val, test))

    clf = MultiLabelLogReg(balanced=args.balanced).fit(X_tr, Y_tr)

    # Thresholds are chosen on VALIDATION, then frozen and applied to TEST (report §13)
    thresholds = tune_thresholds(Y_va, clf.predict_proba(X_va))
    print("\nPer-label thresholds tuned on validation:",
          {l: round(float(t), 2) for l, t in zip(JIGSAW_LABELS, thresholds)})

    P_te = clf.predict_proba(X_te)
    print("\n=== TEST, default threshold 0.5 ===")
    multilabel_report(Y_te, P_te, np.full(6, 0.5), JIGSAW_LABELS)
    print("\n=== TEST, tuned thresholds ===")
    test_metrics = multilabel_report(Y_te, P_te, thresholds, JIGSAW_LABELS)

    # Save everything needed for inference together (report §7.3)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump({"features": feats, "classifier": clf, "labels": JIGSAW_LABELS,
                 "thresholds": thresholds}, args.out)
    import json
    with open(os.path.join(os.path.dirname(args.out), "metrics_model1_tfidf_lr.json"), "w") as f:
        json.dump({"model": "TF-IDF + Logistic Regression", "task": "jigsaw",
                   "demo": args.demo, "test": test_metrics}, f, indent=2)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
