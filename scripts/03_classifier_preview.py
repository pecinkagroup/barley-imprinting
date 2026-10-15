"""
03_classifier_preview.py
========================

Cross-validated classifier preview: how well does the thickness-along-length
profile discriminate WT from lys3.c on the current training set, at the
individual-seed level?

This is a pilot for the future F2 (selfed heterozygote) analysis. The
classifier trained here will be applied to F2 seeds to score each seed as
WT-like or mutant-like; observed phenotypic ratios can then be compared
against Mendelian expectations.

INPUT
-----
thickness_profiles_per_seed.csv
    Produced by 01_extract_thickness_profiles.py. Must contain columns:
        seed, genotype ("WT" or "lys3.c"),
        length_mm, tilt_deg,
        t_000 ... t_099   (thickness in mm at 100 normalized positions)

OUTPUT
------
classifier_scores_per_seed.csv
    One row per seed with cross-validated classifier scores. Columns:
        seed, genotype, length_mm, tilt_deg,
        lda_proba     probability the seed is lys3.c under LDA
        lda_score     log-odds of lda_proba (logit)
        logit_proba   probability the seed is lys3.c under L2-logistic
        cv_fold       which of the 5 folds provided this seed's score

classifier_scores_preview.{png,pdf,svg,eps}
    Two-panel figure. Left: histogram of the LDA log-odds score, split by
    true genotype, with fitted Gaussian overlays. Right: ROC curves for
    LDA and logistic regression.

METHODS
-------
Both classifiers are fit on the 100-point thickness profile (t_000...t_099)
with stratified 5-fold cross-validation; every seed is scored using a fold
of the classifier that never saw it during training.

* LDA (Ledoit-Wolf shrinkage): standard shrunken linear discriminant. The
  shrinkage adapts to the sample size / feature-count ratio (229 seeds ×
  100 features), so no regularization tuning is required.
* Logistic regression (L2, cv=5): reported as a methodological
  cross-check. Its inverse-regularization strength C is chosen internally
  by ROC-AUC on nested CV over 10 candidate values.

Reported metrics: cross-validated overall accuracy at the p=0.5 decision
threshold, sensitivity (mutant recall), specificity (WT recall), and
ROC-AUC. Class separation on the LDA discriminant axis is reported as
Cohen's d.

DEPENDENCIES
------------
    numpy pandas scipy scikit-learn matplotlib
"""

import os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
# Embed TrueType fonts (Type 42) instead of matplotlib's default Type 3.
# Makes PDF/EPS/SVG output importable into CorelDRAW, Illustrator, Inkscape.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as text
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve


N_PROF   = 100
N_SPLITS = 5
RNG_SEED = 0


def main(in_csv, out_dir):
    df = pd.read_csv(in_csv)
    cols = [f"t_{k:03d}" for k in range(N_PROF)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {missing[:3]} ...")

    df = df[df["genotype"].isin(["WT", "lys3.c"])].copy()
    X = df[cols].values
    y = (df["genotype"] == "lys3.c").astype(int).values   # 1 = mutant
    print(f"WT     n = {(y==0).sum()}")
    print(f"lys3.c n = {(y==1).sum()}")
    print(f"Feature matrix: {X.shape[0]} seeds × {X.shape[1]} positions")

    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RNG_SEED)

    # Record which fold each seed came from (for reproducibility)
    fold_id = np.empty(len(y), dtype=int)
    for f, (_, test_idx) in enumerate(cv.split(X, y)):
        fold_id[test_idx] = f

    # -------- LDA with Ledoit-Wolf shrinkage --------
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda_proba = cross_val_predict(lda, X, y, cv=cv, method="predict_proba")[:, 1]
    lda_score = np.log(np.clip(lda_proba, 1e-9, 1 - 1e-9)
                       / np.clip(1 - lda_proba, 1e-9, 1 - 1e-9))
    lda_pred = (lda_proba >= 0.5).astype(int)
    lda_acc = (lda_pred == y).mean()
    lda_auc = roc_auc_score(y, lda_proba)

    # -------- L2 logistic regression, cv-tuned regularisation --------
    logit = LogisticRegressionCV(Cs=10, cv=5, max_iter=3000, scoring="roc_auc")
    log_proba = cross_val_predict(logit, X, y, cv=cv, method="predict_proba")[:, 1]
    log_pred  = (log_proba >= 0.5).astype(int)
    log_acc = (log_pred == y).mean()
    log_auc = roc_auc_score(y, log_proba)

    # -------- Report --------
    tp = int(((lda_pred == 1) & (y == 1)).sum())
    fn = int(((lda_pred == 0) & (y == 1)).sum())
    tn = int(((lda_pred == 0) & (y == 0)).sum())
    fp = int(((lda_pred == 1) & (y == 0)).sum())
    s_wt = lda_score[y == 0]; s_mu = lda_score[y == 1]
    d = (s_wt.mean() - s_mu.mean()) / np.sqrt(
        (s_wt.var(ddof=1) + s_mu.var(ddof=1)) / 2)

    print(f"\nLDA  (shrunken)   accuracy = {lda_acc:.3f}   AUC = {lda_auc:.3f}")
    print(f"Logistic (L2, CV) accuracy = {log_acc:.3f}   AUC = {log_auc:.3f}")
    print(f"\nAt p=0.5 (LDA):")
    print(f"  Sensitivity (mutant recall) = {tp/(tp+fn):.3f}  ({tp}/{tp+fn})")
    print(f"  Specificity (WT     recall) = {tn/(tn+fp):.3f}  ({tn}/{tn+fp})")
    print(f"  Discriminant-axis separation |d| = {abs(d):.2f}")

    # -------- Save per-seed scores --------
    df_out = df[["seed", "genotype", "length_mm", "tilt_deg"]].copy()
    df_out["lda_proba"]   = lda_proba
    df_out["lda_score"]   = lda_score
    df_out["logit_proba"] = log_proba
    df_out["cv_fold"]     = fold_id
    per_seed_csv = os.path.join(out_dir, "classifier_scores_per_seed.csv")
    df_out.to_csv(per_seed_csv, index=False)
    print(f"\nwrote {per_seed_csv}")

    # -------- Figure --------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of LDA scores, split by true label
    bins = np.linspace(lda_score.min() - 0.5, lda_score.max() + 0.5, 36)
    binw = bins[1] - bins[0]
    ax1.hist(s_wt, bins=bins, alpha=0.55, color="#2E75B6",
             label=f"WT (n={len(s_wt)})")
    ax1.hist(s_mu, bins=bins, alpha=0.55, color="#C4371C",
             label=f"lys3.c (n={len(s_mu)})")
    x_grid = np.linspace(bins[0], bins[-1], 300)
    for arr, col in [(s_wt, "#2E75B6"), (s_mu, "#C4371C")]:
        mu, sd = arr.mean(), arr.std(ddof=1)
        ax1.plot(x_grid, stats.norm.pdf(x_grid, mu, sd) * len(arr) * binw,
                 color=col, lw=2)
    ax1.axvline(0, color="k", lw=1, ls="--",
                label="decision boundary (p=0.5)")
    ax1.set_xlabel("Classifier score (LDA log-odds)")
    ax1.set_ylabel("count")
    ax1.set_title(
        f"Cross-validated LDA scores\n"
        f"CV accuracy = {lda_acc:.1%}   AUC = {lda_auc:.3f}   "
        f"|d| = {abs(d):.2f}")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Right: ROC curves
    fpr, tpr, _ = roc_curve(y, lda_proba)
    ax2.plot(fpr, tpr, lw=2, color="#3D5A3C",
             label=f"LDA (AUC={lda_auc:.3f})")
    fpr2, tpr2, _ = roc_curve(y, log_proba)
    ax2.plot(fpr2, tpr2, lw=1.5, color="#8C6B2D", ls="--",
             label=f"Logistic (AUC={log_auc:.3f})")
    ax2.plot([0, 1], [0, 1], "k:", lw=0.7)
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.set_title("ROC (cross-validated)")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    base = os.path.join(out_dir, "classifier_scores_preview")
    plt.savefig(base + ".png", dpi=160, bbox_inches="tight")
    plt.savefig(base + ".pdf", bbox_inches="tight")
    plt.savefig(base + ".svg", bbox_inches="tight")
    plt.savefig(base + ".eps", bbox_inches="tight")
    plt.close()
    for ext in ("png", "pdf", "svg", "eps"):
        print(f"wrote {base}.{ext}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in-csv",
                    default=os.path.join(here, "thickness_profiles_per_seed.csv"),
                    help="input CSV from 01_extract_thickness_profiles.py")
    ap.add_argument("--out-dir", default=here,
                    help="directory for output CSV and figure")
    args = ap.parse_args()
    main(args.in_csv, args.out_dir)
