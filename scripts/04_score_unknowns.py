"""
04_score_unknowns.py
====================

Score seeds of unknown genotype (e.g. F2 seeds from a selfed heterozygote)
using a classifier trained on the labeled WT vs lys3.c dataset.

Use case: you have a bag of F2 seeds — you don't know each seed's
genotype, but you want to estimate the phenotypic ratio in the population
and, optionally, whether the score distribution supports a strictly
recessive (3:1) or semi-dominant (2:1:1) segregation.

WORKFLOW
--------
    01_extract_thickness_profiles.py     [on labeled WT + lys3.c seeds]
        ↓  thickness_profiles_per_seed.csv                                 ← --train-csv
    01_extract_thickness_profiles.py     [on unknown F2 seeds]
        ↓  thickness_profiles_per_seed.csv (all genotype = "Unknown")      ← --in-csv
    04_score_unknowns.py
        ↓  scores_unknowns.csv, scores_unknowns_histogram.{png,pdf,svg,eps}

INPUT
-----
--train-csv : CSV produced by 01_extract_thickness_profiles.py on the
              labeled training set. Must contain rows with genotype "WT"
              and genotype "lys3.c", plus the 100 profile columns
              t_000 ... t_099.
--in-csv    : CSV produced by 01_extract_thickness_profiles.py on the
              unknown seeds. Genotype column is expected to be "Unknown"
              (any other label is treated as unknown for scoring
              purposes). Must contain the same 100 profile columns.

OUTPUT
------
scores_unknowns.csv
    One row per unknown seed. Columns:
        seed, length_mm, tilt_deg,
        lda_proba, lda_score, logit_proba,
        lda_call        "WT" if lda_proba < 0.5 else "lys3.c"

scores_unknowns_histogram.{png,pdf,svg,eps}
    Two-panel figure.
    Left:  histogram of unknowns' LDA log-odds scores, with reference
           Gaussian curves for the training WT and lys3.c distributions
           overlaid (dashed) so the eye can see where each unknown seed
           lands. Vertical dashed line = classifier decision boundary
           (p = 0.5). The percentage of unknowns above the boundary is
           printed on the panel.
    Right: two- vs three-component Gaussian mixture model fit to the
           unknowns' scores. The two-component model corresponds to a
           strictly recessive segregation (only WT-like and mutant-like
           modes; expected 3:1). A three-component model adds an
           intermediate mode, corresponding to semi-dominance (expected
           2:1:1 with aaA intermediate). The BIC of both models is
           printed, and the estimated mixture proportions are shown.

METHODS
-------
Classifiers are LDA with Ledoit-Wolf shrinkage and L2 logistic regression
(inverse-strength C chosen by internal 5-fold ROC-AUC), both trained on
all labeled seeds in --train-csv. Because we don't have ground truth for
the unknowns, there is no cross-validation on them: they are simply
scored by the fitted classifiers. The mixture-model fit uses
scikit-learn's GaussianMixture on the LDA log-odds scores.

Note on prevalence correction: if you want to invert the raw call rate
into an estimated true mutant fraction, use the training-set sensitivity
and specificity (available from 03_classifier_preview.py) via
    true_rate = (observed_rate - (1 - specificity)) /
                (sensitivity - (1 - specificity))
This script does not apply the correction automatically, because it
needs the sensitivity/specificity that come from cross-validation on the
labeled data, and those depend on how the training set was formed.

DEPENDENCIES
------------
    numpy pandas scipy scikit-learn matplotlib
"""

import os, argparse
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegressionCV
from sklearn.mixture import GaussianMixture


N_PROF = 100


def load_profiles(csv_path):
    df = pd.read_csv(csv_path)
    cols = [f"t_{k:03d}" for k in range(N_PROF)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"{csv_path} missing columns: {missing[:3]} ...")
    return df, cols


def logit(p):
    return np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1 - p, 1e-9, 1 - 1e-9))


def fit_gmm(scores, n_components, random_state=0):
    gm = GaussianMixture(n_components=n_components, covariance_type="full",
                         n_init=10, random_state=random_state)
    gm.fit(scores.reshape(-1, 1))
    order = np.argsort(gm.means_.ravel())
    means   = gm.means_.ravel()[order]
    sds     = np.sqrt(gm.covariances_.ravel()[order])
    weights = gm.weights_[order]
    return dict(means=means, sds=sds, weights=weights,
                bic=gm.bic(scores.reshape(-1, 1)))


def main(train_csv, in_csv, out_dir):
    # ---------- Load ----------
    df_train, cols = load_profiles(train_csv)
    df_train = df_train[df_train["genotype"].isin(["WT", "lys3.c"])].copy()
    if (df_train["genotype"] == "WT").sum() < 5 or \
       (df_train["genotype"] == "lys3.c").sum() < 5:
        raise SystemExit("Training set needs both WT and lys3.c rows.")

    df_test, _ = load_profiles(in_csv)
    if df_test.empty:
        raise SystemExit(f"No unknown seeds found in {in_csv}")

    print(f"Training on {len(df_train)} seeds "
          f"(WT n={(df_train.genotype=='WT').sum()}, "
          f"lys3.c n={(df_train.genotype=='lys3.c').sum()})")
    print(f"Scoring {len(df_test)} unknown seeds from {os.path.basename(in_csv)}")

    X_tr = df_train[cols].values
    y_tr = (df_train["genotype"] == "lys3.c").astype(int).values   # 1=mutant
    X_te = df_test[cols].values

    # ---------- Train on ALL labeled data ----------
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(X_tr, y_tr)
    logit_clf = LogisticRegressionCV(Cs=10, cv=5, max_iter=3000,
                                     scoring="roc_auc").fit(X_tr, y_tr)

    # ---------- Score unknowns ----------
    lda_proba   = lda.predict_proba(X_te)[:, 1]
    lda_score   = logit(lda_proba)
    logit_proba = logit_clf.predict_proba(X_te)[:, 1]

    df_out = df_test[["seed"]].copy()
    for c in ("length_mm", "tilt_deg"):
        if c in df_test.columns:
            df_out[c] = df_test[c].values
    df_out["lda_proba"]   = lda_proba
    df_out["lda_score"]   = lda_score
    df_out["logit_proba"] = logit_proba
    df_out["lda_call"]    = np.where(lda_proba >= 0.5, "lys3.c", "WT")
    scores_csv = os.path.join(out_dir, "scores_unknowns.csv")
    df_out.to_csv(scores_csv, index=False)

    n_mut = int((df_out["lda_call"] == "lys3.c").sum())
    n_tot = len(df_out)
    frac  = n_mut / n_tot
    print(f"\n  raw calls: {n_mut}/{n_tot} lys3.c ({100*frac:.1f}%)")
    print(f"  under 3:1  (recessive)     : expect 25% mutant")
    print(f"  under 2:1:1 (semi-dominant): expect 25% mutant + 25% intermediate")

    # ---------- Reference training-set score distributions ----------
    # (score each labeled training seed with the SAME fitted classifier — no
    # CV needed here since we just need reference Gaussians for the plot.)
    tr_score = logit(lda.predict_proba(X_tr)[:, 1])
    s_wt_ref = tr_score[y_tr == 0]
    s_mu_ref = tr_score[y_tr == 1]

    # ---------- Gaussian mixture: 2 vs 3 components ----------
    scores = lda_score
    gmm2 = fit_gmm(scores, 2)
    gmm3 = fit_gmm(scores, 3)
    delta_bic = gmm2["bic"] - gmm3["bic"]      # positive → 3-comp preferred
    print(f"\n  GMM BIC (2-component)   = {gmm2['bic']:.1f}")
    print(f"  GMM BIC (3-component)   = {gmm3['bic']:.1f}   ΔBIC = {delta_bic:+.1f}")
    print(f"  2-comp weights: {np.round(gmm2['weights'], 3)}")
    print(f"  3-comp weights: {np.round(gmm3['weights'], 3)}")
    if delta_bic > 10:
        gmm_verdict = "3-component preferred (evidence for intermediate class)"
    elif delta_bic > 2:
        gmm_verdict = "3-component slightly preferred (weak evidence)"
    elif delta_bic < -2:
        gmm_verdict = "2-component preferred (no intermediate class)"
    else:
        gmm_verdict = "no clear preference between 2 and 3 components"
    print(f"  verdict: {gmm_verdict}")

    # ---------- Figure ----------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bins = np.linspace(min(scores.min(), tr_score.min()) - 0.5,
                       max(scores.max(), tr_score.max()) + 0.5, 36)
    binw = bins[1] - bins[0]
    x_grid = np.linspace(bins[0], bins[-1], 400)

    # -- Left: unknowns' histogram with training-set reference Gaussians --
    ax1.hist(scores, bins=bins, alpha=0.65, color="#7B7B7B",
             label=f"Unknown seeds (n={n_tot})", edgecolor="black", linewidth=0.5)
    for arr, col, lbl in [(s_wt_ref, "#2E75B6", "WT (training)"),
                           (s_mu_ref, "#C4371C", "lys3.c (training)")]:
        mu, sd = arr.mean(), arr.std(ddof=1)
        ax1.plot(x_grid, stats.norm.pdf(x_grid, mu, sd) * n_tot * binw,
                 color=col, lw=2, ls="--",
                 label=f"{lbl}, μ={mu:.2f}, σ={sd:.2f}")
    ax1.axvline(0, color="k", lw=1, ls=":",
                label=f"p=0.5 boundary  ({100*frac:.1f}% called lys3.c)")
    ax1.set_xlabel("Classifier score (LDA log-odds)")
    ax1.set_ylabel("count")
    ax1.set_title("Unknown-seed scores with training-set reference distributions")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

    # -- Right: GMM fits --
    ax2.hist(scores, bins=bins, alpha=0.65, color="#7B7B7B",
             edgecolor="black", linewidth=0.5, label=f"Unknown seeds (n={n_tot})")
    # 2-component in blue
    y2 = np.zeros_like(x_grid)
    for w, m, s in zip(gmm2["weights"], gmm2["means"], gmm2["sds"]):
        y2 += w * stats.norm.pdf(x_grid, m, s)
    ax2.plot(x_grid, y2 * n_tot * binw, color="#2E75B6", lw=2,
             label=f"2-component GMM   BIC={gmm2['bic']:.1f}\n"
                   f"weights: {np.round(gmm2['weights'], 3).tolist()}")
    # 3-component in red
    y3 = np.zeros_like(x_grid)
    for w, m, s in zip(gmm3["weights"], gmm3["means"], gmm3["sds"]):
        y3 += w * stats.norm.pdf(x_grid, m, s)
    ax2.plot(x_grid, y3 * n_tot * binw, color="#C4371C", lw=2, ls="--",
             label=f"3-component GMM   BIC={gmm3['bic']:.1f}\n"
                   f"weights: {np.round(gmm3['weights'], 3).tolist()}")
    ax2.set_xlabel("Classifier score (LDA log-odds)")
    ax2.set_ylabel("count")
    ax2.set_title(f"Gaussian mixture fit — {gmm_verdict}")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.tight_layout()
    base = os.path.join(out_dir, "scores_unknowns_histogram")
    for ext in ("png", "pdf", "svg", "eps"):
        plt.savefig(f"{base}.{ext}",
                    dpi=160 if ext == "png" else None,
                    bbox_inches="tight")
    plt.close()

    print(f"\nwrote {scores_csv}")
    for ext in ("png", "pdf", "svg", "eps"):
        print(f"wrote {base}.{ext}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv",
                    default=os.path.join(here, "thickness_profiles_per_seed.csv"),
                    help="labeled training CSV (WT + lys3.c rows)")
    ap.add_argument("--in-csv", required=True,
                    help="unknown-seed CSV (from 01_extract_thickness_profiles.py)")
    ap.add_argument("--out-dir", default=here,
                    help="directory for output CSV and figure")
    args = ap.parse_args()
    main(args.train_csv, args.in_csv, args.out_dir)
