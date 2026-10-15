"""
05_test_f2_ratio.py
===================

Test whether an observed F2 score distribution matches expected
Mendelian segregation ratios (default: 3:1 recessive).

Applies four complementary tests:
    (1) Prevalence-corrected mutant fraction using training-set
        sensitivity/specificity
    (2) MLE of mutant proportion from a 2-Gaussian mixture model
        (training WT + lys3.c distributions, frozen means and SDs)
        with bootstrap 95% CI
    (3) Likelihood-ratio test against candidate ratios (3:1, 1:1, 7:1,
        and the MLE itself as a sanity check)
    (4) 2-vs-3-component Gaussian mixture model (ΔBIC) to detect an
        intermediate phenotypic class (semi-dominance / 2:1:1)
    (5) Monte-Carlo KS test against a 3:1 reference distribution

INPUTS
------
--train-csv : CSV from 03_classifier_preview.py on the labelled training
              set. Must contain columns:
                  seed, genotype ("WT" or "lys3.c"),
                  lda_proba, lda_score
--f2-csv    : CSV from 04_score_unknowns.py on the F2 seeds. Must
              contain columns:
                  seed, lda_proba, lda_score

OUTPUTS
-------
f2_ratio_test_summary.csv   One row per test with results
f2_ratio_test.{png,pdf,svg} Five-panel figure:
    - F2 histogram with WT/lys3.c reference pdfs and expected 3:1 pdf
    - Monte-Carlo KS null distribution with observed KS statistic
    - Bootstrap distribution of MLE mutant fraction
    - 2-component vs 3-component GMM fit overlay
    - Summary bar chart: raw, corrected, MLE vs 3:1 / 1:1 expectations

DEPENDENCIES
------------
    numpy pandas scipy scikit-learn matplotlib
"""
import os, argparse
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict


DEFAULT_EXPECTED_P = 0.25     # 3:1 recessive → 25% mutant
N_SIM = 1000                   # Monte-Carlo replicates for KS null
N_BOOT = 1000                  # Bootstrap replicates for MLE CI
RNG_SEED = 28


# ======================================================================
# Core statistical functions
# ======================================================================
def training_sens_spec(train):
    """CV sensitivity (lys3.c correctly called) and specificity (WT
    correctly called) from the training classifier scores."""
    y = (train.genotype == "lys3.c").astype(int).values
    pred = (train.lda_proba >= 0.5).astype(int).values
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    return tp / (tp + fn), tn / (tn + fp)


def prevalence_correct(raw_rate, n, sens, spec):
    """Rogan-Gladen correction with Wald-SE 95% CI."""
    denom = sens - (1 - spec)
    if denom <= 0:
        return np.nan, (np.nan, np.nan)
    corrected = max(0.0, min(1.0, (raw_rate - (1 - spec)) / denom))
    se_raw = np.sqrt(raw_rate * (1 - raw_rate) / n)
    se_corr = se_raw / denom
    return corrected, (max(0.0, corrected - 1.96 * se_corr),
                       min(1.0, corrected + 1.96 * se_corr))


def mixture_loglik(p, scores, mu_WT, sd_WT, mu_mu, sd_mu):
    pdf_WT = stats.norm.pdf(scores, mu_WT, sd_WT)
    pdf_mu = stats.norm.pdf(scores, mu_mu, sd_mu)
    like = (1 - p) * pdf_WT + p * pdf_mu
    return np.log(np.clip(like, 1e-300, None)).sum()


def mle_mixture(scores, mu_WT, sd_WT, mu_mu, sd_mu):
    res = optimize.minimize_scalar(
        lambda p: -mixture_loglik(p, scores, mu_WT, sd_WT, mu_mu, sd_mu),
        bounds=(0.001, 0.999), method="bounded")
    return float(res.x)


def bootstrap_mle(scores, mu_WT, sd_WT, mu_mu, sd_mu, n_boot=N_BOOT,
                  rng=None):
    rng = rng or np.random.default_rng(0)
    return np.array([
        mle_mixture(rng.choice(scores, len(scores), replace=True),
                    mu_WT, sd_WT, mu_mu, sd_mu)
        for _ in range(n_boot)
    ])


def sample_from_mixture(n, p_mu, mu_WT, sd_WT, mu_mu, sd_mu, rng):
    is_mu = rng.random(n) < p_mu
    s = np.empty(n)
    s[~is_mu] = rng.normal(mu_WT, sd_WT, (~is_mu).sum())
    s[ is_mu] = rng.normal(mu_mu, sd_mu,   is_mu.sum())
    return s


def mc_ks_test(obs, p_mu, mu_WT, sd_WT, mu_mu, sd_mu, n_sim=N_SIM,
               rng=None):
    rng = rng or np.random.default_rng(RNG_SEED)
    n = len(obs)
    ref = sample_from_mixture(n * 50, p_mu, mu_WT, sd_WT, mu_mu, sd_mu, rng)
    ks_obs = stats.ks_2samp(obs, ref).statistic
    null_ks = np.array([
        stats.ks_2samp(
            sample_from_mixture(n, p_mu, mu_WT, sd_WT, mu_mu, sd_mu, rng),
            ref
        ).statistic
        for _ in range(n_sim)
    ])
    return ks_obs, (null_ks >= ks_obs).mean(), null_ks


def fit_gmm(scores, n_components, random_state=0):
    gm = GaussianMixture(n_components=n_components, covariance_type="full",
                         n_init=10, random_state=random_state)
    gm.fit(scores.reshape(-1, 1))
    order = np.argsort(gm.means_.ravel())
    return dict(means=gm.means_.ravel()[order],
                sds=np.sqrt(gm.covariances_.ravel()[order]),
                weights=gm.weights_[order],
                bic=gm.bic(scores.reshape(-1, 1)))


# ======================================================================
# Main
# ======================================================================
def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    return np.log(p / (1 - p))


def _train_cv_scores(train_df, n_prof=100):
    """Train shrunken LDA on the thickness-profile columns, return
    (lda_proba, lda_score) using 5-fold stratified cross-validation."""
    cols = [f"t_{k:03d}" for k in range(n_prof)]
    missing = [c for c in cols if c not in train_df.columns]
    if missing:
        raise SystemExit(f"train: CSV must contain either lda_proba/lda_score "
                          f"or the full profile columns {cols[0]}..{cols[-1]} "
                          f"(missing {len(missing)} columns)")
    X = train_df[cols].values
    y = (train_df["genotype"] == "lys3.c").astype(int).values
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    proba = cross_val_predict(lda, X, y, cv=cv, method="predict_proba")[:, 1]
    return proba, _logit(proba)


def _apply_lda_to_f2(train_df, f2_df, n_prof=100):
    """Fit LDA on ALL training rows, apply to F2 rows that carry profile
    columns, return (lda_proba, lda_score) for F2."""
    cols = [f"t_{k:03d}" for k in range(n_prof)]
    missing_f2 = [c for c in cols if c not in f2_df.columns]
    if missing_f2:
        raise SystemExit(
            f"\n  F2 CSV lacks both classifier scores (lda_proba / lda_score) "
            f"AND the thickness-profile columns ({cols[0]}..{cols[-1]}).\n"
            f"  → Run 04_score_unknowns.py on your F2 profiles first "
            f"to produce scores_unknowns.csv, then pass that as --f2-csv.")
    missing_tr = [c for c in cols if c not in train_df.columns]
    if missing_tr:
        raise SystemExit(
            f"\n  Training CSV has classifier scores but no profile columns "
            f"({cols[0]}..{cols[-1]}), so a new LDA cannot be fit here to "
            f"score the F2.\n"
            f"  → Run 04_score_unknowns.py on your F2 profiles using the "
            f"same training set that produced this training-score CSV, then "
            f"pass the resulting scores_unknowns.csv as --f2-csv.")
    y_tr = (train_df["genotype"] == "lys3.c").astype(int).values
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    lda.fit(train_df[cols].values, y_tr)
    proba = lda.predict_proba(f2_df[cols].values)[:, 1]
    return proba, _logit(proba)


def _ensure_lda_score(df, name, train_df=None):
    """Ensure df has both lda_proba and lda_score. If missing, derive
    from profile columns:
      * for the training CSV: use 5-fold CV scores
      * for the F2 CSV: fit LDA on `train_df` and apply to F2 profiles
    """
    df = df.copy()
    if "lda_score" in df.columns and "lda_proba" in df.columns:
        return df
    if "lda_score" in df.columns:
        s = df["lda_score"].astype(float).values
        df["lda_proba"] = 1 / (1 + np.exp(-s))
        return df
    if "lda_proba" in df.columns:
        df["lda_score"] = _logit(df["lda_proba"].values)
        return df
    # Neither — compute from profile columns
    if name == "train":
        proba, score = _train_cv_scores(df)
    else:
        if train_df is None:
            raise SystemExit("internal: train_df required to score F2 "
                              "from profile columns")
        proba, score = _apply_lda_to_f2(train_df, df)
    df["lda_proba"] = proba
    df["lda_score"] = score
    print(f"  [{name}] computed lda_proba/lda_score from profile columns")
    return df


def main(train_csv, f2_csv, out_dir, expected_p=DEFAULT_EXPECTED_P):
    train_raw = pd.read_csv(train_csv)
    f2_raw    = pd.read_csv(f2_csv)
    train_raw = train_raw[train_raw.genotype.isin(["WT", "lys3.c"])].copy()

    # If the training CSV lacks classifier scores, we train internally. In
    # that case we MUST also re-score the F2 with the same model, otherwise
    # training-reference and F2 scores come from different fits and the
    # two histograms live on different "score scales".
    train_had_scores = ("lda_score" in train_raw.columns or
                        "lda_proba" in train_raw.columns)
    train = _ensure_lda_score(train_raw, "train")
    if not train_had_scores:
        print("  → training had no classifier scores; re-scoring F2 with the "
              "newly trained LDA for scale consistency")
        proba, score = _apply_lda_to_f2(train_raw, f2_raw)
        f2 = f2_raw.copy()
        f2["lda_proba"] = proba
        f2["lda_score"] = score
    else:
        f2 = _ensure_lda_score(f2_raw, "f2", train_df=train_raw)
    print(f"Training: {len(train)} seeds "
          f"(WT={(train.genotype=='WT').sum()}, "
          f"lys3.c={(train.genotype=='lys3.c').sum()})")
    print(f"F2:       {len(f2)} seeds")

    sc_WT = train[train.genotype == "WT"].lda_score.values
    sc_mu = train[train.genotype == "lys3.c"].lda_score.values
    mu_WT, sd_WT = sc_WT.mean(), sc_WT.std(ddof=1)
    mu_mu, sd_mu = sc_mu.mean(), sc_mu.std(ddof=1)
    sens, spec = training_sens_spec(train)

    obs = f2.lda_score.values
    n = len(obs)
    raw_call = (f2.lda_proba >= 0.5).mean()
    corrected, (c_lo, c_hi) = prevalence_correct(raw_call, n, sens, spec)

    rng = np.random.default_rng(RNG_SEED)
    p_mle = mle_mixture(obs, mu_WT, sd_WT, mu_mu, sd_mu)
    boot = bootstrap_mle(obs, mu_WT, sd_WT, mu_mu, sd_mu,
                         n_boot=N_BOOT, rng=rng)
    mle_lo, mle_hi = np.percentile(boot, [2.5, 97.5])

    # LRT vs candidate ratios
    ll_free = mixture_loglik(p_mle, obs, mu_WT, sd_WT, mu_mu, sd_mu)
    candidates = [
        ("3:1 recessive",      0.25),
        ("1:1 half-mutant",    0.50),
        ("7:1 one-dose",       0.125),
    ]
    lrt_rows = []
    print(f"\n=== LRT vs candidate ratios ===")
    for name, p in candidates:
        ll_fix = mixture_loglik(p, obs, mu_WT, sd_WT, mu_mu, sd_mu)
        lrt = 2 * (ll_free - ll_fix)
        lrt_p = stats.chi2.sf(lrt, df=1)
        verdict = "consistent" if lrt_p > 0.05 else "REJECT"
        lrt_rows.append(dict(hypothesis=name, p_mu=p, LRT_stat=lrt,
                             LRT_p=lrt_p, verdict=verdict))
        print(f"  {name:>22s}  p={p:.3f}  LRT={lrt:6.2f}  "
              f"p={lrt_p:.4f}  → {verdict}")

    # MC KS test vs expected (default 3:1)
    ks_obs, ks_p, null_ks = mc_ks_test(obs, expected_p,
                                        mu_WT, sd_WT, mu_mu, sd_mu,
                                        rng=np.random.default_rng(RNG_SEED))
    ks_verdict = "consistent" if ks_p > 0.05 else "REJECT"
    print(f"\nMC KS vs {expected_p:.0%}: KS={ks_obs:.3f}  p={ks_p:.3f}  "
          f"→ {ks_verdict}")

    # 2-vs-3-component GMM
    g2 = fit_gmm(obs, 2); g3 = fit_gmm(obs, 3)
    delta_bic = g2["bic"] - g3["bic"]
    if delta_bic > 10:
        gmm_verdict = "3-component strongly preferred (semi-dominance)"
    elif delta_bic > 2:
        gmm_verdict = "3-component weakly preferred"
    elif delta_bic < -2:
        gmm_verdict = "2-component preferred (consistent with recessive)"
    else:
        gmm_verdict = "no clear preference"
    print(f"\nGMM: 2-comp BIC={g2['bic']:.1f}  3-comp BIC={g3['bic']:.1f}  "
          f"ΔBIC={delta_bic:+.1f}  → {gmm_verdict}")

    # ---------- Save summary CSV ----------
    rows = [
        dict(test="raw_calls",          value=raw_call,
             ci_lo=np.nan, ci_hi=np.nan, verdict=""),
        dict(test="prevalence_corrected", value=corrected,
             ci_lo=c_lo, ci_hi=c_hi, verdict=""),
        dict(test="MLE_mixture",        value=p_mle,
             ci_lo=mle_lo, ci_hi=mle_hi, verdict=""),
    ]
    for r in lrt_rows:
        rows.append(dict(test=f"LRT_{r['hypothesis']}", value=r['p_mu'],
                         ci_lo=np.nan, ci_hi=np.nan,
                         verdict=f"p={r['LRT_p']:.4f} {r['verdict']}"))
    rows.append(dict(test=f"MC_KS_vs_{expected_p:.0%}", value=ks_obs,
                     ci_lo=np.nan, ci_hi=np.nan,
                     verdict=f"p={ks_p:.3f} {ks_verdict}"))
    rows.append(dict(test="GMM_2vs3_deltaBIC", value=delta_bic,
                     ci_lo=np.nan, ci_hi=np.nan, verdict=gmm_verdict))
    pd.DataFrame(rows).to_csv(out_dir + "/f2_ratio_test_summary.csv",
                              index=False)

    # ---------- Figure ----------
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    # Same binning logic as 04_score_unknowns.py: union of F2 and training
    # reference ranges, so axes align with the training reference pdfs.
    bins = np.linspace(min(obs.min(), train.lda_score.min()) - 0.5,
                       max(obs.max(), train.lda_score.max()) + 0.5, 36)
    binw = bins[1] - bins[0]
    x_g = np.linspace(bins[0], bins[-1], 400)

    pdf_WT = stats.norm.pdf(x_g, mu_WT, sd_WT)
    pdf_mu = stats.norm.pdf(x_g, mu_mu, sd_mu)
    pdf_31 = 0.75 * pdf_WT + 0.25 * pdf_mu
    pdf_11 = 0.50 * pdf_WT + 0.50 * pdf_mu
    pdf_mle = (1 - p_mle) * pdf_WT + p_mle * pdf_mu

    # (1) Observed F2 vs reference pdfs
    ax = fig.add_subplot(gs[0, :2])
    ax.hist(obs, bins=bins, color="#7B7B7B", alpha=0.72, edgecolor="black",
            label=f"Observed F2 (n={n})")
    for arr, c, lbl in [(sc_WT, "#2E75B6", "WT training"),
                        (sc_mu, "#C4371C", "lys3.c training")]:
        m, s = arr.mean(), arr.std(ddof=1)
        ax.plot(x_g, stats.norm.pdf(x_g, m, s) * n * binw,
                color=c, lw=2, ls="--", label=f"{lbl}  μ={m:+.2f}")
    # KDE of the observed F2 — a model-free smooth curve following the
    # F2 histogram shape, directly comparable to the 3:1 / 1:1 expected pdfs
    kde = stats.gaussian_kde(obs, bw_method="scott")
    ax.plot(x_g, kde(x_g) * n * binw, color="#111111", lw=2.5,
            label=f"F2 KDE (n={n})")
    ax.plot(x_g, pdf_31 * n * binw, "-",  color="#4CAF50", lw=2.5,
            label="expected 3:1 (25%)")
    ax.plot(x_g, pdf_11 * n * binw, "-.", color="#FF9800", lw=1.8,
            label="expected 1:1 (50%)")
    ax.plot(x_g, pdf_mle * n * binw, ":", color="black", lw=2.5,
            label=f"MLE pdf ({100*p_mle:.1f}%)")
    ax.axvline(0, color="k", lw=0.6, ls=":")
    ax.set_xlabel("Classifier score (LDA log-odds)")
    ax.set_ylabel("count")
    ax.set_title(f"F2 observed scores vs expected under candidate ratios")
    ax.legend(fontsize=9, loc="upper center"); ax.grid(alpha=0.3)

    # (2) MC KS null + observed
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(null_ks, bins=30, color="#CCCCCC", edgecolor="black",
            alpha=0.75, label=f"null KS ({expected_p:.0%} vs {expected_p:.0%}, "
                              f"N={N_SIM})")
    ax.axvline(ks_obs, color="red", lw=2.5,
               label=f"observed KS = {ks_obs:.3f}\n"
                     f"MC p = {ks_p:.3f}")
    ax.set_xlabel("KS statistic"); ax.set_ylabel("count")
    ax.set_title(f"Monte-Carlo KS vs {expected_p:.0%}  →  {ks_verdict}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (3) Bootstrap MLE
    ax = fig.add_subplot(gs[1, 0])
    ax.hist(100 * boot, bins=25, color="#2196F3", alpha=0.7,
            edgecolor="black", label=f"bootstrap MLE (n={N_BOOT})")
    for ratio_name, ratio_p, ratio_col in [
            ("3:1 (25%)", 0.25, "green"),
            ("1:1 (50%)", 0.50, "orange"),
            ("7:1 (12.5%)", 0.125, "purple")]:
        ax.axvline(100 * ratio_p, color=ratio_col, lw=1.5, ls="--",
                   label=ratio_name)
    ax.axvline(100 * p_mle, color="black", lw=2.5,
               label=f"MLE = {100*p_mle:.1f}%\n"
                     f"95% CI {100*mle_lo:.1f}–{100*mle_hi:.1f}%")
    ax.set_xlabel("Estimated lys3.c fraction (%)")
    ax.set_title("MLE mutant fraction + bootstrap CI")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (4) GMM fits
    ax = fig.add_subplot(gs[1, 1])
    ax.hist(obs, bins=bins, color="#7B7B7B", alpha=0.65,
            edgecolor="black", label=f"F2 (n={n})")
    y2 = sum(w * stats.norm.pdf(x_g, m, s)
             for w, m, s in zip(g2["weights"], g2["means"], g2["sds"]))
    y3 = sum(w * stats.norm.pdf(x_g, m, s)
             for w, m, s in zip(g3["weights"], g3["means"], g3["sds"]))
    ax.plot(x_g, y2 * n * binw, color="#2E75B6", lw=2,
            label=f"2-comp GMM  BIC={g2['bic']:.1f}\n"
                  f"w={np.round(g2['weights'], 3).tolist()}")
    ax.plot(x_g, y3 * n * binw, color="#C4371C", lw=2, ls="--",
            label=f"3-comp GMM  BIC={g3['bic']:.1f}\n"
                  f"w={np.round(g3['weights'], 3).tolist()}")
    ax.set_xlabel("Classifier score"); ax.set_ylabel("count")
    ax.set_title(f"Mixture-model (semi-dominance test): {gmm_verdict}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (5) Summary bar chart
    ax = fig.add_subplot(gs[1, 2])
    groups = ["Raw calls", "Prevalence\ncorrected", "MLE\n(mixture)",
              "3:1 expect.", "1:1 expect."]
    vals = [100*raw_call, 100*corrected, 100*p_mle, 25.0, 50.0]
    errs = [100 * np.sqrt(raw_call*(1-raw_call)/n) * 1.96,
            100 * (c_hi-c_lo) / 2,
            100 * (mle_hi-mle_lo) / 2, 0, 0]
    colors_bar = ["#7B7B7B", "#4CAF50", "#2196F3", "#C4371C", "#FF9800"]
    bars = ax.bar(groups, vals, yerr=errs, color=colors_bar,
                   edgecolor="black", capsize=5, linewidth=1.2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1.5,
                f"{v:.1f}%", ha="center", fontsize=10, weight="bold")
    ax.axhline(100 * expected_p, color="red", lw=0.7, ls="--", alpha=0.6)
    ax.set_ylabel("Mutant fraction (%)")
    ax.set_ylim(0, max(vals) * 1.3)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("Mutant fraction estimates vs expectations")
    ax.tick_params(axis="x", labelsize=9)

    plt.suptitle(
        f"F2 segregation test (n={n})   —   "
        f"training sens={100*sens:.0f}%, spec={100*spec:.0f}%",
        fontsize=13, y=1.00)

    base = os.path.join(out_dir, "f2_ratio_test")
    for ext in ("png", "pdf", "svg"):
        plt.savefig(f"{base}.{ext}",
                    dpi=160 if ext == "png" else None, bbox_inches="tight")
    plt.close()
    print(f"\nsaved {base}.{{png,pdf,svg}}")
    print(f"saved {out_dir}/f2_ratio_test_summary.csv")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-csv", required=True,
                    help="training-set CSV with lda_proba and lda_score "
                         "(from 03_classifier_preview.py)")
    ap.add_argument("--f2-csv", default=os.path.join(here, "scores_unknowns.csv"),
                    help="F2 scores CSV with lda_proba and lda_score "
                         "(from 04_score_unknowns.py)")
    ap.add_argument("--out-dir", default=here,
                    help="directory for output figure and summary CSV")
    ap.add_argument("--expected-p", type=float, default=DEFAULT_EXPECTED_P,
                    help="expected mutant fraction under the null "
                         "(default 0.25 for 3:1 recessive)")
    args = ap.parse_args()
    main(args.train_csv, args.f2_csv, args.out_dir, args.expected_p)
