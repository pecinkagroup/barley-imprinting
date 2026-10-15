"""
02_compare_groups.py
====================

Statistical comparison and figure: WT vs lys3.c (Riso 19) on the pointwise
thickness-along-length profile produced by 01_extract_thickness_profiles.py.

INPUT
-----
thickness_profiles_per_seed.csv
    Produced by 01_extract_thickness_profiles.py. Must contain columns:
        seed, genotype ("WT" or "lys3.c"),
        length_mm, tilt_deg,
        t_000 ... t_099   (thickness in mm at 100 normalized positions)

OUTPUT
------
thickness_profile_WTvsLys_source.csv
    One row per position with per-group summary statistics and test
    results. Columns:
        position_index, position_rel,
        WT_n,    WT_mean_mm,    WT_sd_mm,  WT_sem_mm, WT_ci95_mm,
        lys_n,   lys_mean_mm,   lys_sd_mm, lys_sem_mm, lys_ci95_mm,
        diff_mm    (WT − lys3.c),
        cohen_d    (pooled-SD),
        mw_p_raw   (Mann-Whitney U, two-sided),
        mw_p_BH    (Benjamini-Hochberg FDR across 100 positions),
        significant (bool, mw_p_BH < 0.05)

thickness_profile_WTvsLys.png / .pdf
    Two-panel figure. Top: mean ± 95% CI thickness profile per group, with
    q<0.05 positions shaded. Bottom: pointwise WT − lys3.c difference with
    95% CI band.

STATISTICAL DESIGN
------------------
All wild-type seeds are pooled into a single "WT" group (genotype is the
only grouping variable in the input CSV). For every one of the 100
normalized positions along the seed's major axis we run a two-sided
Mann-Whitney U test comparing WT vs lys3.c. Cohen's d (pooled SD) is
reported as an effect-size estimate. The 100 p-values are corrected for
multiple testing with Benjamini-Hochberg FDR; positions with q < 0.05 are
flagged "significant".

DEPENDENCIES
------------
    numpy pandas scipy matplotlib
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
matplotlib.rcParams["svg.fonttype"] = "none"   # keep text as text (selectable)
import matplotlib.pyplot as plt


N_PROF = 100


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------
def ci95(x):
    """95% t-based CI half-width for a sample vector."""
    n = len(x)
    if n < 2:
        return np.nan
    return x.std(ddof=1) / np.sqrt(n) * stats.t.ppf(0.975, n - 1)


def benjamini_hochberg(p):
    """Return BH-FDR q-values for a 1-D array of p-values."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q_sorted = p[order] * n / (np.arange(1, n + 1))
    # Enforce monotonicity from the end
    q_sorted = np.minimum.accumulate(q_sorted[::-1])[::-1]
    q_sorted = np.clip(q_sorted, 0, 1)
    q = np.empty(n)
    q[order] = q_sorted
    return q


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
def main(in_csv, out_dir):
    df = pd.read_csv(in_csv)
    cols = [f"t_{k:03d}" for k in range(N_PROF)]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise SystemExit(f"Input CSV is missing columns: {missing[:3]} ...")

    # Separate groups
    df_wt  = df[df["genotype"] == "WT"]
    df_lys = df[df["genotype"] == "lys3.c"]
    if len(df_wt) < 3 or len(df_lys) < 3:
        raise SystemExit(
            f"Need at least 3 seeds per group (have WT={len(df_wt)}, "
            f"lys3.c={len(df_lys)}).")

    WT  = df_wt[cols].values    # (n_WT,  100)
    LYS = df_lys[cols].values   # (n_LYS, 100)
    print(f"WT     n = {WT.shape[0]}")
    print(f"lys3.c n = {LYS.shape[0]}")

    pos_rel = (np.arange(N_PROF) + 0.5) / N_PROF

    rows = []
    for k in range(N_PROF):
        a, b = WT[:, k], LYS[:, k]
        m_a, s_a = a.mean(), a.std(ddof=1)
        m_b, s_b = b.mean(), b.std(ddof=1)
        sem_a, sem_b = s_a / np.sqrt(len(a)), s_b / np.sqrt(len(b))
        p_raw = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
        pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        d = (m_a - m_b) / pooled if pooled > 0 else np.nan
        rows.append(dict(
            position_index=k, position_rel=pos_rel[k],
            WT_n=len(a), WT_mean_mm=m_a, WT_sd_mm=s_a,
            WT_sem_mm=sem_a, WT_ci95_mm=ci95(a),
            lys_n=len(b), lys_mean_mm=m_b, lys_sd_mm=s_b,
            lys_sem_mm=sem_b, lys_ci95_mm=ci95(b),
            diff_mm=m_a - m_b, cohen_d=d, mw_p_raw=p_raw,
        ))
    tbl = pd.DataFrame(rows)
    tbl["mw_p_BH"] = benjamini_hochberg(tbl["mw_p_raw"].values)
    tbl["significant"] = tbl["mw_p_BH"] < 0.05

    src_csv = os.path.join(out_dir, "thickness_profile_WTvsLys_source.csv")
    tbl.to_csv(src_csv, index=False)
    n_sig = int(tbl["significant"].sum())
    print(f"{n_sig}/{N_PROF} positions significant at BH-FDR q<0.05")
    print(f"wrote {src_csv}")

    # ---------------------------------------------------------------------
    # Figure
    # ---------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 7), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 2], hspace=0.08))

    c_wt, c_lys = "#2E75B6", "#C4371C"

    ax1.plot(pos_rel, tbl["WT_mean_mm"],  color=c_wt,  lw=2,
             label=f"WT (n={WT.shape[0]})")
    ax1.fill_between(pos_rel,
                     tbl["WT_mean_mm"]  - tbl["WT_ci95_mm"],
                     tbl["WT_mean_mm"]  + tbl["WT_ci95_mm"],
                     color=c_wt, alpha=0.22, linewidth=0)
    ax1.plot(pos_rel, tbl["lys_mean_mm"], color=c_lys, lw=2,
             label=f"lys3.c (n={LYS.shape[0]})")
    ax1.fill_between(pos_rel,
                     tbl["lys_mean_mm"] - tbl["lys_ci95_mm"],
                     tbl["lys_mean_mm"] + tbl["lys_ci95_mm"],
                     color=c_lys, alpha=0.22, linewidth=0)

    sig_mask = tbl["significant"].values
    if sig_mask.any():
        y0, y1 = ax1.get_ylim()
        ax1.fill_between(pos_rel, y0, y1, where=sig_mask,
                         color="#FFD966", alpha=0.25, linewidth=0,
                         label=f"BH-FDR q<0.05 ({n_sig}/{N_PROF} positions)")
        ax1.set_ylim(y0, y1)

    ax1.set_ylabel("Silhouette thickness (mm)")
    ax1.set_title("Thickness-along-length profile: WT vs lys3.c")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower center", frameon=False)

    diff = tbl["diff_mm"].values
    # SE of the difference of means = sqrt(SEM_a^2 + SEM_b^2)
    se_diff = np.sqrt(tbl["WT_sem_mm"]**2 + tbl["lys_sem_mm"]**2)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.plot(pos_rel, diff, color="#3D5A3C", lw=2)
    ax2.fill_between(pos_rel, diff - 1.96*se_diff, diff + 1.96*se_diff,
                     color="#3D5A3C", alpha=0.2, linewidth=0,
                     label="95% CI of mean difference")

    if sig_mask.any():
        y0, y1 = ax2.get_ylim()
        ax2.fill_between(pos_rel, y0, y1, where=sig_mask,
                         color="#FFD966", alpha=0.25, linewidth=0)
        ax2.set_ylim(y0, y1)

    ax2.set_xlabel("Normalized position along length "
                   "(0 = thicker half, 1 = thinner half)")
    ax2.set_ylabel("WT − lys3.c (mm)")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", frameon=False)
    ax1.set_xlim(0, 1)

    base = os.path.join(out_dir, "thickness_profile_WTvsLys")
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
