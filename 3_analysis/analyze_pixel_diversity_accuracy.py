"""
Experiment: does reduced pixel-value diversity ("flatness") in non-growing-
season Sentinel-2 imagery coincide with worse wealth-prediction accuracy?

Theory being tested: outside the growing season, the vegetation-phenology
signal the model leans on is largely absent, so chips look spectrally flatter
(less pixel-to-pixel variation) -- and the flatter the chip, the less signal
the model has to work with, so error should go up. That effect is expected to
be strongest in rural/agricultural tracts, where vegetation IS the texture;
urban tracts (built structures, roads, mixed materials) should keep a richer
signal year-round regardless of season.

That's three separable, testable sub-claims, run in order below:

  H1a  Pixel diversity is itself seasonal: lower in non-growing-season
       quarters (Q4/Q1) than growing-season quarters (Q2/Q3).
  H1b  Pixel diversity predicts per-chip accuracy directly: flatter chips
       have higher error, INCLUDING within a single quarter (i.e. this
       isn't only "winter chips are worse" restated -- a flat chip should
       cost you accuracy even relative to other chips shot the same
       quarter, e.g. a cloudy/hazy summer chip). Also tests the signed
       version: does a flat chip bias the prediction in a consistent
       direction (e.g. dormancy reading as "less developed"), not just
       make it noisier either way.
  H1c  The diversity-accuracy relationship (and the seasonal dip in
       diversity) is bigger in agricultural/rural tracts than urban ones.
       Only runs if GEOID_FROM_NAME is filled in and LC_CSV is provided.

Inputs
------
QUARTERS: list of dicts, one per quarter: {"year", "quarter", "preds_csv",
    "diversity_csv"}. preds_csv is the existing per-quarter model output
    (columns include "name", "pred", "label") like the ones loaded in
    validation.ipynb. diversity_csv is the output of
    compute_pixel_diversity.py run against that same quarter's raw chip
    directory ("<imagery_root>/q{q}_{year}_s2/chips"). The two are joined
    on chip FILENAME (not full path), since they may have been produced
    from different machines/mounts. Build this list by hand, or use
    discover_quarters() below to scan an imagery root that follows
    geoetl's tlag layout (see configs/tlags/*.yml in the geoetl repo).

GEOID_FROM_NAME: chip files are named "{GEOID}.tif" directly -- geoetl's
    pipeline.py writes each chip to
    os.path.join(chips_dir, f"{aoi_id}.{chip_ext}") where
    aoi_id = str(row[uid_column]) and every tlag config sets
    uid_column: GEOID. So Path(name).stem IS the GEOID; no regex needed.
    (Spot-check this against one real filename before trusting H1c --
    this is read from geoetl's current pipeline code, not verified against
    an actual chip on disk.)

LC_CSV: optional, only needed for H1c. Same format as
    analyze_seasonality_lag.py's lc.csv (GEOID index, named land-cover
    fraction columns).

Outputs (under OUT_DIR):
    panel.csv                      merged chip-quarter panel
    diversity_by_quarter.csv/.png  H1a
    diversity_error_scatter.png    H1b (magnitude: diversity vs |error|)
    diversity_bias_scatter.png     H1b (direction: diversity vs signed error)
    h1_summary.txt                 all printed stats, captured to a file
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf


def discover_quarters(imagery_root, diversity_dir, preds_glob="artifacts/*/epoch*preds.csv"):
    """
    Scan imagery_root for geoetl tlag-style quarter directories
    (q{1-4}_{year}_s2, per configs/tlags/*.yml in the geoetl repo) and
    build a QUARTERS list automatically.

    For each quarter directory found, this looks for exactly one file
    matching preds_glob under it (handles both "*_preds.csv" and
    "*_valset_preds.csv" as seen in validation.ipynb) and expects
    compute_pixel_diversity.py to have already been run on
    "<quarter_dir>/chips", writing its output to
    "<diversity_dir>/q{quarter}_{year}.csv".

    Quarters with zero or more than one preds-csv match are skipped with a
    warning rather than guessed at.
    """
    imagery_root = Path(imagery_root)
    diversity_dir = Path(diversity_dir)
    quarters = []
    for qdir in sorted(imagery_root.glob("q[1-4]_????_s2")):
        m = re.match(r"q([1-4])_(\d{4})_s2", qdir.name)
        if not m:
            continue
        quarter, year = int(m.group(1)), int(m.group(2))

        matches = sorted(qdir.glob(preds_glob))
        if len(matches) != 1:
            print(f"Skipping {qdir.name}: found {len(matches)} preds CSVs "
                  f"matching '{preds_glob}' (expected exactly 1)")
            continue

        div_csv = diversity_dir / f"q{quarter}_{year}.csv"
        if not div_csv.exists():
            print(f"Skipping {qdir.name}: expected diversity CSV not found "
                  f"at {div_csv} -- run compute_pixel_diversity.py on "
                  f"{qdir / 'chips'} first")
            continue

        quarters.append({
            "year": year, "quarter": quarter,
            "preds_csv": str(matches[0]), "diversity_csv": str(div_csv),
        })

    print(f"discover_quarters: found {len(quarters)} usable quarters under {imagery_root}")
    return quarters


# <<< The default Q2/Q3-vs-Q1/Q4 growing-season split assumes Ohio
# corn-belt phenology (growing ~April-September) and is very likely WRONG
# for Arizona: AZ agriculture is irrigated (Yuma/Phoenix-area cropland
# grows through the winter -- lettuce, citrus -- with a summer
# dormant/fallow period instead), and Sonoran desert vegetation greens up
# after winter rains, not in summer heat. Override with
# --growing-quarters/--non-growing-quarters for AZ (and any other state)
# once real NDVI/cropping-calendar data is available -- a wrong split
# would show up as a false negative (real seasonality washed out by
# mislabeled quarters), not a false positive.
GROWING_QUARTERS = {2, 3}
NON_GROWING_QUARTERS = {1, 4}

# Preferred diversity metric, in priority order: use the first one present
# in the merged panel. ndvi_std is the more direct test of "vegetation
# signal" if compute_pixel_diversity.py had a NIR band configured; falls
# back to raw RGB spread otherwise.
DIVERSITY_METRIC_PRIORITY = ["ndvi_std", "mean_band_std"]

# Chip filename stem IS the GEOID (see module docstring).
GEOID_FROM_NAME = lambda name: Path(name).stem


# -------------------------------------------------------------------------
# Step 1 -- load + join preds with diversity metrics, on chip filename
# -------------------------------------------------------------------------
def load_quarter(year, quarter, preds_csv, diversity_csv):
    preds = pd.read_csv(preds_csv)
    div = pd.read_csv(diversity_csv)

    preds["chip_file"] = preds["name"].apply(lambda p: Path(p).name)
    div["chip_file"] = div["name"].apply(lambda p: Path(p).name)
    div = div.drop(columns=["name"])

    merged = preds.merge(div, on="chip_file", how="inner", validate="one_to_one")
    n_dropped = len(preds) - len(merged)
    if n_dropped:
        print(f"  q{quarter}_{year}: {n_dropped}/{len(preds)} chips had no "
              f"diversity match (filename join) -- check compute_pixel_diversity.py "
              f"was run on the matching chip directory.")

    merged["year"] = year
    merged["quarter"] = quarter
    merged["error"] = merged["pred"] - merged["label"]
    merged["abs_error"] = merged["error"].abs()
    merged["sq_error"] = merged["error"] ** 2
    return merged


def build_panel(quarters_cfg):
    parts = []
    for cfg in quarters_cfg:
        print(f"Loading q{cfg['quarter']}_{cfg['year']}...")
        parts.append(load_quarter(cfg["year"], cfg["quarter"],
                                   cfg["preds_csv"], cfg["diversity_csv"]))
    panel = pd.concat(parts, ignore_index=True)

    panel["season"] = np.where(panel["quarter"].isin(GROWING_QUARTERS),
                                "growing", "non_growing")
    panel["t"] = (panel["year"] - panel["year"].min()) * 4 + (panel["quarter"] - 1)
    panel["t"] = panel["t"] - panel["t"].min()

    if GEOID_FROM_NAME is not None:
        panel["GEOID"] = panel["name"].apply(GEOID_FROM_NAME)

    return panel


def pick_metric(panel):
    for m in DIVERSITY_METRIC_PRIORITY:
        if m in panel.columns and panel[m].notna().any():
            return m
    raise ValueError(f"None of {DIVERSITY_METRIC_PRIORITY} found in panel columns")


# -------------------------------------------------------------------------
# H1a -- is pixel diversity itself seasonal?
# -------------------------------------------------------------------------
def h1a_seasonal_diversity(panel, metric, out_dir):
    print("\n" + "=" * 70)
    print(f"H1a -- seasonality of pixel diversity ({metric})")
    print("=" * 70)

    by_q = (panel.groupby(["year", "quarter"])[metric]
                 .agg(["mean", "std", "count"])
                 .reset_index()
                 .sort_values(["year", "quarter"]))
    by_q["t"] = (by_q["year"] - by_q["year"].min()) * 4 + (by_q["quarter"] - 1)
    by_q["t"] = by_q["t"] - by_q["t"].min()
    by_q.to_csv(out_dir / "diversity_by_quarter.csv", index=False)
    print(by_q.to_string(index=False))

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(by_q["t"], by_q["mean"], "-o", color="tab:blue")
    ax.fill_between(by_q["t"], by_q["mean"] - by_q["std"], by_q["mean"] + by_q["std"],
                     alpha=0.15, color="tab:blue")
    for _, r in by_q.iterrows():
        color = "tab:green" if r["quarter"] in GROWING_QUARTERS else "tab:orange"
        ax.axvspan(r["t"] - 0.5, r["t"] + 0.5, color=color, alpha=0.06)
    ax.set_xticks(by_q["t"])
    ax.set_xticklabels([f"q{q}_{y}" for y, q in zip(by_q["year"], by_q["quarter"])],
                        rotation=45, ha="right")
    ax.set_ylabel(metric)
    ax.set_title(f"Per-quarter pixel diversity ({metric})\n"
                 "green=growing season, orange=non-growing")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "diversity_by_quarter.png", dpi=150)
    plt.close()

    growing = panel.loc[panel["season"] == "growing", metric].dropna()
    non_growing = panel.loc[panel["season"] == "non_growing", metric].dropna()
    t, p = stats.ttest_ind(growing, non_growing, equal_var=False, alternative="greater")
    u, p_u = stats.mannwhitneyu(growing, non_growing, alternative="greater")
    print(f"\nGrowing-season {metric}: mean={growing.mean():.4f}, n={len(growing)}")
    print(f"Non-growing {metric}:    mean={non_growing.mean():.4f}, n={len(non_growing)}")
    print(f"Welch t-test (growing > non-growing):  t={t:.3f}, p={p:.4f}")
    print(f"Mann-Whitney U (growing > non-growing): U={u:.1f}, p={p_u:.4f}")

    return by_q


# -------------------------------------------------------------------------
# H1b -- does pixel diversity predict accuracy, including within-quarter?
# -------------------------------------------------------------------------
def h1b_diversity_predicts_error(panel, metric, out_dir):
    print("\n" + "=" * 70)
    print(f"H1b -- does {metric} predict accuracy?")
    print("=" * 70)

    d = panel.dropna(subset=[metric, "sq_error", "abs_error"])

    r_pearson, p_pearson = stats.pearsonr(d[metric], d["abs_error"])
    r_spearman, p_spearman = stats.spearmanr(d[metric], d["abs_error"])
    print(f"Overall correlation, {metric} vs abs_error:")
    print(f"  Pearson  r={r_pearson:+.4f}, p={p_pearson:.4g}")
    print(f"  Spearman r={r_spearman:+.4f}, p={p_spearman:.4g}")
    print("(negative r = flatter chips -> bigger errors, as the theory predicts)")

    # Within-quarter correlation: demean both variables by quarter first, so
    # this isolates whether diversity matters even holding season fixed --
    # distinct from H1a's "winter chips are just worse".
    d = d.copy()
    d["metric_c"] = d[metric] - d.groupby(["year", "quarter"])[metric].transform("mean")
    d["abs_error_c"] = d["abs_error"] - d.groupby(["year", "quarter"])["abs_error"].transform("mean")
    r_within, p_within = stats.pearsonr(d["metric_c"], d["abs_error_c"])
    print(f"\nWithin-quarter (season-demeaned) correlation:")
    print(f"  Pearson r={r_within:+.4f}, p={p_within:.4g}")

    print(f"\nOLS: sq_error ~ {metric} + C(quarter), cluster-robust by quarter")
    m = smf.ols(f"sq_error ~ {metric} + C(quarter)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["quarter"].astype(str) + "_" + d["year"].astype(str)}
    )
    print(m.summary())

    # Everything above is about error MAGNITUDE (do flat chips get noisier
    # predictions). This is the same question for error DIRECTION: does a
    # flat chip make the model over-predict or under-predict specifically,
    # not just predict less accurately -- e.g. if winter dormancy reads as
    # "less developed" to the model, flatter chips might systematically
    # under-predict wealth rather than just being noisier either way.
    r_signed, p_signed = stats.pearsonr(d[metric], d["error"])
    print(f"\nSigned-error correlation, {metric} vs error (pred - truth):")
    print(f"  Pearson r={r_signed:+.4f}, p={p_signed:.4g}")
    print("(nonzero r = flatter chips bias predictions in a consistent "
          "direction, not just noisier ones)")

    print(f"\nOLS: error ~ {metric} + C(quarter), cluster-robust by quarter")
    m_signed = smf.ols(f"error ~ {metric} + C(quarter)", data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["quarter"].astype(str) + "_" + d["year"].astype(str)}
    )
    print(m_signed.summary())

    fig, ax = plt.subplots(figsize=(8, 6))
    sample = d.sample(min(5000, len(d)), random_state=0)
    ax.scatter(sample[metric], sample["abs_error"], s=6, alpha=0.25, color="tab:blue")
    ax.set_xlabel(metric)
    ax.set_ylabel("|prediction error|")
    ax.set_title(f"Pixel diversity vs. accuracy\nPearson r={r_pearson:+.3f} (p={p_pearson:.3g})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "diversity_error_scatter.png", dpi=150)
    plt.close()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axhline(0, color="black", lw=1, alpha=0.6)
    ax.scatter(sample[metric], sample["error"], s=6, alpha=0.25, color="tab:purple")
    ax.set_xlabel(metric)
    ax.set_ylabel("Signed error (pred − truth)")
    ax.set_title(f"Pixel diversity vs. prediction bias\nPearson r={r_signed:+.3f} (p={p_signed:.3g})")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "diversity_bias_scatter.png", dpi=150)
    plt.close()

    return m, m_signed


# -------------------------------------------------------------------------
# H1c -- is the effect stronger in rural/agricultural tracts?
# -------------------------------------------------------------------------
def h1c_landscape_moderation(panel, metric, lc_csv, out_dir):
    print("\n" + "=" * 70)
    print("H1c -- landscape moderation (agricultural vs urban)")
    print("=" * 70)

    if "GEOID" not in panel.columns:
        print("Skipped: GEOID_FROM_NAME not configured, can't join land cover.")
        return None

    lc = pd.read_csv(lc_csv, dtype={0: str}, index_col=0)
    lc.index.name = "GEOID"
    lc = lc.reset_index()
    lc_class_cols = [c for c in lc.columns if c != "GEOID"]
    row_sums = lc[lc_class_cols].sum(axis=1)
    if (row_sums > 2).any():
        lc[lc_class_cols] = lc[lc_class_cols].div(row_sums, axis=0)

    d = panel.merge(lc, on="GEOID", how="inner")
    d = d.dropna(subset=[metric, "sq_error", "Cropland"]).copy()
    d["cropland_c"] = d["Cropland"] - d["Cropland"].mean()
    print(f"Joined {len(d):,} / {len(panel):,} chip-quarter rows to land cover.")

    print(f"\nMixed model: sq_error ~ {metric} * cropland_c + (1|GEOID)")
    try:
        m1 = smf.mixedlm(f"sq_error ~ {metric} * cropland_c", data=d, groups=d["GEOID"]).fit(method="lbfgs")
    except np.linalg.LinAlgError:
        m1 = smf.ols(f"sq_error ~ {metric} * cropland_c", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["GEOID"]})
    print(m1.summary())
    print(f"\nInteraction term ({metric}:cropland_c) tests whether the "
          f"diversity->error relationship is steeper in more agricultural tracts.")

    print(f"\nMixed model: {metric} ~ C(season) * cropland_c + (1|GEOID)")
    try:
        m2 = smf.mixedlm(f"{metric} ~ C(season) * cropland_c", data=d, groups=d["GEOID"]).fit(method="lbfgs")
    except np.linalg.LinAlgError:
        m2 = smf.ols(f"{metric} ~ C(season) * cropland_c", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["GEOID"]})
    print(m2.summary())
    print(f"\nInteraction term (season:cropland_c) tests whether the seasonal "
          f"dip in diversity is bigger in more agricultural tracts.")

    return m1, m2


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    global GROWING_QUARTERS, NON_GROWING_QUARTERS

    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State label, e.g. oh/az/ca/ga/pa (used for output naming only)")
    p.add_argument("--imagery-root", required=True, type=Path,
                    help="geoetl tlag imagery root for this state, e.g. .../tlag/az_imagery")
    p.add_argument("--diversity-dir", type=Path, default=Path("./diversity"),
                    help="Where compute_pixel_diversity.py wrote q{quarter}_{year}.csv per quarter")
    p.add_argument("--lc-csv", type=Path, default=None,
                    help="lc.csv (e.g. from compute_landcover.py) -- set to run H1c; omitted skips H1c")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: ./out_pixel_diversity/<state>)")
    p.add_argument("--growing-quarters", default="2,3",
                    help="Comma-separated growing-season quarters (default: 2,3 -- Ohio corn-belt; "
                         "override per state once real phenology data is available)")
    args = p.parse_args()

    GROWING_QUARTERS = {int(q) for q in args.growing_quarters.split(",")}
    NON_GROWING_QUARTERS = {1, 2, 3, 4} - GROWING_QUARTERS

    out_dir = args.out_dir or Path(f"./out_pixel_diversity/{args.state.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    quarters = discover_quarters(args.imagery_root, args.diversity_dir)
    if not quarters:
        raise SystemExit(
            f"No usable quarters found under {args.imagery_root}. Run "
            f"compute_pixel_diversity.py against each quarter's chip dir first."
        )

    panel = build_panel(quarters)
    panel.to_csv(out_dir / "panel.csv", index=False)
    print(f"\n[{args.state}] Panel: {len(panel):,} chip-quarter rows across "
          f"{panel[['year', 'quarter']].drop_duplicates().shape[0]} quarters")

    metric = pick_metric(panel)
    print(f"Using diversity metric: {metric}")

    h1a_seasonal_diversity(panel, metric, out_dir)
    h1b_diversity_predicts_error(panel, metric, out_dir)
    if args.lc_csv:
        h1c_landscape_moderation(panel, metric, args.lc_csv, out_dir)
    else:
        print("\nH1c skipped: --lc-csv not given.")


if __name__ == "__main__":
    main()
