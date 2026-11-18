"""
Does the SIZE of the physical remote-sensing gap between a model's own
quarter and the quarter it's evaluated on explain cross-quarter transfer
accuracy better than just "is it the same season" / "is it the same
year"?

Three inputs, joined on (year, quarter) -- pass at least one of the
first two (both if you have both):
  --band-importance-csv   band_importance.py's long output. Off-diagonal
                           rows only (eval_year != model_year OR
                           eval_quarter != model_quarter) -- those are
                           the cross-quarter TRANSFER cells; the
                           diagonal is a model on its own quarter, not a
                           transfer at all. r2_base is the same value on
                           every row of a given cell (it's the baseline,
                           computed once before any permutation) --
                           deduplicated down to one row per cell. This is
                           a real but EXPENSIVE way to get cell R^2: it
                           comes along for free, but only after also
                           running 5 permutation repeats x 5 band-groups
                           per cell that this script doesn't need at all.
  --cross-quarter-r2-csv   build_cross_quarter_r2_matrix.py's r2_matrix.csv
                           -- the cheaper, purpose-built alternative: no
                           permutation work, just the baseline forward
                           pass. A wide matrix ("Q<quarter> <year>" row/
                           column labels), melted into the same per-cell
                           shape as --band-importance-csv. Prefer this
                           one when its coverage is further along.
                           Passing both sources at once takes the union
                           of their cells (deduplicated, preferring
                           --band-importance-csv's value on overlap --
                           they should agree anyway).
  --rs-summary-csv        compute_quarterly_rs_summary.py's wide per-
                           tract output. Averaged across every tract
                           into ONE state-level number per (year,
                           quarter, metric) -- the cell-level R^2 this
                           compares against is itself a single number
                           per cell, not per tract, so the RS side needs
                           to be a single number per quarter too.

Works with whatever (year, quarter) cells are ALREADY in each CSV --
both inputs can be partially built (the original request's own
situation: --rs-summary-csv was still filling in when this was asked
for). A model-eval cell is used only if BOTH its training quarter and
its eval quarter have RS summary coverage; everything else is skipped
with a per-metric NaN rather than dropped outright, so one metric's
missing quarter doesn't block every other metric's regression.

Step 1 -- rs_gap table (one row per off-diagonal cell, one column pair
per metric): for every metric M in --rs-summary-csv (auto-discovered
from its q<quarter>_<year>_<metric> columns, not hardcoded, so this
tracks compute_quarterly_rs_summary.py's own metric set):
    diff_M     = state_mean_M(eval_year, eval_quarter)
                 - state_mean_M(model_year, model_quarter)
    abs_diff_M = abs(diff_M)
Plus same_season (eval_quarter == model_quarter) and same_year
(eval_year == model_year) dummies -- never both 1 at once among off-
diagonal cells by construction (that combination IS the diagonal).

Step 2 -- for each metric M, THREE OLS fits of cell r2_base, ALL restricted
to the same intersection of cells (only those where M's diff/abs_diff are
non-NaN, i.e. where --rs-summary-csv covers both the model's and the
eval's quarter) -- the dummy models need no RS data at all, so without
this restriction they'd silently fit on every off-diagonal cell while
rs_gap only gets the (often much smaller) RS-covered subset, making
R^2/AIC incomparable between them (R^2 on a near-saturated n=6 fit is not
meaningfully "better" than R^2 on an n=240 fit):
    Model A ("rs_gap"):               r2 ~ diff_M + abs_diff_M
    Model B ("season_year_dummies"):  r2 ~ same_season + same_year
    Model C ("season_pair_dummies"):  r2 ~ one dummy per distinct
        (model_quarter -> eval_quarter) pair actually present (e.g.
        "Q1->Q3"), reference category dropped
Model B only knows whether the calendar label matches -- it can't tell
Q1-model-evaluated-on-Q2 apart from Q2-model-evaluated-on-Q1, even
though those could be genuinely different transfers (spring->summer
green-up isn't the mirror of summer->spring senescence). Model C fixes
that: it's a full categorical interaction that can memorize ANY
systematic quirk of a *specific* quarter pair, with no assumption the
relationship is continuous in physical distance at all -- a much harder
baseline for rs_gap to beat than B, even though adjusted R^2/AIC already
account for its extra parameters. All three compared on R^2/adjusted
R^2/AIC/BIC: does a continuous measure of how different the actual
conditions were explain more of the transfer accuracy than simply
knowing the calendar identity of the transfer (same/different, OR the
exact quarter pair)? Plain OLS, not cluster-robust: the "still building"
dataset this was built for has few off-diagonal cells, and clustered SEs
need more groups than that to be trustworthy -- read the printed n_obs
before trusting any p-value here, and treat this as early/exploratory
per the original request, not a final test. See Step 4 for an
alternative to trusting these p-values/AIC at face value.

Step 3 -- joint regression (does M survive once other metrics are
controlled for?): a single OLS fit of r2 on diff_M/abs_diff_M for EVERY
metric in --joint-metrics at once (default: every metric --rs-summary-csv
has). This exists because the per-metric fits in Step 2 can't tell "M's
gap really drives transfer accuracy" apart from "M's gap just correlates
with some OTHER metric that's the real driver" -- e.g. cloud fraction
and NDVI both move with season, so NDVI fitting well alone doesn't rule
out "it's actually a composite-quality (cloudy/hazy chip) effect, and
NDVI is just riding along." Fitting every metric's diff/abs_diff
together and checking whether a given metric's coefficient/p-value
survives is the direct test of that. Restricted to the intersection of
rows where EVERY included metric has RS coverage (stricter than Step 2's
per-metric restriction, since now every included metric's diff/abs_diff
must be non-NaN simultaneously).

Step 4 -- permutation test (are Step 2's p-values/AIC even trustworthy?):
cells are NOT independent observations -- every (year, quarter) appears
as the model side of several cells AND the eval side of several others
(e.g. with 10 distinct quarters, each one touches 9+9=18 of the 90
possible off-diagonal cells), so a single bad/unusual quarter
contaminates many rows at once, and OLS's usual p-values/AIC (which
assume independent errors) are overstated -- there's much less
independent evidence here than the row count suggests. Rather than lean
on cluster-robust SEs with too few clusters to trust (see Step 2), this
permutes at the QUARTER level: shuffle which quarter's metric VALUE is
attached to which quarter IDENTITY (keeping every cell's actual
model/eval quarter pairing fixed), rebuild diff_M/abs_diff_M from the
shuffled assignment, refit the rs_gap model, and record its R^2 --
repeated --n-perms times to build a null distribution that preserves the
exact same cluster structure as the real data (every permutation still
has each quarter's value touching the same 18-ish cells it really does,
just a different value). The empirical p-value is how often that null
R^2 matches or beats the real one.

Outputs:
    <out-dir>/<state>_<variable>_cross_quarter_rs_gap.csv   step 1 table
    <out-dir>/<state>_<variable>_rs_gap_regression_coefs.csv   every
        term's coefficient/SE/t/p, every model (rs_gap, season_year_
        dummies, season_pair_dummies, joint), every metric
    <out-dir>/<state>_<variable>_rs_gap_regression_fit.csv   one row per
        (metric, model): R^2, adj R^2, AIC, BIC, n_obs -- the actual
        "compare the fit" table, joint model included with
        metric="joint:<m1>+<m2>+..."
    <out-dir>/<state>_<variable>_rs_gap_permutation.csv   one row per
        metric: observed rs_gap R^2, null R^2 mean/95th percentile,
        permutation p-value, n_obs, n_perms

Usage:
    # Cheaper source (no band-importance run needed):
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv 3_analysis/out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv

    # From band_importance.py's output instead (or in addition):
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --band-importance-csv 3_analysis/outputs/band_importance/oh_wealth_index_band_importance.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv

    # Joint model restricted to just NDVI + cloud_frac (the specific
    # composite-quality check), fewer permutations for a quick look:
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv 3_analysis/out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv \
        --joint-metrics ndvi cloud_frac --n-perms 199
"""

import argparse
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

COL_RE = re.compile(r"^q(\d)_(\d{4})_(.+)$")


def discover_quarter_means(rs_df: pd.DataFrame):
    """Returns (means, metrics): means[(year, quarter)][metric] = the
    state-wide mean of that metric's column across every tract (row) in
    rs_df; metrics is the sorted set of metric names found. Parsed
    directly from column names (q<quarter>_<year>_<metric>) rather than
    importing compute_quarterly_rs_summary.py's METRICS constant, so
    this keeps working if that script's own metric set changes."""
    means = {}
    metrics = set()
    for col in rs_df.columns:
        m = COL_RE.match(col)
        if not m:
            continue
        quarter, year, metric = int(m.group(1)), int(m.group(2)), m.group(3)
        metrics.add(metric)
        means.setdefault((year, quarter), {})[metric] = rs_df[col].mean(skipna=True)
    return means, sorted(metrics)


def load_offdiag_cells_from_band_importance(band_csv: Path) -> pd.DataFrame:
    """r2_base per off-diagonal cell, from band_importance.py's long
    output -- correct, but EXPENSIVE as a source: r2_base there is a
    side effect of a baseline forward pass that then also runs 5
    permutation repeats x 5 band-groups on top, none of which this
    needs. Prefer --cross-quarter-r2-csv (build_cross_quarter_r2_
    matrix.py's cheaper, purpose-built matrix) when that coverage is
    further along; this is here for whichever one you actually have."""
    df = pd.read_csv(band_csv)
    required = {"model_year", "model_quarter", "eval_year", "eval_quarter", "r2_base"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{band_csv} is missing columns: {missing}")

    offdiag = df[(df.model_year != df.eval_year) | (df.model_quarter != df.eval_quarter)]
    if offdiag.empty:
        raise SystemExit(f"{band_csv} has no off-diagonal rows (every row has eval == model) -- "
                          f"re-run band_importance.py with --offdiag first.")

    cells = offdiag.drop_duplicates(
        subset=["model_year", "model_quarter", "eval_year", "eval_quarter"]
    )[["model_year", "model_quarter", "eval_year", "eval_quarter", "r2_base"]].copy()
    cells = cells.rename(columns={"r2_base": "r2"})
    return cells.reset_index(drop=True)


CROSS_QUARTER_LABEL_RE = re.compile(r"^Q(\d)\s+(\d{4})$")


def load_offdiag_cells_from_cross_quarter_matrix(matrix_csv: Path) -> pd.DataFrame:
    """r2_matrix.csv from build_cross_quarter_r2_matrix.py: a wide
    matrix, row/column labels "Q<quarter> <year>" (that script's own
    `labels = [f"Q{q} {y}" for y, q in quarters]`), cell = R^2. Melted
    into the same (model_year, model_quarter, eval_year, eval_quarter,
    r2) shape load_offdiag_cells_from_band_importance returns, skipping
    the diagonal and any cell that's still blank (not validated yet --
    this script is meant to work with a matrix that's still filling in)."""
    df = pd.read_csv(matrix_csv, index_col=0)
    rows = []
    for row_label in df.index:
        m_row = CROSS_QUARTER_LABEL_RE.match(str(row_label).strip())
        if not m_row:
            continue
        model_quarter, model_year = int(m_row.group(1)), int(m_row.group(2))
        for col_label in df.columns:
            m_col = CROSS_QUARTER_LABEL_RE.match(str(col_label).strip())
            if not m_col:
                continue
            eval_quarter, eval_year = int(m_col.group(1)), int(m_col.group(2))
            if (model_year, model_quarter) == (eval_year, eval_quarter):
                continue  # diagonal
            val = df.loc[row_label, col_label]
            if pd.isna(val):
                continue  # this cell hasn't been validated yet
            rows.append({"model_year": model_year, "model_quarter": model_quarter,
                         "eval_year": eval_year, "eval_quarter": eval_quarter, "r2": float(val)})
    if not rows:
        raise SystemExit(f"{matrix_csv} has no filled-in off-diagonal cells yet.")
    return pd.DataFrame(rows)


def load_offdiag_cells(band_csv: "Path | None", matrix_csv: "Path | None") -> pd.DataFrame:
    parts = []
    if band_csv is not None:
        cells = load_offdiag_cells_from_band_importance(band_csv)
        print(f"Loaded {len(cells)} off-diagonal cell(s) from {band_csv}")
        parts.append(cells)
    if matrix_csv is not None:
        cells = load_offdiag_cells_from_cross_quarter_matrix(matrix_csv)
        print(f"Loaded {len(cells)} off-diagonal cell(s) from {matrix_csv}")
        parts.append(cells)

    combined = pd.concat(parts, ignore_index=True)
    key = ["model_year", "model_quarter", "eval_year", "eval_quarter"]
    n_before = len(combined)
    combined = combined.drop_duplicates(subset=key, keep="first")
    if len(parts) > 1 and n_before != len(combined):
        print(f"  ({n_before - len(combined)} cell(s) present in both sources -- kept the "
              f"--band-importance-csv value for those, they should agree anyway)")
    return combined.reset_index(drop=True)


def build_rs_gap_table(cells: pd.DataFrame, means: dict, metrics: list) -> pd.DataFrame:
    out = cells.copy()
    out["same_season"] = (out.model_quarter == out.eval_quarter).astype(int)
    out["same_year"] = (out.model_year == out.eval_year).astype(int)
    # Full (model_quarter -> eval_quarter) identity, e.g. "Q1->Q3" -- unlike
    # same_season (which only knows match/no-match), this lets a dummy
    # model tell Q1-model-on-Q3-imagery apart from Q3-model-on-Q1-imagery.
    out["season_pair"] = out.apply(lambda r: f"Q{int(r.model_quarter)}->Q{int(r.eval_quarter)}", axis=1)

    for metric in metrics:
        diffs = []
        for row in out.itertuples(index=False):
            model_key = (row.model_year, row.model_quarter)
            eval_key = (row.eval_year, row.eval_quarter)
            model_val = means.get(model_key, {}).get(metric)
            eval_val = means.get(eval_key, {}).get(metric)
            if model_val is None or eval_val is None or pd.isna(model_val) or pd.isna(eval_val):
                diffs.append(np.nan)
            else:
                diffs.append(eval_val - model_val)
        out[f"diff_{metric}"] = diffs
        out[f"abs_diff_{metric}"] = out[f"diff_{metric}"].abs()

    return out


def fit_ols(y: pd.Series, X: pd.DataFrame):
    data = pd.concat([y.rename("_y"), X], axis=1).dropna()
    if len(data) < X.shape[1] + 3:
        return None, len(data)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(data["_y"], sm.add_constant(data.drop(columns="_y"))).fit()
    return model, len(data)


def build_season_pair_dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One dummy per distinct season_pair value PRESENT IN df (not every
    theoretically-possible pair), reference category dropped. Built
    fresh per metric's own subset (see run_regressions) rather than once
    for the whole gap_table, since which pairs/how many rows are usable
    varies by metric -- this is deliberate, not an oversight: a pair with
    zero usable rows for this metric shouldn't get a dummy column at all."""
    return pd.get_dummies(df["season_pair"], prefix="pair", drop_first=True).astype(float)


def fit_joint(gap_table: pd.DataFrame, metrics: list):
    """One OLS fit of r2 on diff_M/abs_diff_M for EVERY metric in
    `metrics` simultaneously -- restricted to rows where ALL of them have
    RS coverage at once (stricter than the per-metric restriction in
    run_regressions). Tests whether a metric's association with transfer
    accuracy survives once other metrics -- in particular composite-
    quality proxies like cloud_frac/snow_frac that move with the same
    seasonal axis -- are controlled for in the same model."""
    cols = []
    for m in metrics:
        cols += [f"diff_{m}", f"abs_diff_{m}"]
    subset = gap_table.dropna(subset=cols + ["r2"])
    model, n = fit_ols(subset["r2"], subset[cols])
    return model, n


def permutation_test_rs_gap(cells: pd.DataFrame, means: dict, metric: str,
                             n_perms: int = 999, seed: int = 0):
    """Permutation p-value for metric's rs_gap R^2 that respects the
    quarter-level clustering Step 4 (module docstring) describes: shuffles
    which quarter's metric VALUE is attached to which quarter IDENTITY
    (every cell keeps its real model/eval quarter pairing), rebuilds
    diff_M/abs_diff_M from the shuffled assignment, and refits r2 ~ diff_M
    + abs_diff_M each time. Because a quarter's value (not each cell
    independently) is what gets shuffled, every permuted dataset keeps
    exactly the same dependency structure as the real one -- the quarters
    that share cells in reality still share cells (and now a shuffled
    value) in every permutation. Returns None if fewer than 6 cells have
    usable RS coverage for this metric (same floor fit_ols uses)."""
    quarter_keys = sorted(k for k, v in means.items() if metric in v and pd.notna(v[metric]))
    real_vals = {k: means[k][metric] for k in quarter_keys}

    def diffs_for(value_map):
        model_vals = np.array([value_map.get((r.model_year, r.model_quarter), np.nan)
                                for r in cells.itertuples(index=False)], dtype=float)
        eval_vals = np.array([value_map.get((r.eval_year, r.eval_quarter), np.nan)
                               for r in cells.itertuples(index=False)], dtype=float)
        return eval_vals - model_vals

    observed_diff = diffs_for(real_vals)
    valid = ~np.isnan(observed_diff)
    if valid.sum() < 6:
        return None

    y = cells["r2"].to_numpy()[valid]

    def r2_for(diff_vals):
        X = sm.add_constant(np.column_stack([diff_vals, np.abs(diff_vals)]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return sm.OLS(y, X).fit().rsquared

    observed_r2 = r2_for(observed_diff[valid])

    rng = np.random.default_rng(seed)
    values = np.array([real_vals[k] for k in quarter_keys])
    null_r2 = np.empty(n_perms)
    for i in range(n_perms):
        shuffled = dict(zip(quarter_keys, rng.permutation(values)))
        perm_diff = diffs_for(shuffled)[valid]
        null_r2[i] = r2_for(perm_diff)

    p_value = (1 + int(np.sum(null_r2 >= observed_r2))) / (n_perms + 1)
    return {
        "metric": metric, "observed_r2": observed_r2, "null_r2_mean": float(null_r2.mean()),
        "null_r2_95pct": float(np.quantile(null_r2, 0.95)), "p_value": p_value,
        "n_obs": int(valid.sum()), "n_quarters_permuted": len(quarter_keys), "n_perms": n_perms,
    }


def run_regressions(gap_table: pd.DataFrame, metrics: list):
    coef_rows, fit_rows = [], []

    for metric in metrics:
        diff_col, abs_col = f"diff_{metric}", f"abs_diff_{metric}"
        # All three models fit on the SAME rows -- the intersection where
        # this metric actually has RS coverage -- so n matches and
        # R^2/AIC are comparable across them. Without this, the dummy
        # models (which need no RS data) would use every off-diagonal
        # cell while rs_gap only gets this metric's smaller RS-covered
        # subset.
        subset = gap_table.dropna(subset=[diff_col, abs_col, "r2"])
        pair_dummies = build_season_pair_dummies(subset)

        model_a, n_a = fit_ols(subset["r2"], subset[[diff_col, abs_col]])
        model_b, n_b = fit_ols(subset["r2"], subset[["same_season", "same_year"]])
        model_c, n_c = fit_ols(subset["r2"], pair_dummies)

        fitted = {}
        for label, model, n in [("rs_gap", model_a, n_a), ("season_year_dummies", model_b, n_b),
                                 ("season_pair_dummies", model_c, n_c)]:
            if model is None:
                print(f"[{metric}] {label}: only {n} usable row(s) after dropping NaNs -- skipping")
                continue
            fitted[label] = model
            for term in model.params.index:
                coef_rows.append({
                    "metric": metric, "model": label, "term": term,
                    "coef": model.params[term], "std_err": model.bse[term],
                    "t": model.tvalues[term], "p_value": model.pvalues[term], "n_obs": int(model.nobs),
                })
            fit_rows.append({
                "metric": metric, "model": label, "r_squared": model.rsquared,
                "adj_r_squared": model.rsquared_adj, "aic": model.aic, "bic": model.bic,
                "n_obs": int(model.nobs),
            })
            print(f"[{metric}] {label}: R^2={model.rsquared:.4f} adj_R^2={model.rsquared_adj:.4f} "
                  f"AIC={model.aic:.2f} (n={int(model.nobs)})")

        if fitted:
            best = max(fitted, key=lambda l: fitted[l].rsquared_adj)
            print(f"  -> {best} fits best by adjusted R^2 for {metric}\n")

    return pd.DataFrame(coef_rows), pd.DataFrame(fit_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="Used only for output file naming")
    p.add_argument("--variable", required=True, help="Used only for output file naming")
    p.add_argument("--band-importance-csv", type=Path, default=None,
                    help="band_importance.py's long output. Has r2_base for off-diagonal cells "
                         "as a side effect of its (expensive) permutation analysis -- prefer "
                         "--cross-quarter-r2-csv when that coverage is further along.")
    p.add_argument("--cross-quarter-r2-csv", type=Path, default=None,
                    help="build_cross_quarter_r2_matrix.py's r2_matrix.csv -- cheaper, purpose-"
                         "built source for cell R^2 (no permutation work needed). Pass both if "
                         "you have both; cells present in both are deduplicated, preferring "
                         "--band-importance-csv's value (they should agree).")
    p.add_argument("--rs-summary-csv", required=True, type=Path)
    p.add_argument("--joint-metrics", nargs="+", default=None,
                    help="Metrics to include together in the joint 'does M survive once other "
                         "metrics are controlled for' regression (Step 3). Default: every metric "
                         "found in --rs-summary-csv.")
    p.add_argument("--n-perms", type=int, default=999,
                    help="Permutations for the quarter-level permutation test (Step 4)")
    p.add_argument("--seed", type=int, default=0, help="Permutation RNG seed")
    p.add_argument("--out-dir", default="3_analysis/outputs/cross_quarter_rs_gap", type=Path)
    args = p.parse_args()

    if args.band_importance_csv is None and args.cross_quarter_r2_csv is None:
        raise SystemExit("Pass --band-importance-csv and/or --cross-quarter-r2-csv (at least one)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_offdiag_cells(args.band_importance_csv, args.cross_quarter_r2_csv)
    print(f"Total: {len(cells)} off-diagonal (model, eval) cell(s)")

    rs_df = pd.read_csv(args.rs_summary_csv)
    means, metrics = discover_quarter_means(rs_df)
    print(f"RS summary covers {len(means)} (year, quarter) cell(s), metrics: {metrics}")

    gap_table = build_rs_gap_table(cells, means, metrics)
    usable_per_metric = {m: gap_table[f"diff_{m}"].notna().sum() for m in metrics}
    print(f"Usable cells per metric (both quarters covered by --rs-summary-csv): {usable_per_metric}")

    state_variable = f"{args.state.lower()}_{args.variable}"
    gap_path = args.out_dir / f"{state_variable}_cross_quarter_rs_gap.csv"
    gap_table.to_csv(gap_path, index=False)
    print(f"Wrote {gap_path} ({len(gap_table)} rows)\n")

    coef_df, fit_df = run_regressions(gap_table, metrics)
    if fit_df.empty:
        raise SystemExit("No metric had enough usable cells to fit either model -- "
                          "wait for --rs-summary-csv to cover more quarters.")

    # Step 3: joint model -- does each metric survive once every OTHER
    # metric (cloud_frac, snow_frac, etc.) is controlled for at once?
    joint_metrics = args.joint_metrics if args.joint_metrics else metrics
    unknown = set(joint_metrics) - set(metrics)
    if unknown:
        raise SystemExit(f"--joint-metrics {sorted(unknown)} not found in --rs-summary-csv "
                          f"(available: {metrics})")
    joint_label = "joint:" + "+".join(joint_metrics)
    joint_model, joint_n = fit_joint(gap_table, joint_metrics)
    if joint_model is None:
        print(f"\n[{joint_label}] only {joint_n} usable row(s) after dropping NaNs -- skipping "
              f"(need RS coverage for ALL of {joint_metrics} at once)")
    else:
        print(f"\n[{joint_label}] R^2={joint_model.rsquared:.4f} adj_R^2={joint_model.rsquared_adj:.4f} "
              f"(n={int(joint_model.nobs)})")
        for term in joint_model.params.index:
            coef_rows_joint = {
                "metric": joint_label, "model": "joint", "term": term,
                "coef": joint_model.params[term], "std_err": joint_model.bse[term],
                "t": joint_model.tvalues[term], "p_value": joint_model.pvalues[term],
                "n_obs": int(joint_model.nobs),
            }
            coef_df = pd.concat([coef_df, pd.DataFrame([coef_rows_joint])], ignore_index=True)
            survives = "survives" if joint_model.pvalues[term] < 0.05 else "does NOT survive"
            if term.startswith("diff_") or term.startswith("abs_diff_"):
                print(f"    {term}: coef={joint_model.params[term]:+.4f} "
                      f"p={joint_model.pvalues[term]:.3g} -- {survives} controlling for the others")
        fit_df = pd.concat([fit_df, pd.DataFrame([{
            "metric": joint_label, "model": "joint", "r_squared": joint_model.rsquared,
            "adj_r_squared": joint_model.rsquared_adj, "aic": joint_model.aic, "bic": joint_model.bic,
            "n_obs": int(joint_model.nobs),
        }])], ignore_index=True)

    # Step 4: permutation test -- are the per-metric rs_gap p-values/AIC
    # above even trustworthy, given cells aren't independent (every
    # quarter touches many cells)?
    print(f"\nPermutation test ({args.n_perms} perms, quarter-level shuffle):")
    perm_rows = []
    for metric in metrics:
        result = permutation_test_rs_gap(cells, means, metric, n_perms=args.n_perms, seed=args.seed)
        if result is None:
            print(f"  [{metric}] too few usable cells for a permutation test -- skipping")
            continue
        perm_rows.append(result)
        print(f"  [{metric}] observed R^2={result['observed_r2']:.4f} vs. null mean="
              f"{result['null_r2_mean']:.4f} (95th pct={result['null_r2_95pct']:.4f}) "
              f"-> p={result['p_value']:.4f} (n={result['n_obs']}, "
              f"{result['n_quarters_permuted']} quarters permuted)")
    perm_df = pd.DataFrame(perm_rows)

    coef_path = args.out_dir / f"{state_variable}_rs_gap_regression_coefs.csv"
    fit_path = args.out_dir / f"{state_variable}_rs_gap_regression_fit.csv"
    perm_path = args.out_dir / f"{state_variable}_rs_gap_permutation.csv"
    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    perm_df.to_csv(perm_path, index=False)
    print(f"\nWrote {coef_path} ({len(coef_df)} rows)")
    print(f"Wrote {fit_path} ({len(fit_df)} rows)")
    print(f"Wrote {perm_path} ({len(perm_df)} rows)")


if __name__ == "__main__":
    main()
