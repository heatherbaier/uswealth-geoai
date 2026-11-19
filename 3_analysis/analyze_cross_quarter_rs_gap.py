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

--years (default 2017 2018 2019) restricts EVERY state to this same
year range before any fitting -- both the off-diagonal cells (a cell is
dropped if either its model_year or eval_year falls outside --years)
and the RS summary means (a (year, quarter) outside --years is removed
from `means` entirely, so it can never be chosen as a same-season-
control candidate or counted in a quarter-level permutation). Applied
uniformly so results are comparable state to state regardless of which
years each state happens to have the most trained models for.

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

IGNORE THE JOINT MODEL'S PER-TERM RESULTS -- kept in the code (and still
printed/written) as a diagnostic, not deleted, but its coefficients are
NOT reliable evidence about any individual metric. Confirmed failure
mode, not a hypothetical one: Arizona's joint fit has EVERY term
"survive" (p<0.05) with offsetting (opposite-sign, largely canceling)
coefficients -- the textbook symptom of collinearity among the
simultaneous predictors, not N independent metrics each truly mattering.
When predictors move together as tightly as these RS metrics do, OLS
can and will assign them large, unstable, sign-flipping coefficients
that fit jointly without any one of them being individually meaningful.
Use Step 5 (the fixed-order sequential fit) instead for a trustworthy
per-variable test -- entering terms one block at a time sidesteps this
failure mode entirely, since blocks are never simultaneously competing
for the same variance.

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

Step 5 -- fixed-order sequential fit (the trustworthy alternative to
Step 3's joint model): three nested OLS fits of cell r2, entered in a
FIXED order, each adding one block on top of the last --
    Block 1 ("quality"):            r2 ~ diff/abs_diff of
        --quality-metrics (default: cloud_frac, valid_obs_frac)
    Block 2 ("+season_pair"):       Block 1 + season_pair_dummies
    Block 3 ("+<landscape metric>"): Block 2 + diff/abs_diff of
        --landscape-metric (default: ndvi) -- ONE variable, entered LAST
All three fit on the SAME subset of rows (every quality metric's diff/
abs_diff, season_pair, and the landscape metric's diff/abs_diff all
non-NaN at once), so each block's R^2 is directly comparable to the
last. Reports each block's own R^2 and the R^2 ADDED by that block over
the previous one -- composite-quality confounds and calendar identity
are given first claim on the variance, so whatever the landscape
variable adds in Block 3 is variance NEITHER of those could already
explain, not a number competing simultaneously with them the way a
joint model's coefficients do.

Step 6 -- permutation test for Step 5's Block-3 increment: permutes the
landscape metric's values across YEARS *WITHIN EACH SEASON* (quarter) --
not across quarters the way Step 4 does -- since season_pair_dummies is
already a fixed, earlier block here; the question specific to Block 3
is "does the landscape metric's year-to-year variation WITHIN a given
season carry real signal, or would any random year's value at that
season do just as well?" Block 2 (quality + season_pair) is fit once;
each permutation reshuffles which year's value lands at which (year,
quarter) within its own quarter, rebuilds the landscape metric's diff/
abs_diff from that shuffled assignment, refits Block 3 on top of the
SAME unchanged Block 2, and records the R^2 added. The empirical
p-value is how often that null R^2-added matches or beats the real one.

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
    <out-dir>/<state>_<variable>_rs_gap_sequential_fit.csv   one row per
        block (quality, +season_pair, +<landscape metric>): R^2,
        R^2 added over the previous block, n_obs
    <out-dir>/<state>_<variable>_rs_gap_sequential_permutation.csv   one
        row: Block 3's observed R^2-added, null mean/95th percentile,
        permutation p-value, n_obs, n_perms (years-within-season shuffle)

Usage:
    # Cheaper source (no band-importance run needed), restricted by
    # default to --years 2017 2018 2019:
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv 3_analysis/out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv

    # From band_importance.py's output instead (or in addition):
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --band-importance-csv 3_analysis/outputs/band_importance/oh_wealth_index_band_importance.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv

    # Joint model restricted to just NDVI + cloud_frac (the specific
    # composite-quality check -- still diagnostic-only, see Step 3),
    # fewer permutations for a quick look, landscape metric = ndwi for
    # the Step 5/6 sequential fit instead of the default ndvi:
    python 3_analysis/analyze_cross_quarter_rs_gap.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv 3_analysis/out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv \
        --joint-metrics ndvi cloud_frac --landscape-metric ndwi --n-perms 199
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


def fit_sequential(gap_table: pd.DataFrame, quality_metrics: list, landscape_metric: str):
    """Three nested OLS fits of cell r2, in a FIXED order (see Step 5,
    module docstring): quality metrics' diff/abs_diff first, then
    season_pair_dummies, then the one landscape metric last. All three
    fit on the SAME subset of rows (every quality metric's diff/
    abs_diff, season_pair, and the landscape metric's diff/abs_diff all
    non-NaN at once) so each block's R^2 is directly comparable to the
    one before it. Returns (OrderedDict of {block_name: model_or_None},
    n_obs) -- a block is None if that subset is too small for fit_ols's
    own floor, same as every other model in this script."""
    quality_cols = []
    for m in quality_metrics:
        quality_cols += [f"diff_{m}", f"abs_diff_{m}"]
    landscape_cols = [f"diff_{landscape_metric}", f"abs_diff_{landscape_metric}"]

    needed = quality_cols + landscape_cols + ["r2"]
    subset = gap_table.dropna(subset=needed)
    if subset.empty:
        return {}, 0

    pair_dummies = build_season_pair_dummies(subset)
    step1_X = subset[quality_cols]
    step2_X = pd.concat([step1_X, pair_dummies], axis=1)
    step3_X = pd.concat([step2_X, subset[landscape_cols]], axis=1)

    blocks = {}
    for block_name, X in [
        ("quality", step1_X),
        ("quality+season_pair", step2_X),
        (f"quality+season_pair+{landscape_metric}", step3_X),
    ]:
        model, _ = fit_ols(subset["r2"], X)
        blocks[block_name] = model
    return blocks, len(subset)


def run_sequential_regression(gap_table: pd.DataFrame, quality_metrics: list, landscape_metric: str):
    blocks, n = fit_sequential(gap_table, quality_metrics, landscape_metric)
    if not blocks or any(m is None for m in blocks.values()):
        print(f"  only {n} usable row(s) after dropping NaNs -- skipping the sequential fit "
              f"(need RS coverage for {quality_metrics + [landscape_metric]} AND season_pair at once)")
        return pd.DataFrame()

    rows = []
    prev_r2 = 0.0
    for block_name, model in blocks.items():
        added = model.rsquared - prev_r2
        rows.append({"block": block_name, "r_squared": model.rsquared, "r_squared_added": added, "n_obs": n})
        print(f"  [{block_name}] R^2={model.rsquared:.4f} (+{added:.4f}) n={n}")
        prev_r2 = model.rsquared
    return pd.DataFrame(rows)


def permute_years_within_season(means: dict, metric: str, quarters: list, years: list, rng) -> dict:
    """Returns {(year, quarter): shuffled_value} for `metric` only --
    each quarter's set of (year, value) pairs is shuffled INDEPENDENTLY
    across the years present for that quarter (season/quarter identity
    never changes; only which year's real value lands where, within its
    own quarter). See Step 6, module docstring, for why this -- not a
    quarter-level shuffle -- is the right null for Block 3's increment
    once season_pair_dummies is already a fixed earlier block."""
    shuffled = {}
    for q in quarters:
        keys = [(y, q) for y in years if (y, q) in means and metric in means[(y, q)]
                and pd.notna(means[(y, q)][metric])]
        if not keys:
            continue
        values = np.array([means[k][metric] for k in keys])
        permuted_values = rng.permutation(values)
        for k, v in zip(keys, permuted_values):
            shuffled[k] = v
    return shuffled


def permutation_test_sequential(gap_table: pd.DataFrame, means: dict, quality_metrics: list,
                                 landscape_metric: str, quarters: list, years: list,
                                 n_perms: int = 999, seed: int = 0):
    """Permutation p-value for Step 5's Block-3 R^2-added, permuting the
    landscape metric's values across years WITHIN EACH SEASON (see
    permute_years_within_season). Block 2 (quality + season_pair) is fit
    ONCE outside the loop -- it never changes across permutations, only
    Block 3 (which adds the landscape metric on top) does. Returns None
    if the base subset (quality metrics + season_pair, ignoring the
    landscape metric) has too few usable rows."""
    quality_cols = []
    for m in quality_metrics:
        quality_cols += [f"diff_{m}", f"abs_diff_{m}"]
    landscape_diff_col, landscape_abs_col = f"diff_{landscape_metric}", f"abs_diff_{landscape_metric}"

    subset = gap_table.dropna(subset=quality_cols + [landscape_diff_col, landscape_abs_col, "r2"])
    if len(subset) < len(quality_cols) + 6:
        return None

    pair_dummies = build_season_pair_dummies(subset)
    step2_X = pd.concat([subset[quality_cols], pair_dummies], axis=1)
    model2, _ = fit_ols(subset["r2"], step2_X)
    if model2 is None:
        return None

    def diffs_for(value_map):
        model_vals = np.array([value_map.get((r.model_year, r.model_quarter), np.nan)
                                for r in subset.itertuples(index=False)], dtype=float)
        eval_vals = np.array([value_map.get((r.eval_year, r.eval_quarter), np.nan)
                               for r in subset.itertuples(index=False)], dtype=float)
        return eval_vals - model_vals

    def r2_added_for(land_diff_values):
        X3 = step2_X.copy()
        X3[landscape_diff_col] = land_diff_values
        X3[landscape_abs_col] = np.abs(land_diff_values)
        model3, _ = fit_ols(subset["r2"], X3)
        if model3 is None:
            return np.nan
        return model3.rsquared - model2.rsquared

    real_value_map = {k: v[landscape_metric] for k, v in means.items() if landscape_metric in v}
    observed_diff = diffs_for(real_value_map)
    if np.isnan(observed_diff).any():
        # subset was already restricted to rows with non-NaN diff_<landscape_metric>, so a
        # NaN here would mean the stored diff_ column and a fresh recompute disagree --
        # fail loudly rather than silently feeding a bad fit.
        raise RuntimeError("diffs_for(real_value_map) produced NaN(s) on rows already filtered "
                            "to have a usable diff_<landscape_metric> -- data/means mismatch")
    observed_added = r2_added_for(observed_diff)

    rng = np.random.default_rng(seed)
    null_added = np.empty(n_perms)
    for i in range(n_perms):
        shuffled_map = permute_years_within_season(means, landscape_metric, quarters, years, rng)
        null_added[i] = r2_added_for(diffs_for(shuffled_map))

    valid_null = null_added[~np.isnan(null_added)]
    p_value = (1 + int(np.sum(valid_null >= observed_added))) / (len(valid_null) + 1) if len(valid_null) else np.nan
    return {
        "observed_r2_added": observed_added, "null_mean": float(np.nanmean(null_added)),
        "null_95pct": float(np.nanquantile(valid_null, 0.95)) if len(valid_null) else np.nan,
        "p_value": p_value, "n_obs": len(subset), "n_perms": n_perms,
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
    p.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019],
                    help="Restrict every state to this year range before any fitting -- both the "
                         "off-diagonal cells and the RS summary means (default: 2017 2018 2019)")
    p.add_argument("--joint-metrics", nargs="+", default=None,
                    help="Metrics to include together in the joint 'does M survive once other "
                         "metrics are controlled for' regression (Step 3) -- diagnostic only, see "
                         "that step's docstring. Default: every metric found in --rs-summary-csv.")
    p.add_argument("--quality-metrics", nargs="+", default=["cloud_frac", "valid_obs_frac"],
                    help="Composite-quality metrics entered FIRST in the Step 5 sequential fit "
                         "(default: cloud_frac valid_obs_frac)")
    p.add_argument("--landscape-metric", default="ndvi",
                    help="The ONE landscape metric entered LAST in the Step 5 sequential fit "
                         "(default: ndvi)")
    p.add_argument("--n-perms", type=int, default=999,
                    help="Permutations for the quarter-level (Step 4) and years-within-season "
                         "(Step 6) permutation tests")
    p.add_argument("--seed", type=int, default=0, help="Permutation RNG seed")
    p.add_argument("--out-dir", default="3_analysis/outputs/cross_quarter_rs_gap", type=Path)
    args = p.parse_args()

    if args.band_importance_csv is None and args.cross_quarter_r2_csv is None:
        raise SystemExit("Pass --band-importance-csv and/or --cross-quarter-r2-csv (at least one)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_offdiag_cells(args.band_importance_csv, args.cross_quarter_r2_csv)
    n_before_years = len(cells)
    cells = cells[cells.model_year.isin(args.years) & cells.eval_year.isin(args.years)].reset_index(drop=True)
    print(f"Total: {len(cells)} off-diagonal (model, eval) cell(s) within --years {args.years} "
          f"({n_before_years - len(cells)} dropped for falling outside it)")

    rs_df = pd.read_csv(args.rs_summary_csv)
    means, metrics = discover_quarter_means(rs_df)
    n_quarters_before_years = len(means)
    means = {k: v for k, v in means.items() if k[0] in args.years}
    print(f"RS summary covers {len(means)} (year, quarter) cell(s) within --years {args.years} "
          f"({n_quarters_before_years - len(means)} dropped), metrics: {metrics}")

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
        print("  NOTE: joint-model coefficients above are DIAGNOSTIC ONLY -- collinearity among "
              "these RS metrics can (and for some states, does -- see module docstring's Step 3) "
              "make every term look individually significant with offsetting coefficients. Use "
              "the Step 5 sequential fit below for a trustworthy per-variable test.")

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

    # Step 5: fixed-order sequential fit -- quality metrics first, then
    # season_pair_dummies, then the one landscape metric last. The
    # trustworthy alternative to Step 3's joint model (see that step's
    # docstring for why its per-term results are diagnostic-only).
    unknown_quality = set(args.quality_metrics) - set(metrics)
    if unknown_quality:
        raise SystemExit(f"--quality-metrics {sorted(unknown_quality)} not found in --rs-summary-csv "
                          f"(available: {metrics})")
    if args.landscape_metric not in metrics:
        raise SystemExit(f"--landscape-metric {args.landscape_metric!r} not found in --rs-summary-csv "
                          f"(available: {metrics})")
    print(f"\nSequential fit (fixed order: {args.quality_metrics} -> season_pair -> "
          f"{args.landscape_metric}):")
    sequential_df = run_sequential_regression(gap_table, args.quality_metrics, args.landscape_metric)

    # Step 6: permutation test for Block 3's R^2-added, years permuted
    # within each season (not across quarters -- season_pair is already
    # a fixed earlier block here, see that function's docstring).
    print(f"\nSequential permutation test ({args.n_perms} perms, years-within-season shuffle):")
    quarters_present = sorted(cells["model_quarter"].unique().tolist() + cells["eval_quarter"].unique().tolist())
    quarters_present = sorted(set(quarters_present))
    seq_perm_result = permutation_test_sequential(
        gap_table, means, args.quality_metrics, args.landscape_metric, quarters_present, args.years,
        n_perms=args.n_perms, seed=args.seed,
    )
    if seq_perm_result is None:
        print("  too few usable cells for the sequential permutation test -- skipping")
        seq_perm_df = pd.DataFrame()
    else:
        print(f"  Block 3 ({args.landscape_metric}) R^2 added: observed="
              f"{seq_perm_result['observed_r2_added']:.4f} vs. null mean="
              f"{seq_perm_result['null_mean']:.4f} (95th pct={seq_perm_result['null_95pct']:.4f}) "
              f"-> p={seq_perm_result['p_value']:.4f} (n={seq_perm_result['n_obs']})")
        seq_perm_df = pd.DataFrame([{"landscape_metric": args.landscape_metric, **seq_perm_result}])

    coef_path = args.out_dir / f"{state_variable}_rs_gap_regression_coefs.csv"
    fit_path = args.out_dir / f"{state_variable}_rs_gap_regression_fit.csv"
    perm_path = args.out_dir / f"{state_variable}_rs_gap_permutation.csv"
    sequential_path = args.out_dir / f"{state_variable}_rs_gap_sequential_fit.csv"
    sequential_perm_path = args.out_dir / f"{state_variable}_rs_gap_sequential_permutation.csv"
    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    perm_df.to_csv(perm_path, index=False)
    sequential_df.to_csv(sequential_path, index=False)
    seq_perm_df.to_csv(sequential_perm_path, index=False)
    print(f"\nWrote {coef_path} ({len(coef_df)} rows)")
    print(f"Wrote {fit_path} ({len(fit_df)} rows)")
    print(f"Wrote {perm_path} ({len(perm_df)} rows)")
    print(f"Wrote {sequential_path} ({len(sequential_df)} rows)")
    print(f"Wrote {sequential_perm_path} ({len(seq_perm_df)} rows)")


if __name__ == "__main__":
    main()
