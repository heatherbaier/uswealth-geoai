"""
Does the SIZE of the physical remote-sensing gap between a model's own
quarter and the quarter it's evaluated on explain cross-quarter transfer
accuracy better than just "is it the same season" / "is it the same
year"?

Two inputs, joined on (year, quarter):
  --band-importance-csv   band_importance.py's long output. Off-diagonal
                           rows only (eval_year != model_year OR
                           eval_quarter != model_quarter) -- those are
                           the cross-quarter TRANSFER cells; the
                           diagonal is a model on its own quarter, not a
                           transfer at all. r2_base is the same value on
                           every row of a given cell (it's the baseline,
                           computed once before any permutation) --
                           deduplicated down to one row per cell.
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

Step 2 -- for each metric M, two OLS fits of cell r2_base:
    Model A ("rs_gap"):          r2 ~ diff_M + abs_diff_M
    Model B ("season_year_dummies"): r2 ~ same_season + same_year
compared on R^2/adjusted R^2/AIC/BIC -- does a continuous measure of how
different the actual conditions were explain more of the transfer
accuracy than simply knowing whether it's literally the same calendar
quarter/year? Plain OLS, not cluster-robust: the "still building"
dataset this was built for has few off-diagonal cells, and clustered SEs
need more groups than that to be trustworthy -- read the printed n_obs
before trusting any p-value here, and treat this as early/exploratory
per the original request, not a final test.

Outputs:
    <out-dir>/<state>_<variable>_cross_quarter_rs_gap.csv   step 1 table
    <out-dir>/<state>_<variable>_rs_gap_regression_coefs.csv   every
        term's coefficient/SE/t/p, both models, every metric
    <out-dir>/<state>_<variable>_rs_gap_regression_fit.csv   one row per
        (metric, model): R^2, adj R^2, AIC, BIC, n_obs -- the actual
        "compare the fit" table

Usage:
    python 3_analysis/analyze_cross_quarter_rs_gap.py \
        --band-importance-csv 3_analysis/outputs/band_importance/oh_wealth_index_band_importance.csv \
        --rs-summary-csv 3_analysis/outputs/rs_summary/oh_quarterly_rs_summary.csv
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


def load_offdiag_cells(band_csv: Path) -> pd.DataFrame:
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


def build_rs_gap_table(cells: pd.DataFrame, means: dict, metrics: list) -> pd.DataFrame:
    out = cells.copy()
    out["same_season"] = (out.model_quarter == out.eval_quarter).astype(int)
    out["same_year"] = (out.model_year == out.eval_year).astype(int)

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


def run_regressions(gap_table: pd.DataFrame, metrics: list):
    coef_rows, fit_rows = [], []

    for metric in metrics:
        diff_col, abs_col = f"diff_{metric}", f"abs_diff_{metric}"
        model_a, n_a = fit_ols(gap_table["r2"], gap_table[[diff_col, abs_col]])
        model_b, n_b = fit_ols(gap_table["r2"], gap_table[["same_season", "same_year"]])

        for label, model, n in [("rs_gap", model_a, n_a), ("season_year_dummies", model_b, n_b)]:
            if model is None:
                print(f"[{metric}] {label}: only {n} usable row(s) after dropping NaNs -- skipping")
                continue
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

        if model_a is not None and model_b is not None:
            better = "rs_gap" if model_a.rsquared_adj > model_b.rsquared_adj else "season_year_dummies"
            print(f"  -> {better} fits better by adjusted R^2 for {metric}\n")

    return pd.DataFrame(coef_rows), pd.DataFrame(fit_rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--band-importance-csv", required=True, type=Path)
    p.add_argument("--rs-summary-csv", required=True, type=Path)
    p.add_argument("--out-dir", default="3_analysis/outputs/cross_quarter_rs_gap", type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells = load_offdiag_cells(args.band_importance_csv)
    print(f"Loaded {len(cells)} off-diagonal (model, eval) cell(s) from {args.band_importance_csv}")

    rs_df = pd.read_csv(args.rs_summary_csv)
    means, metrics = discover_quarter_means(rs_df)
    print(f"RS summary covers {len(means)} (year, quarter) cell(s), metrics: {metrics}")

    gap_table = build_rs_gap_table(cells, means, metrics)
    usable_per_metric = {m: gap_table[f"diff_{m}"].notna().sum() for m in metrics}
    print(f"Usable cells per metric (both quarters covered by --rs-summary-csv): {usable_per_metric}")

    state_variable = Path(args.band_importance_csv).stem.replace("_band_importance", "")
    gap_path = args.out_dir / f"{state_variable}_cross_quarter_rs_gap.csv"
    gap_table.to_csv(gap_path, index=False)
    print(f"Wrote {gap_path} ({len(gap_table)} rows)\n")

    coef_df, fit_df = run_regressions(gap_table, metrics)
    if fit_df.empty:
        raise SystemExit("No metric had enough usable cells to fit either model -- "
                          "wait for --rs-summary-csv to cover more quarters.")

    coef_path = args.out_dir / f"{state_variable}_rs_gap_regression_coefs.csv"
    fit_path = args.out_dir / f"{state_variable}_rs_gap_regression_fit.csv"
    coef_df.to_csv(coef_path, index=False)
    fit_df.to_csv(fit_path, index=False)
    print(f"\nWrote {coef_path} ({len(coef_df)} rows)")
    print(f"Wrote {fit_path} ({len(fit_df)} rows)")


if __name__ == "__main__":
    main()
