"""
Step 4 of the seasonal-amplitude pipeline: does seasonal instability in
predicted wealth concentrate in particular kinds of tracts? Regresses
every amplitude measure from compute_amplitude.py on the tract
covariates from compute_tract_covariates.py, one model per state plus
one pooled model with state fixed effects, and tests whether the result
survives z-scoring and the quality controls.

Outcomes (8): seasonal_amplitude_raw/_z, noise_floor_raw/_z,
model_disagreement_raw/_z, and amplitude_minus_noise_raw/_z
(seasonal_amplitude - noise_floor, same variant). Each is log1p-
transformed if its skewness exceeds --skew-threshold (default 1.0) --
EXCEPT the two "minus noise floor" outcomes, which can be negative (a
difference of two non-negative quantities), so log-transforming them is
never valid regardless of skew; that exception is applied unconditionally,
not just when skew happens to be high.

Covariates (13, all standardized -- z-scored WITHIN each regression's own
analysis sample, i.e. a state's own mean/std for its own-state model, the
full pooled sample's mean/std for the pooled model -- not one global
standardization reused everywhere, a real modeling choice worth stating
explicitly since the original request didn't specify it): wealth_index,
renter_pct, population_density, white_pct, black_pct, asian_pct,
hispanic_pct, tree_canopy_pct, cropland_pct, tract_area_km2,
valid_obs_frac_mean, cloud_frac_mean, snow_frac_mean. A covariate that's
ENTIRELY missing for a state (--lc-csv or --chip-quality-csv wasn't given
to compute_tract_covariates.py for that state) is dropped from THAT
state's regression with a printed note, not left in to silently drop
every single row (statsmodels' missing='drop' would otherwise drop every
row, since an all-NaN column has no non-missing rows at all).

Standard errors are clustered by county (GEOID[:5], added by compute_
tract_covariates.py) via statsmodels' cov_type='cluster'. Moran's I of
each fitted model's residuals is computed with analyze_band_importance_
spatial.py's own morans_i() (imported, not reimplemented) using tract
centroid lat/lon from compute_tract_covariates.py.

Outputs:
    <out-dir>/regression_table.csv     one row per (state_or_pooled,
                                        outcome, covariate): coefficient,
                                        std_err, t, p_value, logged,
                                        n_obs
    <out-dir>/morans_i_table.csv       one row per (state_or_pooled,
                                        outcome): Moran's I of residuals,
                                        p_value
    <fig-dir>/amplitude_by_wealth_decile.png   mean seasonal_amplitude_raw
                                        by wealth_index decile, one line
                                        per state, noise_floor_raw shown
                                        as a reference line per state
    <out-dir>/summary.txt              short written summary (see
                                        write_summary())

Usage:
    python 3_analysis/analyze_amplitude_covariates.py --states oh pa az \
        --variable wealth_index
"""

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skew

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_band_importance_spatial import morans_i  # noqa: E402

OUTCOME_BASE = ["seasonal_amplitude", "noise_floor", "model_disagreement"]
VARIANTS = ["raw", "z"]
DIFF_OUTCOMES = ["amplitude_minus_noise_raw", "amplitude_minus_noise_z"]  # never logged -- can be negative

COVARIATE_COLS = [
    "wealth_index", "renter_pct", "population_density", "white_pct", "black_pct",
    "asian_pct", "hispanic_pct", "tree_canopy_pct", "cropland_pct", "tract_area_km2",
    "valid_obs_frac_mean", "cloud_frac_mean", "snow_frac_mean",
]


def load_and_merge(state, amplitude_dir, covariates_dir, variable):
    amp_path = amplitude_dir / f"{state}_{variable}_amplitude.csv"
    cov_path = covariates_dir / f"{state}_tract_covariates.csv"
    if not amp_path.exists():
        print(f"[{state.upper()}] no amplitude file at {amp_path}, skipping")
        return None
    if not cov_path.exists():
        print(f"[{state.upper()}] no covariates file at {cov_path}, skipping")
        return None

    amp = pd.read_csv(amp_path, dtype={"GEOID": str})
    cov = pd.read_csv(cov_path, dtype={"GEOID": str, "county_fips": str})
    df = amp.merge(cov, on="GEOID", how="inner", suffixes=("", "_cov"))
    print(f"[{state.upper()}] {len(amp)} amplitude rows x {len(cov)} covariate rows -> "
          f"{len(df)} merged tracts")

    df["amplitude_minus_noise_raw"] = df["seasonal_amplitude_raw"] - df["noise_floor_raw"]
    df["amplitude_minus_noise_z"] = df["seasonal_amplitude_z"] - df["noise_floor_z"]
    return df


def maybe_log(y: pd.Series, outcome_name: str, skew_threshold: float):
    """Returns (transformed_y, was_logged). Never logs a DIFF outcome
    (can be negative) regardless of skew -- see module docstring."""
    if outcome_name in DIFF_OUTCOMES:
        return y, False
    valid = y.dropna()
    if len(valid) < 8 or (valid < 0).any():
        return y, False
    s = skew(valid)
    if abs(s) > skew_threshold:
        return np.log1p(y), True
    return y, False


def standardize(df: pd.DataFrame, cols: list) -> "tuple[pd.DataFrame, list]":
    """z-scores each column within df's own rows; drops (with a printed
    note) any column that's entirely NaN here -- an all-NaN column has
    zero std (and zero non-missing rows), so leaving it in would make
    statsmodels' missing='drop' drop every row instead of just that
    covariate."""
    usable = [c for c in cols if df[c].notna().any()]
    dropped = [c for c in cols if c not in usable]
    if dropped:
        print(f"    dropping entirely-missing covariate(s) for this regression: {dropped}")
    z = pd.DataFrame(index=df.index)
    for c in usable:
        mu, sd = df[c].mean(), df[c].std(ddof=1)
        z[c] = (df[c] - mu) / sd if sd and sd > 0 else 0.0
    return z, usable


def fit_one(df: pd.DataFrame, outcome_col: str, covariate_cols: list, group_col: str,
            state_label: str, extra_dummies: "pd.DataFrame | None" = None):
    y_raw = df[outcome_col]
    y, logged = maybe_log(y_raw, outcome_col, skew_threshold=1.0)

    X, usable_covs = standardize(df, covariate_cols)
    if extra_dummies is not None:
        X = pd.concat([X, extra_dummies.loc[X.index]], axis=1)
    X = sm.add_constant(X)

    data = pd.concat([y.rename("_y"), X, df[[group_col]]], axis=1).dropna()
    if len(data) < len(usable_covs) + 5:
        print(f"    [{state_label} / {outcome_col}] only {len(data)} usable rows after dropping "
              f"NaNs -- skipping this regression")
        return None, None

    y_fit = data["_y"]
    X_fit = data.drop(columns=["_y", group_col])
    groups = data[group_col]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y_fit, X_fit, missing="drop").fit(cov_type="cluster", cov_kwds={"groups": groups})

    coef_rows = []
    for cov in X_fit.columns:
        coef_rows.append({
            "state": state_label, "outcome": outcome_col, "covariate": cov,
            "coef": model.params[cov], "std_err": model.bse[cov],
            "t": model.tvalues[cov], "p_value": model.pvalues[cov],
            "logged": logged, "n_obs": int(model.nobs),
        })

    resid = model.resid
    lat = df.loc[resid.index, "lat"].to_numpy()
    lon = df.loc[resid.index, "lon"].to_numpy()
    moran = morans_i(resid.to_numpy(), lat, lon)
    moran_row = {"state": state_label, "outcome": outcome_col, "moran_i": moran["I"],
                 "moran_p_value": moran["p_value"], "n_obs": int(model.nobs), "logged": logged}

    return coef_rows, moran_row


def run_all_regressions(pooled: pd.DataFrame, states: list, skew_threshold: float):
    outcomes = [f"{base}_{variant}" for base in OUTCOME_BASE for variant in VARIANTS] + DIFF_OUTCOMES
    coef_rows, moran_rows = [], []

    for state in states:
        sub = pooled[pooled.state == state]
        if sub.empty:
            continue
        print(f"\n[{state.upper()}] per-state regressions ({len(sub)} tracts)")
        for outcome in outcomes:
            coefs, moran_row = fit_one(sub, outcome, COVARIATE_COLS, "county_fips", state.upper())
            if coefs is not None:
                coef_rows += coefs
                moran_rows.append(moran_row)

    if len(states) > 1:
        print(f"\n[POOLED] with state fixed effects ({len(pooled)} tracts)")
        state_dummies = pd.get_dummies(pooled["state"], prefix="state", drop_first=True).astype(float)
        for outcome in outcomes:
            coefs, moran_row = fit_one(pooled, outcome, COVARIATE_COLS, "county_fips", "POOLED",
                                        extra_dummies=state_dummies)
            if coefs is not None:
                coef_rows += coefs
                moran_rows.append(moran_row)

    return pd.DataFrame(coef_rows), pd.DataFrame(moran_rows)


def plot_amplitude_by_decile(pooled: pd.DataFrame, states: list, out_path: Path):
    fig, ax = plt.subplots(figsize=(9, 6))
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948", "#008300"]

    for i, state in enumerate(states):
        sub = pooled[pooled.state == state].copy()
        if sub["wealth_index"].notna().sum() < 10:
            continue
        sub["decile"] = pd.qcut(sub["wealth_index"], 10, labels=False, duplicates="drop")
        agg = sub.groupby("decile")[["seasonal_amplitude_raw", "noise_floor_raw"]].mean()
        color = palette[i % len(palette)]
        ax.plot(agg.index + 1, agg["seasonal_amplitude_raw"], "-o", color=color,
                label=f"{state.upper()} seasonal amplitude", linewidth=2)
        ax.plot(agg.index + 1, agg["noise_floor_raw"], "--", color=color, alpha=0.6,
                label=f"{state.upper()} noise floor")

    ax.set_xlabel("Wealth index decile (1 = poorest)")
    ax.set_ylabel("Mean SD of predictions")
    ax.set_title("Seasonal amplitude vs. noise floor, by wealth decile")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def write_summary(coef_df: pd.DataFrame, pooled: pd.DataFrame, out_path: Path):
    lines = ["Seasonal amplitude vs. tract covariates -- summary", "=" * 55, ""]

    pooled_rows = coef_df[(coef_df.state == "POOLED") & (coef_df.outcome == "seasonal_amplitude_raw")]
    key_covs = ["wealth_index", "population_density", "cropland_pct", "tree_canopy_pct"]

    if pooled_rows.empty:
        lines.append("No pooled seasonal_amplitude_raw model fit (not enough states/data) -- "
                      "falling back to whatever per-state models exist below.")
    else:
        lines.append("Pooled model (state fixed effects), outcome = seasonal_amplitude_raw:")
        for cov in key_covs:
            row = pooled_rows[pooled_rows.covariate == cov]
            if row.empty:
                continue
            r = row.iloc[0]
            sig = "significant" if r.p_value < 0.05 else "not significant"
            direction = "higher" if r.coef > 0 else "lower"
            lines.append(f"  {cov}: coef={r.coef:+.4f} (p={r.p_value:.3g}, {sig}) -- "
                          f"a 1-SD increase in {cov} is associated with {direction} seasonal "
                          f"amplitude")
        lines.append("")

        wealth_row = pooled_rows[pooled_rows.covariate == "wealth_index"]
        pop_row = pooled_rows[pooled_rows.covariate == "population_density"]
        crop_row = pooled_rows[pooled_rows.covariate == "cropland_pct"]
        claims = []
        if not wealth_row.empty and wealth_row.iloc[0].coef < 0 and wealth_row.iloc[0].p_value < 0.05:
            claims.append("poorer")
        if not pop_row.empty and pop_row.iloc[0].coef < 0 and pop_row.iloc[0].p_value < 0.05:
            claims.append("more rural (lower population density)")
        if not crop_row.empty and crop_row.iloc[0].coef > 0 and crop_row.iloc[0].p_value < 0.05:
            claims.append("more agricultural (higher cropland share)")
        if claims:
            lines.append(f"HEADLINE: instability concentrates in tracts that are {', '.join(claims)} "
                          f"(pooled model, raw predictions, p<0.05 on the relevant covariate).")
        else:
            lines.append("HEADLINE: no covariate among wealth_index/population_density/cropland_pct "
                          "showed a significant association with seasonal amplitude in the pooled, "
                          "raw-prediction model -- no clear concentration pattern to report from "
                          "this run.")
        lines.append("")

    # Robustness: does the same pattern hold in the z-scored version?
    z_rows = coef_df[(coef_df.state == "POOLED") & (coef_df.outcome == "seasonal_amplitude_z")]
    if not z_rows.empty and not pooled_rows.empty:
        lines.append("Robustness to z-scoring (removes a per-cell calibration shift):")
        for cov in key_covs:
            raw_r = pooled_rows[pooled_rows.covariate == cov]
            z_r = z_rows[z_rows.covariate == cov]
            if raw_r.empty or z_r.empty:
                continue
            raw_sig = raw_r.iloc[0].p_value < 0.05
            z_sig = z_r.iloc[0].p_value < 0.05
            same_sign = np.sign(raw_r.iloc[0].coef) == np.sign(z_r.iloc[0].coef)
            verdict = "SURVIVES" if (raw_sig == z_sig and same_sign) else "DOES NOT fully survive"
            lines.append(f"  {cov}: raw p={raw_r.iloc[0].p_value:.3g}, z p={z_r.iloc[0].p_value:.3g} "
                          f"-> {verdict} z-scoring")
        lines.append("")

    # Robustness: quality controls (valid_obs/cloud/snow) -- were they even usable, and
    # did including them change anything material? (Coarse check: were they significant
    # themselves, which would suggest composite quality itself drives some of the result.)
    quality_rows = coef_df[(coef_df.state == "POOLED") & (coef_df.outcome == "seasonal_amplitude_raw")
                            & (coef_df.covariate.isin(["valid_obs_frac_mean", "cloud_frac_mean",
                                                        "snow_frac_mean"]))]
    if quality_rows.empty:
        lines.append("Quality controls: not available for this run (no --chip-quality-csv given "
                      "to compute_tract_covariates.py for the states included) -- the result above "
                      "has NOT been checked against composite-quality confounding.")
    else:
        sig_quality = quality_rows[quality_rows.p_value < 0.05]
        if sig_quality.empty:
            lines.append("Quality controls: none of valid_obs_frac_mean/cloud_frac_mean/"
                          "snow_frac_mean were significant themselves -- composite quality doesn't "
                          "look like it's driving the result above.")
        else:
            lines.append(f"Quality controls: {list(sig_quality.covariate)} WERE significant -- "
                          f"part of the seasonal-amplitude pattern may be composite-quality-"
                          f"related (cloud/snow/missing-data proxies, not real landscape/wealth "
                          f"structure). Worth a closer look before trusting the headline above.")

    lines.append("")
    lines.append("See regression_table.csv for every coefficient (all states + pooled, all 8 "
                  "outcomes) and morans_i_table.csv for residual spatial autocorrelation per model.")

    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print("\n" + "\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--amplitude-dir", default="3_analysis/outputs/amplitude", type=Path)
    p.add_argument("--covariates-dir", default="3_analysis/outputs/amplitude", type=Path)
    p.add_argument("--out-dir", default="3_analysis/outputs/amplitude", type=Path)
    p.add_argument("--fig-dir", default="3_analysis/figures/amplitude", type=Path)
    p.add_argument("--skew-threshold", type=float, default=1.0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    states = [s.lower() for s in args.states]
    parts = []
    for state in states:
        df = load_and_merge(state, args.amplitude_dir, args.covariates_dir, args.variable)
        if df is not None:
            parts.append(df)

    if not parts:
        raise SystemExit("No state had both an amplitude file and a covariates file -- nothing to analyze.")
    pooled = pd.concat(parts, ignore_index=True)
    usable_states = sorted(pooled["state"].unique())

    coef_df, moran_df = run_all_regressions(pooled, usable_states, args.skew_threshold)
    coef_path = args.out_dir / "regression_table.csv"
    moran_path = args.out_dir / "morans_i_table.csv"
    coef_df.to_csv(coef_path, index=False)
    moran_df.to_csv(moran_path, index=False)
    print(f"\nWrote {coef_path} ({len(coef_df):,} rows)")
    print(f"Wrote {moran_path} ({len(moran_df):,} rows)")

    plot_amplitude_by_decile(pooled, usable_states, args.fig_dir / "amplitude_by_wealth_decile.png")
    write_summary(coef_df, pooled, args.out_dir / "summary.txt")


if __name__ == "__main__":
    main()
