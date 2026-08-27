"""
Analyze seasonal variation and temporal lag in imagery-based wealth
prediction, moderated by landscape composition.

Inputs (both DataFrames indexed / keyed by tract GEOID):
    preds_df:  wide-format with columns:
        GEOID, q{1..4}_{year}_pred, q{1..4}_{year}_error
        (error = pred - truth, so error² is squared error)
    lc_df:     wide-format with columns:
        Tree cover, Grassland, Cropland, Built-up, ...
        Values may be raw pixel counts OR percentages — the script
        normalizes to fractions before use.

Outputs:
    - Long-format merged panel written to CSV
    - Per-quarter R² and RMSE by landscape group (CSV + figure)
    - Seasonal-amplitude paired comparison (agricultural vs urban)
    - Linear-slope-in-time comparison (agricultural vs urban)
    - Mixed model summaries testing the landscape × time interactions
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import statsmodels.formula.api as smf


# -------------------------------------------------------------------------
# Config
# -------------------------------------------------------------------------
PREDS_CSV = "./preds.csv"   # <<< change me
LC_CSV    = "./lc.csv"     # <<< change me
OUT_DIR   = Path("/home/hbaier/projects/tlags_v2/out")   # <<< change me

# Thresholds for the categorical typology. Adjust after looking at
# lc_df[["Cropland", "Built-up"]].hist() on your actual data.
AG_CROPLAND_THRESHOLD   = 0.30   # tract is 'agricultural' if cropland fraction >= 0.50
URBAN_BUILT_THRESHOLD   = 0.30   # tract is 'urban'        if built-up fraction >= 0.40

# Minimum tract count in a (landscape × year × quarter) cell for R² to be
# considered reliable. Cells below this are still plotted but flagged.
MIN_CELL_N = 30


# -------------------------------------------------------------------------
# Step 1 — load + merge + reshape into long format
# -------------------------------------------------------------------------
def load_and_merge(preds_csv, lc_csv):
    preds = pd.read_csv(preds_csv, dtype={"GEOID": str})
    lc = pd.read_csv(lc_csv, dtype={0: str}, index_col=0)
    lc.index.name = "GEOID"
    lc = lc.reset_index()

    # Normalize land cover to fractions if it looks like raw counts / m^2
    lc_class_cols = [c for c in lc.columns if c != "GEOID"]
    row_sums = lc[lc_class_cols].sum(axis=1)
    if (row_sums > 2).any():
        # Not in [0,1] — normalize
        lc[lc_class_cols] = lc[lc_class_cols].div(row_sums, axis=0)

    # # Landscape typology
    # lc["landscape"] = "other"
    # lc.loc[lc["Cropland"]  >= AG_CROPLAND_THRESHOLD, "landscape"] = "agricultural"
    # lc.loc[lc["Built-up"]  >= URBAN_BUILT_THRESHOLD, "landscape"] = "urban"
    
    lc["landscape"] = "other"
    is_ag_dominant = (lc["Cropland"] >= 0.30) & (lc["Built-up"] < 0.15)
    is_urban_dominant = (lc["Built-up"] >= 0.30) & (lc["Cropland"] < 0.15)
    lc.loc[is_ag_dominant, "landscape"] = "agricultural"
    lc.loc[is_urban_dominant, "landscape"] = "urban"
        

    print("Landscape category counts:")
    print(lc["landscape"].value_counts().to_string())
    print()

    return preds, lc


def to_long(preds):
    """
    Reshape the wide preds table (one row per tract, columns like
    q3_2017_pred / q3_2017_error) into a long-format DataFrame with
    one row per (tract, quarter) plus columns pred and error.
    """
    id_cols = [c for c in preds.columns if not re.match(r"q[1-4]_\d{4}_", c)]
    value_cols = [c for c in preds.columns if re.match(r"q[1-4]_\d{4}_", c)]

    long = preds.melt(
        id_vars=id_cols,
        value_vars=value_cols,
        var_name="period",
        value_name="value",
    )

    # Split "q3_2017_pred" -> quarter=3, year=2017, kind='pred'
    parsed = long["period"].str.extract(r"q(?P<quarter>[1-4])_(?P<year>\d{4})_(?P<kind>pred|error)")
    long = pd.concat([long.drop(columns=["period"]), parsed], axis=1)
    long["quarter"] = long["quarter"].astype(int)
    long["year"] = long["year"].astype(int)

    # Pivot pred/error into their own columns
    long = long.pivot_table(
        index=id_cols + ["year", "quarter"],
        columns="kind",
        values="value",
    ).reset_index()
    long.columns.name = None

    # truth = pred - error  (since error = pred - truth)
    long["truth"] = long["pred"] - long["error"]

    # Continuous time index (0 for earliest quarter, increasing by 1)
    long["t"] = (long["year"] - long["year"].min()) * 4 + (long["quarter"] - 1)
    long["t"] = long["t"] - long["t"].min()

    return long


# -------------------------------------------------------------------------
# Step 2 — per-quarter R² and RMSE by landscape group
# -------------------------------------------------------------------------
def per_quarter_metrics(panel):
    """Compute R² and RMSE per (landscape, year, quarter)."""
    def _metrics(g):
        g = g.dropna(subset=["pred", "truth"])
        n = len(g)
        if n < 2:
            return pd.Series({"n": n, "r2": np.nan, "rmse": np.nan})
        return pd.Series({
            "n": n,
            "r2": r2_score(g["truth"], g["pred"]),
            "rmse": float(np.sqrt(np.mean((g["truth"] - g["pred"]) ** 2))),
        })

    out = (panel.groupby(["landscape", "year", "quarter"])
                .apply(_metrics)
                .reset_index())
    out["t"] = (out["year"] - out["year"].min()) * 4 + (out["quarter"] - 1)
    out["reliable"] = out["n"] >= MIN_CELL_N
    return out.sort_values(["landscape", "t"]).reset_index(drop=True)


# -------------------------------------------------------------------------
# Step 3 — seasonality: is within-year amplitude larger in ag than urban?
# -------------------------------------------------------------------------
def seasonal_amplitude_test(per_q):
    """
    For each (landscape, year), compute the standard deviation of the
    quarterly R² values. Then paired-test whether ag amplitude > urban
    amplitude across years.
    """
    amp = (per_q[per_q["reliable"]]
           .groupby(["landscape", "year"])["r2"]
           .agg(["std", "count"])
           .reset_index()
           .rename(columns={"std": "seasonal_amplitude", "count": "n_quarters"}))
    # Only keep years with all 4 quarters present for a fair amplitude
    amp = amp[amp["n_quarters"] == 4].copy()

    print("Seasonal amplitude per (landscape, year):")
    print(amp.to_string(index=False))
    print()

    ag = amp[amp["landscape"] == "agricultural"].set_index("year")["seasonal_amplitude"]
    ur = amp[amp["landscape"] == "urban"].set_index("year")["seasonal_amplitude"]
    common_years = ag.index.intersection(ur.index)

    if len(common_years) < 2:
        print("Not enough overlapping years for a paired test.")
        return amp, None

    ag = ag.loc[common_years]
    ur = ur.loc[common_years]
    diffs = ag - ur

    print(f"Paired amplitude differences (ag − urban), by year:")
    for y, d in diffs.items():
        print(f"  {y}: {d:+.4f}")

    # One-sided Wilcoxon: is ag amplitude > urban amplitude?
    if len(common_years) >= 3:
        try:
            stat, p = stats.wilcoxon(ag, ur, alternative="greater")
            print(f"\nWilcoxon signed-rank (ag > urban amplitude): stat={stat:.3f}, p={p:.4f}")
        except ValueError as e:
            print(f"Wilcoxon failed: {e}")
    # And a paired t-test as a complement
    t, p_t = stats.ttest_rel(ag, ur, alternative="greater")
    print(f"Paired t-test (ag > urban amplitude):        t={t:.3f}, p={p_t:.4f}")

    return amp, diffs


# -------------------------------------------------------------------------
# Step 4 — temporal lag: is R² improvement over time steeper in one group?
# -------------------------------------------------------------------------
def temporal_slope_test(per_q):
    """
    Fit R² ~ t (linear time index) separately for each landscape group.
    Report and compare slopes.
    """
    print("Linear R²-vs-time slopes per landscape group:")
    results = {}
    for landscape in ["agricultural", "urban"]:
        sub = per_q[(per_q["landscape"] == landscape) & per_q["reliable"]].copy()
        if len(sub) < 3:
            print(f"  {landscape}: too few reliable quarters ({len(sub)})")
            continue
        res = stats.linregress(sub["t"], sub["r2"])
        print(f"  {landscape}: slope={res.slope:+.5f}/quarter, "
              f"intercept={res.intercept:.3f}, p={res.pvalue:.4f}, "
              f"R²={res.rvalue**2:.3f}")
        results[landscape] = res
    return results


# -------------------------------------------------------------------------
# Step 5 — mixed models on tract-level data
# -------------------------------------------------------------------------
def mixed_models(panel):
    """
    Fit two mixed models on the tract-level squared-error panel:

    Model 1 (seasonality × landscape):
        sq_err ~ C(quarter) * cropland_frac + (1 | GEOID)
        Interaction quarter × cropland tests whether the effect of season
        depends on agricultural intensity.

    Model 2 (temporal lag × landscape):
        sq_err ~ t * cropland_frac + (1 | GEOID)
        Interaction t × cropland tests whether the temporal-trend effect
        depends on agricultural intensity.
    """
    d = panel.dropna(subset=["pred", "truth", "Cropland"]).copy()
    d["sq_err"] = (d["pred"] - d["truth"]) ** 2
    # Center cropland fraction so the intercept is interpretable and
    # collinearity between intercept and the moderator is reduced.
    d["cropland_c"] = d["Cropland"] - d["Cropland"].mean()

    def _try_fit(formula, label):
        print("\n" + "=" * 70)
        print(f"{label}: {formula}")
        print("=" * 70)
        try:
            m = smf.mixedlm(formula, data=d, groups=d["GEOID"]).fit(method="lbfgs")
            print(m.summary())
            return m
        except np.linalg.LinAlgError as e:
            print(f"Mixed model failed ({e}); falling back to OLS with cluster-robust SEs.")
            import statsmodels.api as sm
            ols_formula = formula
            m = smf.ols(ols_formula, data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d["GEOID"]}
            )
            print(m.summary())
            return m

    m1 = _try_fit(
        "sq_err ~ C(quarter) * cropland_c",
        "MODEL 1 (seasonality × landscape): sq_err ~ C(quarter) * cropland_c + (1|GEOID)",
    )
    m2 = _try_fit(
        "sq_err ~ t * cropland_c",
        "MODEL 2 (temporal lag × landscape): sq_err ~ t * cropland_c + (1|GEOID)",
    )
    return m1, m2


# -------------------------------------------------------------------------
# Step 6 — plots
# -------------------------------------------------------------------------
def plot_r2_curves(per_q, out_path):
    fig, ax = plt.subplots(figsize=(11, 5))
    for landscape, color in [("agricultural", "tab:green"),
                              ("urban",        "tab:red"),
                              ("other",        "tab:gray")]:
        sub = per_q[per_q["landscape"] == landscape].sort_values("t")
        ax.plot(sub["t"], sub["r2"], "-o", label=landscape, color=color, alpha=0.9)
        # Mark unreliable cells
        un = sub[~sub["reliable"]]
        if len(un):
            ax.plot(un["t"], un["r2"], "x", color=color, markersize=10, mew=2)

    # Tick labels: q_year for each t value present
    ticks = per_q[["t", "year", "quarter"]].drop_duplicates().sort_values("t")
    ax.set_xticks(ticks["t"])
    ax.set_xticklabels([f"q{q}_{y}" for y, q in zip(ticks["year"], ticks["quarter"])],
                       rotation=45, ha="right")
    ax.set_ylabel("R² (predicting ACS wealth)")
    ax.set_xlabel("Quarter")
    ax.set_title("Per-quarter R² by landscape type\n(× = fewer than "
                 f"{MIN_CELL_N} tracts, treat cautiously)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    preds, lc = load_and_merge(PREDS_CSV, LC_CSV)
    long = to_long(preds)

    # In the script, right after building 'long':
    tracts_all_quarters = long.dropna(subset=["pred"]).groupby("GEOID").size()
    complete_tracts = tracts_all_quarters[tracts_all_quarters == 15].index
    long = long[long["GEOID"].isin(complete_tracts)]
    
    panel = long.merge(lc, on="GEOID", how="inner")
    print(f"Panel size: {len(panel):,} rows "
          f"({panel['GEOID'].nunique():,} tracts × "
          f"{panel[['year','quarter']].drop_duplicates().shape[0]} quarters)")

    panel.to_csv(OUT_DIR / "panel_long.csv", index=False)

    per_q = per_quarter_metrics(panel)
    per_q.to_csv(OUT_DIR / "per_quarter_metrics.csv", index=False)
    print("\nPer-quarter R² summary (first 12 rows):")
    print(per_q.head(12).to_string(index=False))

    plot_r2_curves(per_q, OUT_DIR / "r2_curves_by_landscape.png")

    print("\n" + "=" * 70)
    print("TEST A — SEASONAL AMPLITUDE (ag vs urban)")
    print("=" * 70)
    amp, diffs = seasonal_amplitude_test(per_q)
    amp.to_csv(OUT_DIR / "seasonal_amplitude.csv", index=False)

    print("\n" + "=" * 70)
    print("TEST B — TEMPORAL SLOPE (ag vs urban)")
    print("=" * 70)
    temporal_slope_test(per_q)

    mixed_models(panel)


if __name__ == "__main__":
    main()
