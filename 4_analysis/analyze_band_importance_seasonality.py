"""
Does each Sentinel-2 band's contribution to the wealth prediction vary by
season -- e.g. does NIR matter more in summer, when vegetation-phenology
signal is actually present, than in winter?

Inputs are per-quarter band_importance.csv files, produced by sail's
`task: band_importance` (see sail/src/sail/explain/band_importance.py):
one row per band with columns band_index, band_name, baseline_mse,
delta_mse, delta_mse_std, pct_increase. pct_increase (not raw delta_mse)
is the number these tests use -- it's normalized by that quarter's own
baseline MSE, so a harder quarter overall doesn't look like "every band
mattered more" by itself.

Two tests, run per band:

  H2a  Is a band's importance itself seasonal: higher pct_increase in
       growing-season quarters than non-growing ones.
  H2b  Descriptive: pct_increase by quarter, one line per band, to spot
       patterns a pairwise test might miss (e.g. a band peaking mid-
       season rather than differing cleanly by growing/non-growing).

With only a handful of quarters trained so far, H2a's tests are going to
be very underpowered (this prints n alongside every p-value for exactly
that reason) -- treat early results as descriptive, not confirmatory,
until more quarters/years are in the panel.

Outputs (under OUT_DIR):
    band_importance_panel.csv        long-format panel across quarters
    band_importance_by_quarter.png   H2b
    h2a_seasonal_summary.csv         per-band growing vs non-growing test
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# -------------------------------------------------------------------------
# Config -- <<< fill these in
# -------------------------------------------------------------------------
QUARTERS = [
    # {"year": 2016, "quarter": 1,
    #  "csv": "/data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/artifacts/az_q1_2016_allbands_v1/band_importance.csv"},
    # ... one entry per quarter with a trained checkpoint + band_importance run,
    # or build with discover_quarters() below
]

OUT_DIR = Path("./out_band_importance")  # <<< change me

# <<< Same Ohio-corn-belt assumption (and same AZ caveat) as
# analyze_pixel_diversity_accuracy.py -- verify against real AZ NDVI/
# cropping-calendar data before trusting H2a here.
GROWING_QUARTERS = {2, 3}
NON_GROWING_QUARTERS = {1, 4}

# Bands worth calling out specifically in the plot (vegetation/moisture-
# sensitive ones the seasonality theory is actually about); everything
# else is drawn in gray so the plot doesn't turn into 12 overlapping lines.
HIGHLIGHT_BANDS = {"B08": "tab:green", "B8A": "tab:olive",
                    "B11": "tab:orange", "B12": "tab:red",
                    "B04": "tab:blue"}


def discover_quarters(artifacts_root, csv_name="band_importance.csv"):
    """
    Scan artifacts_root (a state's imagery root, e.g.
    ".../az_imagery/") for geoetl/sail tlag-style quarter+experiment
    directories and find each quarter's band_importance.csv.

    Expects the same layout used throughout this project:
      <artifacts_root>/q{1-4}_{year}_s2*/artifacts/*/band_importance.csv
    (the "_s2*" glob tolerates suffixes like "_allbands").
    """
    artifacts_root = Path(artifacts_root)
    quarters = []
    for qdir in sorted(artifacts_root.glob("q[1-4]_????_s2*")):
        m = re.match(r"q([1-4])_(\d{4})_s2", qdir.name)
        if not m:
            continue
        quarter, year = int(m.group(1)), int(m.group(2))

        matches = sorted(qdir.glob(f"artifacts/*/{csv_name}"))
        if len(matches) != 1:
            print(f"Skipping {qdir.name}: found {len(matches)} {csv_name} "
                  f"files (expected exactly 1) -- run task: band_importance first")
            continue

        quarters.append({"year": year, "quarter": quarter, "csv": str(matches[0])})

    print(f"discover_quarters: found {len(quarters)} usable quarters under {artifacts_root}")
    return quarters


# Set this to auto-populate QUARTERS instead of listing entries by hand.
ARTIFACTS_ROOT = None   # <<< e.g. "/data/hbaier/new_data/tlag/az_imagery"
if ARTIFACTS_ROOT:
    QUARTERS = discover_quarters(ARTIFACTS_ROOT)


# -------------------------------------------------------------------------
# Step 1 -- load + build the long panel
# -------------------------------------------------------------------------
def build_panel(quarters_cfg):
    parts = []
    for cfg in quarters_cfg:
        df = pd.read_csv(cfg["csv"])
        df["year"] = cfg["year"]
        df["quarter"] = cfg["quarter"]
        parts.append(df)
    panel = pd.concat(parts, ignore_index=True)

    panel["season"] = np.where(panel["quarter"].isin(GROWING_QUARTERS),
                                "growing", "non_growing")
    panel["t"] = (panel["year"] - panel["year"].min()) * 4 + (panel["quarter"] - 1)
    panel["t"] = panel["t"] - panel["t"].min()
    return panel


# -------------------------------------------------------------------------
# H2a -- is a band's importance itself seasonal?
# -------------------------------------------------------------------------
def h2a_seasonal_importance(panel, out_dir):
    print("\n" + "=" * 70)
    print("H2a -- per-band seasonality of pct_increase (growing vs non-growing)")
    print("=" * 70)

    rows = []
    for band_name, g in panel.groupby("band_name"):
        growing = g.loc[g["season"] == "growing", "pct_increase"].dropna()
        non_growing = g.loc[g["season"] == "non_growing", "pct_increase"].dropna()
        row = {
            "band_name": band_name,
            "n_growing": len(growing),
            "n_non_growing": len(non_growing),
            "mean_growing": growing.mean() if len(growing) else np.nan,
            "mean_non_growing": non_growing.mean() if len(non_growing) else np.nan,
        }
        row["diff_growing_minus_non_growing"] = row["mean_growing"] - row["mean_non_growing"]

        if len(growing) >= 2 and len(non_growing) >= 2:
            t, p = stats.ttest_ind(growing, non_growing, equal_var=False)
            row["t_stat"] = t
            row["p_value"] = p
        else:
            row["t_stat"] = np.nan
            row["p_value"] = np.nan

        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("diff_growing_minus_non_growing", ascending=False)
    summary.to_csv(out_dir / "h2a_seasonal_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\n(n_growing/n_non_growing this small means these p-values are "
          "descriptive at best -- more quarters/years needed for real power.)")
    return summary


# -------------------------------------------------------------------------
# H2b -- band importance by quarter, plotted
# -------------------------------------------------------------------------
def plot_band_importance_by_quarter(panel, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    for band_name, g in panel.groupby("band_name"):
        g = g.sort_values("t")
        if band_name in HIGHLIGHT_BANDS:
            ax.plot(g["t"], g["pct_increase"], "-o", label=band_name,
                    color=HIGHLIGHT_BANDS[band_name], linewidth=2, zorder=3)
        else:
            ax.plot(g["t"], g["pct_increase"], "-", color="gray", alpha=0.35,
                    linewidth=1, zorder=1)

    for _, r in panel[["t", "quarter"]].drop_duplicates().iterrows():
        color = "tab:green" if r["quarter"] in GROWING_QUARTERS else "tab:orange"
        ax.axvspan(r["t"] - 0.5, r["t"] + 0.5, color=color, alpha=0.06)

    ticks = panel[["t", "year", "quarter"]].drop_duplicates().sort_values("t")
    ax.set_xticks(ticks["t"])
    ax.set_xticklabels([f"q{q}_{y}" for y, q in zip(ticks["year"], ticks["quarter"])],
                       rotation=45, ha="right")
    ax.set_ylabel("% increase in MSE when band is permuted\n(higher = band matters more)")
    ax.set_xlabel("Quarter")
    ax.set_title("Per-quarter band importance\n"
                 "colored = vegetation/moisture-sensitive bands, gray = other bands\n"
                 "green background=growing season, orange=non-growing")
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
    if not QUARTERS:
        raise SystemExit(
            "QUARTERS is empty -- fill in {year, quarter, csv} entries at "
            "the top of this script (one per quarter with a completed "
            "task: band_importance run), or set ARTIFACTS_ROOT."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = build_panel(QUARTERS)
    panel.to_csv(OUT_DIR / "band_importance_panel.csv", index=False)
    print(f"Panel: {len(panel):,} band-quarter rows across "
          f"{panel[['year', 'quarter']].drop_duplicates().shape[0]} quarters, "
          f"{panel['band_name'].nunique()} bands")

    plot_band_importance_by_quarter(panel, OUT_DIR / "band_importance_by_quarter.png")
    h2a_seasonal_importance(panel, OUT_DIR)


if __name__ == "__main__":
    main()
