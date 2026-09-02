"""
Per-tract band importance: does it moderate by urban/rural landscape, and
is it spatially autocorrelated (does a band matter in geographic
clusters, or is its importance scattered randomly across tracts)?

Input is one quarter's band_importance_per_image.csv, produced by sail's
`task: band_importance` with `band_importance.per_image: true` (see
sail/src/sail/explain/band_importance.py). One row per tract:
GEOID, lat, lon, label, pred_baseline, then delta_pred_<band>/
delta_sqerr_<band> for every band (and every group, if configured).
delta_sqerr is the one these tests use: positive = removing that band's
information made THIS tract's prediction worse (i.e. the model was
relying on it here), negative = made it better (the model was misled by
it here).

Two tests:

  H3a  Landscape moderation: does mean delta_sqerr for a band differ
       between agricultural and urban tracts? (Same ag/urban typology as
       analyze_seasonality_lag.py -- Cropland/Built-up thresholds on the
       lc.csv land-cover fractions.)
  H3b  Spatial autocorrelation: Moran's I on each band's per-tract
       delta_sqerr, using a k-NN spatial weight matrix built from lat/lon
       and a permutation-based (not the closed-form normal-approximation)
       significance test. I near +1 = the band's importance clusters
       geographically (e.g. matters across a contiguous urban core, not
       just scattered high-wealth tracts); I near 0 = no spatial
       structure; I near -1 = neighboring tracts alternate high/low
       (checkerboard) importance.

Outputs (under OUT_DIR):
    per_image_with_landscape.csv   the input joined to land cover
    h3a_landscape_moderation.csv   per-band ag vs urban comparison
    h3b_morans_i.csv               per-band Moran's I + p-value
    landscape_boxplot.png          delta_sqerr distribution by landscape, per highlighted band
    morans_i_bar.png               Moran's I per band, significance marked
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


# Same thresholds as analyze_seasonality_lag.py -- keep these in sync if
# you change one.
AG_CROPLAND_THRESHOLD = 0.30
URBAN_BUILT_THRESHOLD = 0.30

# Default bands to test and plot when --band-columns isn't given -- assumes
# this quarter's CSV has RGB+NIR+SWIR columns (band_importance.band_names
# in the sail config that produced it). Override with --band-columns for a
# different band set.
DEFAULT_BAND_COLUMNS = ["B02", "B03", "B04", "B08", "B11", "B12"]

# k-NN neighborhood size for the spatial weight matrix, and permutation
# count for Moran's I's significance test.
KNN_K = 8
MORAN_N_PERM = 999
SEED = 1337


# -------------------------------------------------------------------------
# Step 1 -- load + join to land cover
# -------------------------------------------------------------------------
def load_and_join(per_image_csv, lc_csv):
    df = pd.read_csv(per_image_csv, dtype={"GEOID": str})

    lc = pd.read_csv(lc_csv, dtype={0: str}, index_col=0)
    lc.index.name = "GEOID"
    lc = lc.reset_index()
    lc_class_cols = [c for c in lc.columns if c != "GEOID"]
    row_sums = lc[lc_class_cols].sum(axis=1)
    if (row_sums > 2).any():
        lc[lc_class_cols] = lc[lc_class_cols].div(row_sums, axis=0)

    lc["landscape"] = "other"
    is_ag = (lc["Cropland"] >= AG_CROPLAND_THRESHOLD) & (lc["Built-up"] < 0.15)
    is_urban = (lc["Built-up"] >= URBAN_BUILT_THRESHOLD) & (lc["Cropland"] < 0.15)
    lc.loc[is_ag, "landscape"] = "agricultural"
    lc.loc[is_urban, "landscape"] = "urban"

    merged = df.merge(lc[["GEOID", "landscape"]], on="GEOID", how="inner")
    print(f"Joined {len(merged):,} / {len(df):,} tracts to land cover.")
    print(merged["landscape"].value_counts().to_string())
    return merged


# -------------------------------------------------------------------------
# H3a -- landscape moderation
# -------------------------------------------------------------------------
def h3a_landscape_moderation(panel, band_columns, out_dir):
    print("\n" + "=" * 70)
    print("H3a -- landscape moderation (agricultural vs urban)")
    print("=" * 70)

    rows = []
    for band in band_columns:
        col = f"delta_sqerr_{band}"
        if col not in panel.columns:
            print(f"  Skipping {band}: no column {col}")
            continue
        ag = panel.loc[panel["landscape"] == "agricultural", col].dropna()
        ur = panel.loc[panel["landscape"] == "urban", col].dropna()
        row = {
            "band": band, "n_ag": len(ag), "n_urban": len(ur),
            "mean_ag": ag.mean() if len(ag) else np.nan,
            "mean_urban": ur.mean() if len(ur) else np.nan,
        }
        row["diff_ag_minus_urban"] = row["mean_ag"] - row["mean_urban"]
        if len(ag) >= 2 and len(ur) >= 2:
            t, p = stats.ttest_ind(ag, ur, equal_var=False)
            row["t_stat"], row["p_value"] = t, p
        else:
            row["t_stat"], row["p_value"] = np.nan, np.nan
        rows.append(row)
        print(f"  {band}: mean_ag={row['mean_ag']:+.5f} mean_urban={row['mean_urban']:+.5f} "
              f"diff={row['diff_ag_minus_urban']:+.5f} p={row['p_value']:.4g}")

    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "h3a_landscape_moderation.csv", index=False)

    fig, axes = plt.subplots(1, len(band_columns), figsize=(3.2 * len(band_columns), 5), sharey=False)
    if len(band_columns) == 1:
        axes = [axes]
    for ax, band in zip(axes, band_columns):
        col = f"delta_sqerr_{band}"
        if col not in panel.columns:
            continue
        data = [panel.loc[panel["landscape"] == lc, col].dropna()
                for lc in ["agricultural", "urban", "other"]]
        ax.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax.boxplot(data, labels=["ag", "urban", "other"], showfliers=False)
        ax.set_title(band)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("delta_sqerr (worse when band removed →)")
    plt.suptitle("Per-tract band importance by landscape type")
    plt.tight_layout()
    plt.savefig(out_dir / "landscape_boxplot.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'landscape_boxplot.png'}")

    return summary


# -------------------------------------------------------------------------
# H3b -- spatial autocorrelation (Moran's I)
# -------------------------------------------------------------------------
def _build_knn_weights(lat, lon, k):
    """
    Row-standardized k-NN spatial weight matrix (sparse), from lat/lon in
    degrees. Longitude scaled by cos(mean latitude) as a simple planar
    approximation -- fine at the scale of one state's census tracts, not
    a real geodesic distance.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    n = len(lat)
    k = min(k, n - 1)
    x = lon * np.cos(np.radians(lat.mean()))
    coords = np.column_stack([x, lat])
    tree = cKDTree(coords)
    _, idx = tree.query(coords, k=k + 1)  # +1: includes self at distance 0

    rows, cols, data = [], [], []
    for i in range(n):
        neighbors = idx[i, 1:]  # drop self
        rows.extend([i] * len(neighbors))
        cols.extend(neighbors.tolist())
        data.extend([1.0 / len(neighbors)] * len(neighbors))
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def morans_i(values, lat, lon, k=8, n_perm=999, seed=1337):
    """
    Moran's I with a permutation-based (not closed-form normal-
    approximation) two-sided p-value: shuffle `values` across the fixed
    tract locations n_perm times, p = fraction of permuted |I| >= |I_obs|.

    I near +1: this band's importance clusters spatially (e.g. matters
        across a contiguous region, not scattered high/low tracts).
    I near 0: no detectable spatial structure.
    I near -1: neighboring tracts alternate high/low importance.
    """
    values = np.asarray(values, dtype=float)
    mask = np.isfinite(values)
    if mask.sum() < max(k + 1, 4):
        return {"I": np.nan, "p_value": np.nan, "n": int(mask.sum())}

    values = values[mask]
    lat = np.asarray(lat)[mask]
    lon = np.asarray(lon)[mask]
    n = len(values)

    W = _build_knn_weights(lat, lon, k)
    W_sum = W.sum()

    def _I(xv):
        xc = xv - xv.mean()
        d = np.sum(xc ** 2)
        if d == 0:
            return np.nan
        return (n / W_sum) * (xc @ (W @ xc)) / d

    I_obs = _I(values)

    rng = np.random.RandomState(seed)
    I_perm = np.array([_I(rng.permutation(values)) for _ in range(n_perm)])
    I_perm = I_perm[np.isfinite(I_perm)]
    if len(I_perm) == 0 or not np.isfinite(I_obs):
        return {"I": I_obs, "p_value": np.nan, "n": n}

    p_value = (np.sum(np.abs(I_perm) >= abs(I_obs)) + 1) / (len(I_perm) + 1)
    return {"I": I_obs, "p_value": p_value, "n": n,
            "I_perm_mean": I_perm.mean(), "I_perm_std": I_perm.std()}


def h3b_spatial_autocorrelation(panel, band_columns, out_dir, k=KNN_K, n_perm=MORAN_N_PERM, seed=SEED):
    print("\n" + "=" * 70)
    print(f"H3b -- spatial autocorrelation (Moran's I, k={k} neighbors, {n_perm} permutations)")
    print("=" * 70)

    rows = []
    for band in band_columns:
        col = f"delta_sqerr_{band}"
        if col not in panel.columns:
            print(f"  Skipping {band}: no column {col}")
            continue
        res = morans_i(panel[col].values, panel["lat"].values, panel["lon"].values,
                        k=k, n_perm=n_perm, seed=seed)
        res["band"] = band
        rows.append(res)
        sig = "*" if (not np.isnan(res.get("p_value", np.nan)) and res["p_value"] < 0.05) else " "
        print(f"  {band}: I={res['I']:+.4f}{sig} p={res.get('p_value', float('nan')):.4g} n={res['n']}")

    summary = pd.DataFrame(rows)[["band", "I", "p_value", "n"]]
    summary.to_csv(out_dir / "h3b_morans_i.csv", index=False)
    print("\n(* = p < 0.05. Bonferroni-correct across bands if testing many "
          "at once -- with len(band_columns) tests here, treat borderline "
          "p-values accordingly.)")

    fig, ax = plt.subplots(figsize=(max(6, len(summary) * 1.1), 5))
    colors = ["tab:blue" if p < 0.05 else "lightgray" for p in summary["p_value"].fillna(1.0)]
    ax.bar(summary["band"], summary["I"], color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Moran's I")
    ax.set_title("Spatial autocorrelation of per-tract band importance\n"
                 "(blue = p<0.05 by permutation test, gray = not significant)")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_dir / "morans_i_bar.png", dpi=150)
    plt.close()
    print(f"Saved {out_dir / 'morans_i_bar.png'}")

    return summary


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State label, e.g. oh/az/ca/ga/pa (used for output naming only)")
    p.add_argument("--per-image-csv", required=True, type=Path,
                    help="band_importance_per_image.csv from sail's task: band_importance")
    p.add_argument("--lc-csv", required=True, type=Path,
                    help="lc.csv (e.g. from compute_landcover.py)")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: ./out_band_importance_spatial/<state>)")
    p.add_argument("--band-columns", default=",".join(DEFAULT_BAND_COLUMNS),
                    help=f"Comma-separated band names to test (default: {','.join(DEFAULT_BAND_COLUMNS)})")
    args = p.parse_args()

    band_columns = args.band_columns.split(",")
    out_dir = args.out_dir or Path(f"./out_band_importance_spatial/{args.state.lower()}")
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = load_and_join(args.per_image_csv, args.lc_csv)
    panel.to_csv(out_dir / "per_image_with_landscape.csv", index=False)

    h3a_landscape_moderation(panel, band_columns, out_dir)
    h3b_spatial_autocorrelation(panel, band_columns, out_dir)


if __name__ == "__main__":
    main()
