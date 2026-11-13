"""
Figures + the season-vs-year test for band_importance.py's output: does a
model's band-group reliance profile cluster by SEASON (same quarter,
different year) more than it does by YEAR (same year, different quarter)?
That's the direct test of whether the imagery->wealth function's physical
basis (which bands it leans on) is organized by season, the way build_
cross_quarter_r2_matrix.py/analyze_cross_quarter_significance.py test
whether ACCURACY is.

Reads <out-dir>/<state>_<variable>_band_importance_summary.csv (written
by band_importance.py) -- never the raw long-format CSV, since every
figure/test here uses the per-(model, eval, method, unit, feature)
MEAN across repeats (and, for group importance, the normalized_
importance column), not individual repeat rows.

Figures (under --fig-dir):
    a) <state>_heatmap.png           rows=model quarters, cols=12 bands,
                                      cells=normalized permutation
                                      importance (diagonal only)
    b) <state>_by_quarter.png        one line per band group, x=quarter,
                                      y=mean normalized group importance
                                      across --mean-years, error bars=SD
                                      across those years (diagonal only)
    c) all_states_by_quarter.png     (b) as small multiples, one subplot
                                      per state, shared y-axis

Season-vs-year test (per state, saved to season_vs_year_test.csv and
printed): cosine DISTANCE (1 - cosine similarity) between every pair of
model-quarters' normalized grouped-importance vectors (5-dim: one entry
per band group, group_permutation method, diagonal cells only). Compares
mean distance for SAME-season/different-year pairs against DIFFERENT-
season/same-year pairs -- if the seasonal-function hypothesis holds,
same-season pairs should be closer (lower distance) than same-year,
different-season pairs. Null: permute which quarter label each year's
models carry, independently per year ("shuffle season labels within each
year"), recompute the same statistic, 5000 draws. Two-sided bootstrap
p-value. Run once with every available year, once excluding 2016 (2016
is often a partial/first-year-of-imagery artifact elsewhere in this
project -- same reason analyze_seasonality_lag.py and others break out a
with/without-2016 comparison).

A state/quarter/year missing from the summary CSV (not every cell is
necessarily trained) is simply absent from these figures/tests -- no
special handling needed beyond "operate on whatever rows exist".

Usage:
    python 3_analysis/plot_band_importance.py --states oh az pa ga ca \
        --variable wealth_index
"""

import argparse
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

# Duplicated from band_importance.py (not imported) so this plotting
# script doesn't need sail/torch installed -- band_importance.py runs on
# the GPU cluster, this is meant to also run on a plain laptop against
# its output CSVs. Keep these two in sync with band_importance.py by hand.
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B11", "B12"]
BAND_GROUPS = {
    "visible": ["B02", "B03", "B04"],
    "red_edge": ["B05", "B06", "B07"],
    "nir": ["B08", "B8A"],
    "swir": ["B11", "B12"],
    "atmospheric": ["B01", "B09"],
}

GROUP_ORDER = list(BAND_GROUPS)  # visible, red_edge, nir, swir, atmospheric

# Dataviz-skill-validated categorical palette, slots 1-5 (one per band
# group) -- same palette used elsewhere in this project (e.g.
# build_cross_state_r2_matrix.py's GROUP_PALETTE) for a consistent look.
GROUP_STYLE = {
    "visible": {"color": "#2a78d6", "marker": "o"},
    "red_edge": {"color": "#eb6834", "marker": "s"},
    "nir": {"color": "#1baf7a", "marker": "^"},
    "swir": {"color": "#eda100", "marker": "D"},
    "atmospheric": {"color": "#e87ba4", "marker": "v"},
}


def load_summary(out_dir: Path, state: str, variable: str) -> "pd.DataFrame | None":
    path = out_dir / f"{state}_{variable}_band_importance_summary.csv"
    if not path.exists():
        print(f"[{state.upper()}] no summary CSV at {path}, skipping")
        return None
    return pd.read_csv(path)


# -------------------------------------------------------------------------
# (a) heatmap: model-quarters x bands, normalized permutation importance
# -------------------------------------------------------------------------
def plot_heatmap(summary: pd.DataFrame, state: str, variable: str, out_path: Path):
    diag = summary[(summary.method == "permutation") & (summary.unit == "band")
                   & (summary.eval_year == summary.model_year) & (summary.eval_quarter == summary.model_quarter)]
    if diag.empty:
        print(f"[{state.upper()}] no diagonal permutation rows -- skipping heatmap")
        return

    model_quarters = sorted(diag[["model_year", "model_quarter"]].drop_duplicates().itertuples(index=False, name=None))
    labels = [f"Q{q} {y}" for y, q in model_quarters]
    grid = pd.DataFrame(np.nan, index=labels, columns=BAND_NAMES)
    for (y, q), label in zip(model_quarters, labels):
        for band in BAND_NAMES:
            row = diag[(diag.model_year == y) & (diag.model_quarter == q) & (diag.feature == band)]
            if len(row):
                grid.loc[label, band] = row["normalized_importance"].iloc[0]

    finite = grid.values[np.isfinite(grid.values)]
    if finite.size == 0:
        print(f"[{state.upper()}] no finite normalized_importance values -- skipping heatmap")
        return
    vmin, vmax = float(finite.min()), float(finite.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(BAND_NAMES)), max(6, 0.35 * len(labels))))
    im = ax.imshow(grid.values, cmap="RdBu", norm=norm,
                    vmin=None if norm else vmin, vmax=None if norm else vmax, aspect="auto")
    ax.set_xticks(range(len(BAND_NAMES)))
    ax.set_xticklabels(BAND_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title(f"{state.upper()} {variable}: normalized permutation importance by band\n"
                 f"(diagonal: each model on its own held-out test set)")
    fig.colorbar(im, ax=ax, label="Normalized importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


# -------------------------------------------------------------------------
# (b)/(c) grouped importance by quarter-of-year, mean +/- SD across years
# -------------------------------------------------------------------------
def grouped_importance_by_quarter(summary: pd.DataFrame, mean_years: list) -> pd.DataFrame:
    diag = summary[(summary.method == "group_permutation") & (summary.unit == "group")
                   & (summary.eval_year == summary.model_year) & (summary.eval_quarter == summary.model_quarter)
                   & (summary.model_year.isin(mean_years))]
    rows = []
    for group_name in GROUP_ORDER:
        g = diag[diag.feature == group_name]
        for quarter in [1, 2, 3, 4]:
            vals = g.loc[g.model_quarter == quarter, "normalized_importance"].dropna()
            rows.append({"group": group_name, "quarter": quarter,
                         "mean": vals.mean() if len(vals) else np.nan,
                         "std": vals.std(ddof=1) if len(vals) > 1 else 0.0,
                         "n_years": len(vals)})
    return pd.DataFrame(rows)


def _plot_groups_on_ax(ax, agg: pd.DataFrame, title: str):
    for group_name in GROUP_ORDER:
        g = agg[agg.group == group_name].sort_values("quarter")
        if g["mean"].notna().sum() == 0:
            continue
        style = GROUP_STYLE[group_name]
        ax.errorbar(g["quarter"], g["mean"], yerr=g["std"], fmt=f"-{style['marker']}",
                    color=style["color"], label=group_name, capsize=3, linewidth=1.8, markersize=6)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Quarter")
    ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.3)


def plot_by_quarter(summary: pd.DataFrame, state: str, variable: str, mean_years: list, out_path: Path):
    agg = grouped_importance_by_quarter(summary, mean_years)
    if agg["mean"].notna().sum() == 0:
        print(f"[{state.upper()}] no diagonal group_permutation rows for years {mean_years} -- skipping")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    _plot_groups_on_ax(ax, agg, f"{state.upper()} {variable}\nmean normalized group importance, "
                                 f"years {min(mean_years)}-{max(mean_years)}")
    ax.set_ylabel("Normalized importance")
    ax.legend(title="Band group", frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_by_quarter_all_states(summaries: dict, variable: str, mean_years: list, out_path: Path, ncols: int = 3):
    states = list(summaries)
    n = len(states)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()

    any_data = False
    for ax, state in zip(axes, states):
        agg = grouped_importance_by_quarter(summaries[state], mean_years)
        if agg["mean"].notna().sum() == 0:
            ax.set_visible(False)
            continue
        any_data = True
        _plot_groups_on_ax(ax, agg, state.upper())
    for ax in axes[n:]:
        ax.set_visible(False)
    if not any_data:
        print("No state had usable diagonal group_permutation rows -- skipping small multiples")
        plt.close(fig)
        return

    axes[0].set_ylabel("Normalized importance")
    handles = [plt.Line2D([0], [0], color=GROUP_STYLE[g]["color"], marker=GROUP_STYLE[g]["marker"], label=g)
               for g in GROUP_ORDER]
    fig.legend(handles=handles, title="Band group", loc="upper center", ncol=len(GROUP_ORDER),
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(f"{variable}: mean normalized group importance by quarter, years "
                 f"{min(mean_years)}-{max(mean_years)}", y=1.08)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


# -------------------------------------------------------------------------
# Season-vs-year cosine-distance permutation test
# -------------------------------------------------------------------------
def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu == 0 or nv == 0:
        return float("nan")
    return float(1.0 - np.dot(u, v) / (nu * nv))


def _season_year_stat(years: np.ndarray, quarters: np.ndarray, dist: np.ndarray) -> "tuple[float, float, float]":
    """Returns (mean_same_season_diff_year, mean_diff_season_same_year,
    their difference). `dist` is the full pairwise distance matrix;
    `quarters`/`years` are the CURRENT label assignment (real or
    permuted)."""
    n = len(years)
    iu = np.triu_indices(n, k=1)
    same_season = quarters[iu[0]] == quarters[iu[1]]
    same_year = years[iu[0]] == years[iu[1]]
    pair_dist = dist[iu]

    ssdy = pair_dist[same_season & ~same_year]   # same season, different year
    dssy = pair_dist[~same_season & same_year]   # different season, same year
    m_ssdy = ssdy.mean() if len(ssdy) else np.nan
    m_dssy = dssy.mean() if len(dssy) else np.nan
    return m_ssdy, m_dssy, m_ssdy - m_dssy


def season_vs_year_test(summary: pd.DataFrame, state: str, years_filter: "list | None",
                         n_permutations: int, seed: int) -> dict:
    diag = summary[(summary.method == "group_permutation") & (summary.unit == "group")
                   & (summary.eval_year == summary.model_year) & (summary.eval_quarter == summary.model_quarter)]
    if years_filter is not None:
        diag = diag[diag.model_year.isin(years_filter)]

    model_quarters = sorted(diag[["model_year", "model_quarter"]].drop_duplicates().itertuples(index=False, name=None))
    vectors, years, quarters = [], [], []
    for y, q in model_quarters:
        row = diag[(diag.model_year == y) & (diag.model_quarter == q)]
        vec = np.array([row.loc[row.feature == g, "normalized_importance"].iloc[0]
                        if (row.feature == g).any() else np.nan for g in GROUP_ORDER])
        if np.isnan(vec).any():
            continue
        vectors.append(vec)
        years.append(y)
        quarters.append(q)

    n = len(vectors)
    result = {"state": state, "years_included": "all" if years_filter is None else "excl_2016",
              "n_models": n, "n_permutations": n_permutations}
    if n < 4:
        print(f"[{state.upper()}] only {n} usable model-quarter(s) ({result['years_included']}) -- "
              f"too few for the season-vs-year test, skipping")
        result.update({"mean_same_season_diff_year": np.nan, "mean_diff_season_same_year": np.nan,
                        "observed_stat": np.nan, "p_value": np.nan})
        return result

    vectors = np.array(vectors)
    years = np.array(years)
    quarters = np.array(quarters)
    n_bands_vec = vectors.shape[1]
    dist = np.zeros((n, n))
    for i, j in itertools.combinations(range(n), 2):
        d = cosine_distance(vectors[i], vectors[j])
        dist[i, j] = dist[j, i] = d

    m_ssdy, m_dssy, observed = _season_year_stat(years, quarters, dist)

    rng = np.random.default_rng(seed)
    null = np.full(n_permutations, np.nan)
    unique_years = np.unique(years)
    for p in range(n_permutations):
        perm_quarters = quarters.copy()
        for y in unique_years:
            mask = years == y
            perm_quarters[mask] = rng.permutation(quarters[mask])
        _, _, stat = _season_year_stat(years, perm_quarters, dist)
        null[p] = stat

    valid_null = null[~np.isnan(null)]
    if len(valid_null) == 0 or np.isnan(observed):
        p_value = float("nan")
    else:
        p_left = float(np.mean(valid_null <= observed))
        p_right = float(np.mean(valid_null >= observed))
        p_value = min(1.0, 2 * min(p_left, p_right))

    result.update({"mean_same_season_diff_year": m_ssdy, "mean_diff_season_same_year": m_dssy,
                    "observed_stat": observed, "p_value": p_value, "n_valid_permutations": len(valid_null)})
    print(f"[{state.upper()}] ({result['years_included']}, n={n} models) "
          f"mean same-season/diff-year dist={m_ssdy:.4f}, mean diff-season/same-year dist={m_dssy:.4f}, "
          f"observed diff={observed:+.4f}, p={p_value:.4g}")
    if observed < 0 and p_value < 0.05:
        print(f"    -> same-season pairs are CLOSER than same-year pairs, significantly "
              f"(p<0.05): supports a season-organized band-importance profile.")
    elif p_value < 0.05:
        print(f"    -> significant, but in the OPPOSITE direction from the seasonal-function "
              f"hypothesis (same-year pairs closer than same-season pairs).")
    else:
        print(f"    -> not significant at p<0.05.")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--out-dir", default="3_analysis/outputs/band_importance",
                    help="Where band_importance.py wrote its summary CSVs")
    p.add_argument("--fig-dir", default="3_analysis/figures/band_importance")
    p.add_argument("--mean-years", nargs="+", type=int, default=[2017, 2018, 2019],
                    help="Years averaged over for figures (b)/(c) (default: 2017 2018 2019)")
    p.add_argument("--n-permutations", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for state in args.states:
        state = state.lower()
        summary = load_summary(out_dir, state, args.variable)
        if summary is not None:
            summaries[state] = summary

    if not summaries:
        raise SystemExit("No state had a usable summary CSV -- run band_importance.py first.")

    for state, summary in summaries.items():
        plot_heatmap(summary, state, args.variable, fig_dir / f"{state}_heatmap.png")
        plot_by_quarter(summary, state, args.variable, args.mean_years, fig_dir / f"{state}_by_quarter.png")

    plot_by_quarter_all_states(summaries, args.variable, args.mean_years,
                                fig_dir / "all_states_by_quarter.png")

    print(f"\n{'=' * 70}\nSeason-vs-year test\n{'=' * 70}")
    rows = []
    for state, summary in summaries.items():
        all_years = sorted(summary.model_year.unique())
        rows.append(season_vs_year_test(summary, state, None, args.n_permutations, args.seed))
        if 2016 in all_years:
            rows.append(season_vs_year_test(summary, state, [y for y in all_years if y != 2016],
                                              args.n_permutations, args.seed))
    results = pd.DataFrame(rows)
    results_path = out_dir / "season_vs_year_test.csv"
    results.to_csv(results_path, index=False)
    print(f"\nWrote {results_path}")


if __name__ == "__main__":
    main()
