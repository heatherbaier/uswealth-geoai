"""
Paper-ready figures for the weight-drift RQ (do season-trained models
learn measurably different weights, and do those weights group by
season rather than year?), combining every state in --states into each
figure instead of one plot per state. Pure CSV-in, PNG-out -- reads
analyze_weight_drift.py's and analyze_weight_drift_by_layer.py's output
CSVs directly, no torch/sail/GPU needed.

Three figures:
  fig_season_vs_year_clustering.png    One panel. Grouped bar, one
      group per state: the season-clustering gap vs. the year-
      clustering gap (from <state>_<variable>_weight_clustering.csv),
      p-value labeled above each bar. The headline result.
  fig_weight_similarity_vs_transfer.png   Small multiples, one panel
      per state: weight-delta cosine similarity vs. transfer R^2
      (from <state>_<variable>_weight_similarity_long.csv), colored by
      same/different season, with a per-group trend line. Each panel
      annotated with its overall and season/year-partial Pearson r
      (from <state>_<variable>_weight_vs_transfer_corr.csv).
  fig_season_clustering_by_layer.png   One panel. Grouped bar, one
      group per layer (patch_embed -> stage1..4 -> final_norm -> head,
      from <state>_<variable>_weight_clustering_by_layer.csv): each
      state's season-clustering gap at that layer, bars grouped by
      layer so the where-does-it-concentrate pattern is directly
      comparable across states.

Usage:
    python 4_analysis/plot_weight_drift_figures.py --states az ga oh pa
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Must match analyze_weight_drift_by_layer.py's own ANATOMICAL_ORDER.
LAYER_ORDER = ["patch_embed", "stage1", "stage2", "stage3", "stage4", "final_norm", "head"]

# Fixed categorical colors, assigned in a FIXED order (never reassigned
# based on which states are actually present) -- validated colorblind-
# safe (scripts/validate_palette.js) for both the 4-state adjacent set
# and every 2-color pair used below.
STATE_COLORS = {"az": "#2a78d6", "ga": "#eb6834", "oh": "#1baf7a", "pa": "#eda100",
                "ca": "#e87ba4"}
SEASON_COLOR = "#1baf7a"   # same season / season-clustering gap
YEAR_COLOR = "#2a78d6"     # different season / year-clustering gap

BAR_EDGE = "#2b2b2b"  # thin dark edge -- the skill's "relief" requirement for
                       # the lighter palette slots (aqua/yellow) at low contrast


def state_color(state: str) -> str:
    return STATE_COLORS.get(state, "#52514e")


def load_state_data(state: str, variable: str, data_dir: Path):
    prefix = data_dir / f"{state}_{variable}"
    paths = {
        "clustering": Path(f"{prefix}_weight_clustering.csv"),
        "corr": Path(f"{prefix}_weight_vs_transfer_corr.csv"),
        "long": Path(f"{prefix}_weight_similarity_long.csv"),
        "by_layer": Path(f"{prefix}_weight_clustering_by_layer.csv"),
    }
    data = {}
    for name, path in paths.items():
        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            print(f"[{state}] missing {path}")
            data[name] = None
    return data


def fig_season_vs_year_clustering(all_data: dict, out_path: Path):
    states = [s for s, d in all_data.items() if d["clustering"] is not None]
    if not states:
        print("  no --clustering data for any state, skipping fig_season_vs_year_clustering")
        return

    fig, ax = plt.subplots(figsize=(1.4 * len(states) + 2, 5))
    x = np.arange(len(states))
    width = 0.35

    season_gaps, season_p, year_gaps, year_p = [], [], [], []
    for s in states:
        df = all_data[s]["clustering"]
        season_row = df[df.clustering_by == "season"].iloc[0]
        year_row = df[df.clustering_by == "year"].iloc[0]
        season_gaps.append(season_row.observed_gap)
        season_p.append(season_row.p_value)
        year_gaps.append(year_row.observed_gap)
        year_p.append(year_row.p_value)

    bars_season = ax.bar(x - width / 2, season_gaps, width, label="Season", color=SEASON_COLOR,
                          edgecolor=BAR_EDGE, linewidth=0.8)
    bars_year = ax.bar(x + width / 2, year_gaps, width, label="Year", color=YEAR_COLOR,
                        edgecolor=BAR_EDGE, linewidth=0.8)

    for bars, pvals in [(bars_season, season_p), (bars_year, year_p)]:
        for bar, p in zip(bars, pvals):
            y = bar.get_height()
            offset = 0.002 if y >= 0 else -0.012
            ax.annotate(f"p={p:.3f}", (bar.get_x() + bar.get_width() / 2, y + offset),
                        ha="center", va="bottom" if y >= 0 else "top", fontsize=8)

    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in states])
    ax.set_ylabel("Same-group minus different-group\nmean cosine similarity")
    ax.set_title("Weight-delta clustering: season vs. year")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_weight_similarity_vs_transfer(all_data: dict, out_path: Path):
    states = [s for s, d in all_data.items() if d["long"] is not None and d["corr"] is not None]
    if not states:
        print("  no --long/--corr data for any state, skipping fig_weight_similarity_vs_transfer")
        return

    ncols = 2 if len(states) > 1 else 1
    nrows = int(np.ceil(len(states) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for ax, s in zip(axes_flat, states):
        df = all_data[s]["long"].dropna(subset=["pair_transfer_r2"])
        corr = all_data[s]["corr"]

        for label, mask, color in [("same season", df.same_season == 1, SEASON_COLOR),
                                    ("different season", df.same_season == 0, YEAR_COLOR)]:
            group = df[mask]
            ax.scatter(group.cosine_similarity, group.pair_transfer_r2, c=color, s=22, alpha=0.75,
                       edgecolors="white", linewidths=0.4, label=label)
            if len(group) >= 4:
                slope, intercept = np.polyfit(group.cosine_similarity, group.pair_transfer_r2, 1)
                xs = np.linspace(group.cosine_similarity.min(), group.cosine_similarity.max(), 50)
                ax.plot(xs, slope * xs + intercept, color=color, linewidth=2)

        overall = corr[corr.group == "overall"]
        partial = corr[corr.group == "partial_controlling_for_season_and_year"]
        lines = []
        if not overall.empty:
            lines.append(f"overall r={overall.iloc[0].pearson_r:+.2f} (p={overall.iloc[0].p_value:.2g})")
        if not partial.empty:
            lines.append(f"partial r={partial.iloc[0].pearson_r:+.2f} (p={partial.iloc[0].p_value:.2g})")
        if lines:
            ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, fontsize=9,
                    va="top", ha="left", bbox=dict(boxstyle="round", fc="white", ec="#c3c2b7", alpha=0.9))

        ax.set_title(s.upper())
        ax.set_xlabel("weight-delta cosine similarity")
        ax.set_ylabel("symmetrized transfer R^2")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.2)

    for ax in axes_flat[len(states):]:
        ax.axis("off")

    fig.suptitle("Weight similarity vs. cross-quarter transfer R^2", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def fig_season_clustering_by_layer(all_data: dict, out_path: Path):
    states = [s for s, d in all_data.items() if d["by_layer"] is not None]
    if not states:
        print("  no --by_layer data for any state, skipping fig_season_clustering_by_layer")
        return

    layers_present = []
    for s in states:
        df = all_data[s]["by_layer"]
        layers_present += list(df[df.clustering_by == "season"]["layer_group"].unique())
    layers = [l for l in LAYER_ORDER if l in set(layers_present)]
    if not layers:
        print("  no recognized layer groups, skipping fig_season_clustering_by_layer")
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(layers) + 2, 5.5))
    x = np.arange(len(layers))
    width = 0.8 / len(states)

    for i, s in enumerate(states):
        df = all_data[s]["by_layer"]
        season_df = df[df.clustering_by == "season"].set_index("layer_group")
        gaps = [season_df.loc[l, "observed_gap"] if l in season_df.index else np.nan for l in layers]
        offset = (i - (len(states) - 1) / 2) * width
        ax.bar(x + offset, gaps, width, label=s.upper(), color=state_color(s),
               edgecolor=BAR_EDGE, linewidth=0.6)

    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("Same-season minus different-season\nmean cosine similarity")
    ax.set_title("Season-clustering gap by layer, across states")
    ax.legend(frameon=False, title="State")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", default=["az", "ga", "oh", "pa"])
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--data-dir", default="4_analysis/outputs/weight_drift", type=Path)
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift/paper_figures", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    states = [s.lower() for s in args.states]
    all_data = {s: load_state_data(s, args.variable, args.data_dir) for s in states}

    fig_season_vs_year_clustering(all_data, args.out_dir / "fig_season_vs_year_clustering.png")
    fig_weight_similarity_vs_transfer(all_data, args.out_dir / "fig_weight_similarity_vs_transfer.png")
    fig_season_clustering_by_layer(all_data, args.out_dir / "fig_season_clustering_by_layer.png")


if __name__ == "__main__":
    main()
