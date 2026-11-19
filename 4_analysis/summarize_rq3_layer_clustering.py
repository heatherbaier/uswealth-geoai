"""
Condenses RQ3 (which parts of the model carry the seasonal difference,
early layers that read spectral/textural features or later layers that
combine them into a wealth estimate?) into one table and one figure
across states.

Reads analyze_weight_drift_by_layer.py's own output directly --
<data-dir>/<state>_<variable>_weight_clustering_by_layer.csv (one row
per layer_group x clustering_by in {season, year}: observed_gap,
p_value, n_params, is_delta -- see that script's module docstring for
what "season-clustering gap" means and why year is the control). This
script does no new statistics, it only reshapes that CSV into
something readable across every state and layer at once.

Outputs:
  <out-dir>/rq3_summary_table.csv   one row per (state, layer_group):
      season_gap, season_p_value, year_gap, year_p_value, n_params.
  <out-dir>/rq3_layer_clustering.png   one grouped-bar panel, one group
      per layer (patch_embed -> stage1..4 -> final_norm -> head, in
      anatomical order), bars grouped by state, showing the season-
      clustering gap at that layer, each bar annotated with its
      permutation p-value.

Usage:
    python 4_analysis/summarize_rq3_layer_clustering.py --states az ca oh pa
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Must match analyze_weight_drift_by_layer.py's own ANATOMICAL_ORDER.
LAYER_ORDER = ["patch_embed", "stage1", "stage2", "stage3", "stage4", "final_norm", "head"]

# Fixed categorical colors, one per state, assigned in a FIXED order --
# same palette/state mapping as plot_weight_drift_figures.py, validated
# colorblind-safe (scripts/validate_palette.js).
STATE_COLORS = {"az": "#2a78d6", "ga": "#eb6834", "oh": "#1baf7a", "pa": "#eda100",
                "ca": "#e87ba4"}
BAR_EDGE = "#2b2b2b"


def state_color(state: str) -> str:
    return STATE_COLORS.get(state, "#52514e")


def load_state_layers(state: str, variable: str, data_dir: Path):
    path = data_dir / f"{state}_{variable}_weight_clustering_by_layer.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df


def build_table(states: list, variable: str, data_dir: Path) -> pd.DataFrame:
    rows = []
    for state in [s.lower() for s in states]:
        df = load_state_layers(state, variable, data_dir)
        if df is None:
            print(f"[{state}] missing or empty weight_clustering_by_layer.csv in {data_dir}, skipping")
            continue

        season = df[df.clustering_by == "season"].set_index("layer_group")
        year = df[df.clustering_by == "year"].set_index("layer_group")
        layers = [l for l in LAYER_ORDER if l in season.index or l in year.index]
        for layer in layers:
            row = {"state": state, "layer_group": layer}
            if layer in season.index:
                row["season_gap"] = season.loc[layer, "observed_gap"]
                row["season_p_value"] = season.loc[layer, "p_value"]
                row["n_params"] = int(season.loc[layer, "n_params"])
            if layer in year.index:
                row["year_gap"] = year.loc[layer, "observed_gap"]
                row["year_p_value"] = year.loc[layer, "p_value"]
            rows.append(row)
            print(f"[{state}] {layer}: season_gap={row.get('season_gap', float('nan')):+.4f} "
                  f"(p={row.get('season_p_value', float('nan')):.4f}) "
                  f"year_gap={row.get('year_gap', float('nan')):+.4f} "
                  f"(p={row.get('year_p_value', float('nan')):.4f})")

    return pd.DataFrame(rows)


def plot_figure(table: pd.DataFrame, out_path: Path):
    if table.empty:
        print("  no state has a usable weight_clustering_by_layer.csv -- skipping figure")
        return

    states = sorted(table["state"].unique(), key=lambda s: list(table["state"].unique()).index(s))
    layers = [l for l in LAYER_ORDER if l in table["layer_group"].values]
    if not layers:
        print("  no recognized layer groups -- skipping figure")
        return

    fig, ax = plt.subplots(figsize=(1.6 * len(layers) + 2, 5.5))
    x = np.arange(len(layers))
    width = 0.8 / len(states)

    for i, state in enumerate(states):
        state_df = table[table.state == state].set_index("layer_group")
        gaps = [state_df.loc[l, "season_gap"] if l in state_df.index else np.nan for l in layers]
        pvals = [state_df.loc[l, "season_p_value"] if l in state_df.index else np.nan for l in layers]
        offset = (i - (len(states) - 1) / 2) * width
        bars = ax.bar(x + offset, gaps, width, label=state.upper(), color=state_color(state),
                       edgecolor=BAR_EDGE, linewidth=0.6)
        for bar, p in zip(bars, pvals):
            if np.isnan(p):
                continue
            y = bar.get_height()
            offset_y = 0.002 if y >= 0 else -0.012
            ax.annotate(f"{p:.3f}", (bar.get_x() + bar.get_width() / 2, y + offset_y),
                        ha="center", va="bottom" if y >= 0 else "top", fontsize=6, rotation=90)

    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=30, ha="right")
    ax.set_ylabel("Same-season minus different-season\nmean cosine similarity")
    ax.set_title("RQ3: season-clustering gap by layer, across states\n(labels: permutation p-value)")
    ax.legend(frameon=False, title="State")
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--data-dir", default="4_analysis/outputs/weight_drift", type=Path,
                    help="analyze_weight_drift_by_layer.py's own --out-dir")
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(args.states, args.variable, args.data_dir)
    if table.empty:
        raise SystemExit("No state had a usable weight_clustering_by_layer.csv -- nothing to write.")

    table_path = args.out_dir / "rq3_summary_table.csv"
    table.to_csv(table_path, index=False)
    print(f"\nWrote {table_path}")

    plot_figure(table, args.out_dir / "rq3_layer_clustering.png")


if __name__ == "__main__":
    main()
