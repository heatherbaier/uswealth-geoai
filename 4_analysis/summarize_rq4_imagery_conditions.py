"""
Condenses RQ4 (does the physical imagery-condition gap between a
model's own quarter and the quarter it's evaluated on explain
cross-quarter transfer accuracy, beyond quality confounds and calendar
identity?) into one table and one figure across states.

Reads analyze_cross_quarter_rs_gap.py's own Step 5/6 output directly --
<data-dir>/<state>_<variable>_rs_gap_sequential_fit.csv (3 rows per
state: quality -> quality+season_pair -> quality+season_pair+<landscape
metric>, cumulative R^2 at each block) and
<data-dir>/<state>_<variable>_rs_gap_sequential_permutation.csv (1 row
per state: the landscape block's R^2-added and its permutation
p-value). That sequential fit is the trustworthy per-state answer (see
that script's module docstring on why the Step 3 joint model is
diagnostic-only); this script does no new statistics, it only reshapes
those two CSVs into something readable across every state at once.

Outputs:
  <out-dir>/rq4_summary_table.csv   one row per state: each block's
      cumulative R^2, the season_pair and landscape blocks' R^2 added,
      the landscape metric's permutation p-value, n_obs.
  <out-dir>/rq4_imagery_conditions.png   one grouped-bar panel, one
      group per state, three bars per group (quality / +season_pair /
      +landscape cumulative R^2), the landscape bar annotated with its
      permutation p-value.

Usage:
    python 4_analysis/summarize_rq4_imagery_conditions.py --states az ca oh pa
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed categorical colors for the three sequential blocks, assigned in
# a FIXED order -- validated colorblind-safe for a 3-color adjacent set
# (scripts/validate_palette.js), same palette plot_weight_drift_figures.py uses.
BLOCK_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # quality, +season_pair, +landscape
BAR_EDGE = "#2b2b2b"


def load_state_summary(state: str, variable: str, data_dir: Path):
    prefix = data_dir / f"{state}_{variable}"
    fit_path = Path(f"{prefix}_rs_gap_sequential_fit.csv")
    perm_path = Path(f"{prefix}_rs_gap_sequential_permutation.csv")
    if not fit_path.exists() or not perm_path.exists():
        return None

    fit_df = pd.read_csv(fit_path)
    perm_df = pd.read_csv(perm_path)
    if fit_df.empty or len(fit_df) != 3 or perm_df.empty:
        return None

    quality_row, season_row, landscape_row = fit_df.iloc[0], fit_df.iloc[1], fit_df.iloc[2]
    landscape_metric = perm_df.iloc[0]["landscape_metric"]
    return {
        "state": state,
        "quality_r2": quality_row["r_squared"],
        "quality_season_pair_r2": season_row["r_squared"],
        f"quality_season_pair_{landscape_metric}_r2": landscape_row["r_squared"],
        "season_pair_r2_added": season_row["r_squared_added"],
        "landscape_metric": landscape_metric,
        "landscape_r2_added": landscape_row["r_squared_added"],
        "landscape_p_value": perm_df.iloc[0]["p_value"],
        "n_obs": int(landscape_row["n_obs"]),
    }


def build_table(states: list, variable: str, data_dir: Path) -> pd.DataFrame:
    rows = []
    for state in [s.lower() for s in states]:
        summary = load_state_summary(state, variable, data_dir)
        if summary is None:
            print(f"[{state}] missing or incomplete sequential-fit/-permutation CSVs in {data_dir}, skipping")
            continue
        rows.append(summary)
        print(f"[{state}] quality R^2={summary['quality_r2']:.4f} -> "
              f"+season_pair R^2={summary['quality_season_pair_r2']:.4f} "
              f"(+{summary['season_pair_r2_added']:.4f}) -> "
              f"+{summary['landscape_metric']} R^2 added={summary['landscape_r2_added']:+.4f} "
              f"(p={summary['landscape_p_value']:.4f}, n={summary['n_obs']})")
    return pd.DataFrame(rows)


def plot_figure(table: pd.DataFrame, out_path: Path):
    if table.empty:
        print("  no state has both result files -- skipping figure")
        return

    states = table["state"].tolist()
    landscape_metric = table["landscape_metric"].iloc[0]
    block_cols = ["quality_r2", "quality_season_pair_r2", f"quality_season_pair_{landscape_metric}_r2"]
    block_labels = ["Quality", "+ Season pair", f"+ {landscape_metric}"]

    x = np.arange(len(states))
    width = 0.25
    fig, ax = plt.subplots(figsize=(1.8 * len(states) + 2, 5.5))

    for i, (col, label, color) in enumerate(zip(block_cols, block_labels, BLOCK_COLORS)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, table[col], width, label=label, color=color,
                       edgecolor=BAR_EDGE, linewidth=0.8)
        if i == 2:
            for bar, p in zip(bars, table["landscape_p_value"]):
                y = bar.get_height()
                ax.annotate(f"p={p:.3f}", (bar.get_x() + bar.get_width() / 2, y + 0.005),
                            ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in states])
    ax.set_ylabel("Cumulative R^2 (cross-quarter transfer accuracy)")
    ax.set_title(f"RQ4: does {landscape_metric} add explanatory power\n"
                 "beyond imagery quality and calendar identity?")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--data-dir", default="4_analysis/outputs/cross_quarter_rs_gap", type=Path,
                    help="analyze_cross_quarter_rs_gap.py's own --out-dir")
    p.add_argument("--out-dir", default="4_analysis/outputs/cross_quarter_rs_gap", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    table = build_table(args.states, args.variable, args.data_dir)
    if table.empty:
        raise SystemExit("No state had both result files -- nothing to write.")

    table_path = args.out_dir / "rq4_summary_table.csv"
    table.to_csv(table_path, index=False)
    print(f"\nWrote {table_path}")

    plot_figure(table, args.out_dir / "rq4_imagery_conditions.png")


if __name__ == "__main__":
    main()
