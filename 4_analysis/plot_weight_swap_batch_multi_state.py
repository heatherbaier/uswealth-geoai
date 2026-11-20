"""
Same visualization as plot_weight_swap_batch_state.py (delta R^2 and
recovery-fraction heatmaps over every off-diagonal recipient x donor
pair), extended to small multiples across every state in --states --
one row per state, so the pattern can be compared side by side instead
of state by state.

Reuses plot_weight_swap_batch_state.py's quarter_labels()/build_matrix()/
plot_heatmap() directly (one implementation, not two copies).

Reads weight_swap_intervention_batch.py's own output per state --
<data-dir>/<state>_<variable>_swap_batch_summary.csv (and
..._swap_batch_long.csv, if present, for each state's swap_groups
label). A state missing its summary CSV is skipped with a message, not
a crash.

Usage:
    python 4_analysis/plot_weight_swap_batch_multi_state.py --states az ca oh pa
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_weight_swap_batch_state import build_matrix, plot_heatmap, quarter_labels  # noqa: E402


def load_state(state: str, variable: str, data_dir: Path):
    prefix = data_dir / f"{state}_{variable}"
    summary_path = Path(f"{prefix}_swap_batch_summary.csv")
    if not summary_path.exists():
        return None, None

    df = pd.read_csv(summary_path)
    if df.empty:
        return None, None

    long_path = Path(f"{prefix}_swap_batch_long.csv")
    swap_groups = None
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        if "swap_groups" in long_df.columns and not long_df.empty:
            swap_groups = long_df["swap_groups"].iloc[0]
    return df, swap_groups


def plot_all_states(all_data: dict, out_path: Path):
    states = list(all_data.keys())
    max_labels = max(len(quarter_labels(df)) for df, _ in all_data.values())
    fig, axes = plt.subplots(len(states), 2,
                              figsize=(0.9 * max_labels + 6, (0.5 * max_labels + 3.5) * len(states)),
                              squeeze=False)

    for row, state in enumerate(states):
        df, swap_groups = all_data[state]
        labels = quarter_labels(df)
        delta_matrix = build_matrix(df, "delta_r2", labels)
        recovery_matrix = build_matrix(df, "recovery_fraction", labels)
        swap_str = f" (swapped: {swap_groups})" if swap_groups else ""

        im0 = plot_heatmap(axes[row][0], delta_matrix, f"{state.upper()}: delta R^2{swap_str}")
        im1 = plot_heatmap(axes[row][1], recovery_matrix, f"{state.upper()}: recovery fraction",
                            vrange=(-1.0, 2.0))
        if im0 is not None:
            fig.colorbar(im0, ax=axes[row][0], fraction=0.046)
        if im1 is not None:
            fig.colorbar(im1, ax=axes[row][1], fraction=0.046)

    fig.suptitle("Weight-swap intervention, every off-diagonal pair, across states")
    fig.text(0.5, 0.005, "Gray = no data (diagonal) or excluded (donor R^2 <= baseline R^2) -- not a true 0",
              ha="center", fontsize=8, color="#595959")
    fig.tight_layout(rect=(0, 0.015, 1, 1))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--data-dir", default="4_analysis/outputs/weight_drift", type=Path,
                    help="weight_swap_intervention_batch.py's own --out-dir")
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_data = {}
    for state in [s.lower() for s in args.states]:
        df, swap_groups = load_state(state, args.variable, args.data_dir)
        if df is None:
            print(f"[{state}] missing or empty swap_batch_summary.csv in {args.data_dir}, skipping")
            continue
        all_data[state] = (df, swap_groups)
        print(f"[{state}] {len(df)} pair(s), swap_groups={swap_groups!r}")

    if not all_data:
        raise SystemExit("No state had a usable swap_batch_summary.csv -- nothing to plot.")

    out_path = args.out_dir / f"{args.variable}_swap_batch_heatmaps_all_states.png"
    plot_all_states(all_data, out_path)


if __name__ == "__main__":
    main()
