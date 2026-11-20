"""
Visualizes ONE state's weight_swap_intervention_batch.py --all-cells
run: every (recipient, donor) pair in that state's full off-diagonal
grid, as two heatmaps side by side --

  delta_r2             intervention_r2 - baseline_r2. Diverging,
                        centered at 0: blue = the swap helped, red = it
                        hurt. Auto-ranged to the data (no fixed
                        natural range for an R^2 delta).
  recovery_fraction     (intervention_r2 - baseline_r2) /
                        (donor_r2 - baseline_r2). Diverging, fixed
                        range -1..2 (0 = no help, 1 = fully closed the
                        gap to the donor's own native R^2, >1 =
                        overshot it) -- NOT auto-ranged, since a few
                        cells with a tiny denominator blow this ratio
                        up to +-10 or more and would otherwise wash out
                        the color scale for every other cell. Left
                        blank (NaN) wherever the batch script itself
                        left it NaN (donor_r2 <= baseline_r2 -- see
                        that script's module docstring).

Reads weight_swap_intervention_batch.py's own output directly --
<data-dir>/<state>_<variable>_swap_batch_summary.csv (one row per
pair). If <data-dir>/<state>_<variable>_swap_batch_long.csv is also
present, its swap_groups column (identical on every row of a single
run) is pulled into the figure title, so the figure is unambiguous
about which layer group(s) were swapped without having to ask.

Usage:
    python 4_analysis/plot_weight_swap_batch_state.py --state az --variable wealth_index
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def quarter_labels(df: pd.DataFrame) -> list:
    pairs = set(zip(df["recipient_year"], df["recipient_quarter"])) | \
        set(zip(df["donor_year"], df["donor_quarter"]))
    return [f"Q{q} {y}" for y, q in sorted(pairs)]


def build_matrix(df: pd.DataFrame, value_col: str, labels: list) -> pd.DataFrame:
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for row in df.itertuples(index=False):
        i_label = f"Q{row.recipient_quarter} {row.recipient_year}"
        j_label = f"Q{row.donor_quarter} {row.donor_year}"
        if i_label in matrix.index and j_label in matrix.columns:
            matrix.loc[i_label, j_label] = getattr(row, value_col)
    return matrix


def plot_heatmap(ax, matrix: pd.DataFrame, title: str, vrange: "tuple | None" = None):
    finite = matrix.values[np.isfinite(matrix.values)]
    if finite.size == 0:
        ax.set_title(f"{title}\n(no finite values)")
        return None

    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    im = ax.imshow(matrix.values, cmap="RdBu", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xlabel("Donor (weights swapped in)")
    ax.set_ylabel("Recipient (model being patched)")
    ax.set_title(title)
    return im


def plot_state(df: pd.DataFrame, state: str, swap_groups: "str | None", out_path: Path):
    labels = quarter_labels(df)
    delta_matrix = build_matrix(df, "delta_r2", labels)
    recovery_matrix = build_matrix(df, "recovery_fraction", labels)

    fig, axes = plt.subplots(1, 2, figsize=(1.0 * len(labels) + 6, 0.55 * len(labels) + 4))
    im0 = plot_heatmap(axes[0], delta_matrix, "Delta R^2 (intervention - baseline)")
    im1 = plot_heatmap(axes[1], recovery_matrix, "Recovery fraction", vrange=(-1.0, 2.0))
    if im0 is not None:
        fig.colorbar(im0, ax=axes[0], fraction=0.046)
    if im1 is not None:
        fig.colorbar(im1, ax=axes[1], fraction=0.046)

    swap_str = f" (swapped: {swap_groups})" if swap_groups else ""
    fig.suptitle(f"{state.upper()}: weight-swap intervention, every off-diagonal pair{swap_str}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--data-dir", default="4_analysis/outputs/weight_drift", type=Path,
                    help="weight_swap_intervention_batch.py's own --out-dir")
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift", type=Path)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    state = args.state.lower()
    prefix = args.data_dir / f"{state}_{args.variable}"
    summary_path = Path(f"{prefix}_swap_batch_summary.csv")
    if not summary_path.exists():
        raise SystemExit(f"{summary_path} not found")

    df = pd.read_csv(summary_path)
    if df.empty:
        raise SystemExit(f"{summary_path} is empty -- nothing to plot")

    long_path = Path(f"{prefix}_swap_batch_long.csv")
    swap_groups = None
    if long_path.exists():
        long_df = pd.read_csv(long_path)
        if "swap_groups" in long_df.columns and not long_df.empty:
            swap_groups = long_df["swap_groups"].iloc[0]

    print(f"[{state}] {len(df)} pair(s), swap_groups={swap_groups!r}")
    out_path = args.out_dir / f"{state}_{args.variable}_swap_batch_heatmaps.png"
    plot_state(df, state, swap_groups, out_path)


if __name__ == "__main__":
    main()
