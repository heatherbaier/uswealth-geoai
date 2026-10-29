"""
For every (model state, imagery state) off-diagonal cell in a cross-state
R^2 matrix, is that cell's accuracy STATISTICALLY DISTINGUISHABLE from the
imagery state's OWN model on its OWN held-out set (the diagonal cell)?
The geographic-transfer counterpart of analyze_cross_quarter_
significance.py's seasonal-transfer test -- same question, same R^2-isn't-
a-per-tract-quantity reasoning, but a different (weaker) test, because the
two matrices don't have the same thing to exploit.

Why this is an UNPAIRED bootstrap, not a paired one like analyze_cross_
quarter_significance.py: that script's power comes specifically from
pairing on shared GEOIDs -- generate_cross_validate_config.py's docstring
establishes that every quarter for a state shares the same stable/spatial
split, so a (train_q, eval_q) cell and the (eval_q, eval_q) diagonal cell
cover the EXACT SAME set of held-out tracts, and resampling that shared
set once and scoring both cells on it cancels out tract-level difficulty
(a tract that's just hard to predict pulls both R^2 down together,
leaving the DIFFERENCE less noisy than either R^2 alone).

GEOIDs are state-specific (state+county+tract FIPS) -- an Ohio model's
cross-state cell on Pennsylvania imagery and Pennsylvania's own diagonal
cell share ZERO tracts, by construction (see generate_cross_state_
validate_config.py's docstring). There is no shared population to pair
on, so each cell's R^2 has to be treated as coming from its own
independent population: resample the transfer cell's rows and the
diagonal cell's rows SEPARATELY each bootstrap draw (different n, no
shared index) and look at the resulting difference's distribution. This
is a strictly less powerful test than the paired version -- wider CIs for
the same effect size, because there's no shared-tract noise to cancel --
which also means "not significant" here is weaker evidence of "no real
difference" than it would be from the cross-quarter script. Keep that
asymmetry in mind when comparing the two matrices' significance counts
directly (e.g. "3/12 seasonal cells significant vs. 5/12 geographic
cells" is a real comparison of EFFECT sizes caught; it is not a symmetric
apples-to-apples comparison of STATISTICAL POWER, which is higher for the
seasonal test).

For each off-diagonal cell:
    delta_r2    = R^2(model_state model, imagery_state imagery)
                  - R^2(imagery_state model, imagery_state's own set)
    ci          = percentile bootstrap CI on that delta (--bootstrap-iters
                  resamples, independent per side)
    p_value     = two-sided bootstrap p-value (2x the smaller tail beyond 0)
    significant = p_value < --alpha

Reuses state_ckpt_dir/state_prefix/find_diagonal_preds_csv/
find_cross_state_preds_csv from build_cross_state_r2_matrix.py rather
than reimplementing cell discovery a second time.

Usage:
    python 3_analysis/analyze_cross_state_significance.py \
        --variable wealth_index --year 2016 --quarter 1 \
        --states oh pa az --out-dir ./out_cross_state_sig/2016q1_wealth_index
"""

import sys
from pathlib import Path

import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_state_r2_matrix import (  # noqa: E402
    state_ckpt_dir, state_prefix, find_diagonal_preds_csv, find_cross_state_preds_csv,
)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry  # noqa: E402

MIN_ROWS = 10  # below this, the bootstrap isn't meaningful either side


def load_cell_preds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return pd.DataFrame({"label": df["label"].to_numpy(), "pred": df["pred"].to_numpy()})


def unpaired_bootstrap_r2_delta(label_a, pred_a, label_b, pred_b, n_iters: int, alpha: float, seed: int):
    """a = the TRANSFER cell (model_state on imagery_state), b = imagery_
    state's own diagonal -- independent populations (see module
    docstring), so each is resampled with its OWN size, independently,
    every draw. Returns the point-estimate delta = R2(a) - R2(b), a
    percentile CI, and a two-sided bootstrap p-value for delta != 0 -- or
    None if every resample was degenerate on either side (see bootstrap_r2
    in build_cross_quarter_r2_matrix.py for the same edge case)."""
    rng = np.random.default_rng(seed)
    n_a, n_b = len(label_a), len(label_b)
    point_delta = r2_score(label_a, pred_a) - r2_score(label_b, pred_b)

    deltas = np.full(n_iters, np.nan)
    for i in range(n_iters):
        idx_a = rng.integers(0, n_a, size=n_a)
        idx_b = rng.integers(0, n_b, size=n_b)
        la, lb = label_a[idx_a], label_b[idx_b]
        if np.ptp(la) == 0 or np.ptp(lb) == 0:
            continue
        deltas[i] = r2_score(la, pred_a[idx_a]) - r2_score(lb, pred_b[idx_b])

    valid = deltas[~np.isnan(deltas)]
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_left = float(np.mean(valid <= 0))
    p_right = float(np.mean(valid >= 0))
    p_value = min(1.0, 2 * min(p_left, p_right))
    return {
        "delta_r2": float(point_delta), "ci_lower": float(lo), "ci_upper": float(hi),
        "p_value": p_value, "n_boot": int(valid.size), "n_a": n_a, "n_b": n_b,
    }


def build_significance_matrices(variable, year, quarter, states, version, registry,
                                 n_iters, alpha, seed):
    labels = [s.upper() for s in states]
    n = len(states)
    delta_r2 = pd.DataFrame(np.nan, index=labels, columns=labels)
    p_value = pd.DataFrame(np.nan, index=labels, columns=labels)
    significant = pd.DataFrame(False, index=labels, columns=labels)

    ckpt_dirs = {}
    for state in states:
        try:
            ckpt_dirs[state] = state_ckpt_dir(state, variable, year, quarter, version, registry)
        except SystemExit as e:
            print(f"[{state.upper()}] no trained model found, skipping this row: {e}")

    diag_cells = {}
    for state in states:
        if state not in ckpt_dirs:
            continue
        prefix = state_prefix(state, variable, year, quarter, registry)
        csv_path = find_diagonal_preds_csv(ckpt_dirs[state], prefix)
        if csv_path is None:
            print(f"[{state.upper()} diagonal] not validated -- can't test any cell in this column")
            continue
        diag_cells[state] = load_cell_preds(csv_path)

    detail_rows = []
    for j, imagery_state in enumerate(states):
        diag = diag_cells.get(imagery_state)
        if diag is None:
            continue
        for i, model_state in enumerate(states):
            if model_state == imagery_state or model_state not in ckpt_dirs:
                continue
            prefix = state_prefix(imagery_state, variable, year, quarter, registry)
            csv_path = find_cross_state_preds_csv(ckpt_dirs[model_state], prefix)
            if csv_path is None:
                continue
            cell = load_cell_preds(csv_path)
            if len(cell) < MIN_ROWS or len(diag) < MIN_ROWS:
                print(f"[{labels[i]} -> {labels[j]}] only {len(cell)} transfer / {len(diag)} "
                      f"diagonal row(s) (< {MIN_ROWS}), skipping")
                continue

            result = unpaired_bootstrap_r2_delta(
                cell["label"].to_numpy(), cell["pred"].to_numpy(),
                diag["label"].to_numpy(), diag["pred"].to_numpy(),
                n_iters, alpha, seed,
            )
            if result is None:
                print(f"[{labels[i]} -> {labels[j]}] every bootstrap resample was degenerate "
                      f"(zero label variance) on one side, skipping")
                continue

            is_sig = result["p_value"] < alpha
            delta_r2.iloc[i, j] = result["delta_r2"]
            p_value.iloc[i, j] = result["p_value"]
            significant.iloc[i, j] = is_sig
            detail_rows.append({"model_state": labels[i], "imagery_state": labels[j],
                                 "significant": is_sig, **result})

            marker = "SIGNIFICANT" if is_sig else "not significant"
            direction = "drop" if result["delta_r2"] < 0 else "increase"
            print(f"[{labels[i]} model on {labels[j]} imagery] {direction} of "
                  f"{result['delta_r2']:+.4f} R^2 vs. {labels[j]}'s own model on its own set, "
                  f"95% CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}], "
                  f"p={result['p_value']:.4g}  -- {marker} (n={result['n_a']} vs {result['n_b']})")

    return delta_r2, p_value, significant, pd.DataFrame(detail_rows)


def plot_significance_matrix(delta_r2: pd.DataFrame, significant: pd.DataFrame,
                              year: int, quarter: int, variable: str, alpha: float, out_path: str):
    fig, ax = plt.subplots(figsize=(1.2 * len(delta_r2.columns) + 3, 1.2 * len(delta_r2.index) + 3))

    finite = delta_r2.values[np.isfinite(delta_r2.values)]
    if finite.size == 0:
        raise SystemExit("No delta-R^2 values to plot -- run the significance test first.")

    vmin, vmax = float(finite.min()), float(finite.max())
    if vmin < 0 < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(delta_r2.values, cmap="RdBu", norm=norm,
                    vmin=None if norm else vmin, vmax=None if norm else vmax)

    ax.set_xticks(range(len(delta_r2.columns)))
    ax.set_xticklabels(delta_r2.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(delta_r2.index)))
    ax.set_yticklabels(delta_r2.index)
    ax.set_xlabel("Evaluated on imagery from")
    ax.set_ylabel("Model trained on")
    ax.set_title(f"{year} Q{quarter} {variable}: ΔR² vs. that state's own model\n"
                 f"(* = significant, unpaired bootstrap p<{alpha}; weaker test than cross-quarter -- see docstring)")

    for i in range(len(delta_r2.index)):
        for j in range(len(delta_r2.columns)):
            if i == j:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=True,
                                            facecolor="lightgray", edgecolor="black", linewidth=1))
                ax.text(j, i, "base", ha="center", va="center", fontsize=8, color="dimgray")
                continue
            val = delta_r2.iloc[i, j]
            if not np.isfinite(val):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="gray")
                continue
            star = "*" if bool(significant.iloc[i, j]) else ""
            ax.text(j, i, f"{val:+.2f}{star}", ha="center", va="center", fontsize=9,
                    fontweight="bold" if star else "normal")

    fig.colorbar(im, ax=ax, label="ΔR² (transfer minus that state's own model)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variable", required=True)
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--quarter", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--states", required=True, nargs="+", help="Used as BOTH rows and columns, e.g. oh pa az")
    p.add_argument("--version", default=None, help="Trained version per row (default: latest existing)")
    p.add_argument("--bootstrap-iters", type=int, default=2000,
                    help="Unpaired bootstrap resamples per cell (default 2000)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="Default: ./out_cross_state_sig/<year>q<quarter>_<variable>")
    args = p.parse_args()

    states = [s.lower() for s in args.states]
    registry = load_registry()

    delta_r2, p_value, significant, detail_df = build_significance_matrices(
        args.variable, args.year, args.quarter, states, args.version, registry,
        args.bootstrap_iters, args.alpha, args.seed,
    )

    out_dir = Path(args.out_dir or f"./out_cross_state_sig/{args.year}q{args.quarter}_{args.variable}")
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_r2.to_csv(out_dir / "delta_r2_matrix.csv")
    p_value.to_csv(out_dir / "p_value_matrix.csv")
    significant.to_csv(out_dir / "significant_matrix.csv")
    detail_df.to_csv(out_dir / "significance_detail.csv", index=False)
    print(f"\nWrote {out_dir / 'delta_r2_matrix.csv'}, p_value_matrix.csv, significant_matrix.csv, "
          f"significance_detail.csv")

    if not detail_df.empty:
        n_sig = int(detail_df["significant"].sum())
        print(f"\n{n_sig} / {len(detail_df)} off-diagonal cell(s) significant at alpha={args.alpha}")
        plot_significance_matrix(delta_r2, significant, args.year, args.quarter, args.variable,
                                  args.alpha, str(out_dir / "significance_matrix.png"))


if __name__ == "__main__":
    main()
