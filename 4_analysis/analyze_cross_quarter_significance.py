"""
For every (train quarter, eval quarter) pair in a cross-quarter R^2
matrix, is that pair's accuracy STATISTICALLY DISTINGUISHABLE from the
matched model's own accuracy on that same imagery (the diagonal cell --
e.g. the 2018 Q2 model evaluated on 2018 Q3 imagery, tested against the
2018 Q3 model evaluated on its own held-out set)? A lower R^2 off the
diagonal isn't automatically meaningful -- with a small held-out set, two
R^2 values a few hundredths apart can easily be sampling noise. This
answers that with a real test instead of eyeballing build_cross_quarter_
r2_matrix.py's numbers.

Method: paired bootstrap on delta R^2 (R^2(transfer) - R^2(matched)), not
a t-test on per-tract squared error. R^2 is a nonlinear function of the
WHOLE error distribution, not a per-tract quantity you can average and
run a t-test on -- resampling tracts and recomputing R^2 each time is the
direct way to get a sampling distribution for it. "Paired" matters:
generate_cross_validate_config.py's docstring establishes that every
quarter for a state shares the same train/val/test split (same seed/
spatial_block_deg), so a (train_q, eval_q) cell and the (eval_q, eval_q)
diagonal cell cover the SAME set of held-out GEOIDs -- each bootstrap
draw resamples that shared tract set ONCE and scores both cells on it,
preserving the tract-level correlation between the two (a tract that's
just hard to predict pulls both R^2 down together) instead of treating
them as independent samples.

For each off-diagonal cell:
    delta_r2   = R^2(train_q model, eval_q imagery) - R^2(eval_q model, eval_q imagery)
    ci         = percentile bootstrap CI on that delta (--bootstrap-iters resamples)
    p_value    = two-sided bootstrap p-value (2x the smaller tail beyond 0)
    significant = p_value < --alpha

Reuses model_ckpt_dir/imagery_prefix/find_preds_csv from
build_cross_quarter_r2_matrix.py (state_registry.yml-driven -- AZ/GA/PA
today) and template_ckpt_dir/template_imagery_prefix from
analyze_transfer_penalty_landcover.py (path-template-driven, works for a
pre-registry state like OH without needing a state_registry.yml entry) --
same dual discovery this project's OH scripts already settled on, not
reimplemented a third time.

Usage:
    # OH, single year (matches "is the 2018 Q2 model on Q3 imagery
    # significantly worse than the 2018 Q3 model on its own set" exactly):
    python 4_analysis/analyze_cross_quarter_significance.py \
        --states oh --variable wealth_index --quarters 2018Q1 2018Q2 2018Q3 2018Q4 \
        --out-dir ./out_cross_quarter_sig

    # Several states in one run -- --quarters/--variable apply to all of
    # them, each state gets its own subfolder under --out-dir. Every state
    # here has to actually have that variable/those quarters trained and
    # cross-validated already; a state that doesn't just prints "no
    # trained model found" for that row and moves on, same as a single-
    # state run:
    python 4_analysis/analyze_cross_quarter_significance.py \
        --states oh az ga pa --variable wealth_index --quarters 2018Q1 2018Q2 2018Q3 2018Q4 \
        --out-dir ./out_cross_quarter_sig

    # AZ, already in state_registry.yml (applies to every --states entry
    # in the run, so mix registry-based and template-based states via
    # separate invocations, not one --states list):
    python 4_analysis/analyze_cross_quarter_significance.py \
        --states az ga pa --variable wealth_index_sat --quarters 2016Q1 2016Q2 2016Q3 2016Q4 \
        --use-state-registry --out-dir ./out_cross_quarter_sig
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from build_cross_quarter_r2_matrix import (  # noqa: E402
    model_ckpt_dir, imagery_prefix, find_preds_csv, load_registry, parse_imagery_quarter,
)
from analyze_transfer_penalty_landcover import (  # noqa: E402
    template_ckpt_dir, template_imagery_prefix,
    DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)
from update_ys_labels import normalize_geoid  # noqa: E402

MIN_OVERLAP_TRACTS = 10  # below this, the paired bootstrap isn't meaningful


def load_cell_preds(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return pd.DataFrame({
        "GEOID": df["name"].apply(lambda p: normalize_geoid(Path(p).stem)),
        "label": df["label"],
        "pred": df["pred"],
    }).drop_duplicates(subset="GEOID")


def paired_bootstrap_r2_delta(label_a, pred_a, label_b, pred_b, n_iters: int, alpha: float, seed: int):
    """a = the TRANSFER cell, b = the MATCHED (diagonal) cell, both
    already aligned to the same tracts in the same order (caller's job).
    Returns the point-estimate delta = R2(a) - R2(b), a percentile CI,
    and a two-sided bootstrap p-value for delta != 0 -- or None if every
    resample was degenerate (see bootstrap_r2 in
    build_cross_quarter_r2_matrix.py for the same edge case)."""
    rng = np.random.default_rng(seed)
    n = len(label_a)
    point_delta = r2_score(label_a, pred_a) - r2_score(label_b, pred_b)

    deltas = np.full(n_iters, np.nan)
    for b in range(n_iters):
        idx = rng.integers(0, n, size=n)
        la, lb = label_a[idx], label_b[idx]
        if np.ptp(la) == 0 or np.ptp(lb) == 0:
            continue
        deltas[b] = r2_score(la, pred_a[idx]) - r2_score(lb, pred_b[idx])

    valid = deltas[~np.isnan(deltas)]
    if valid.size == 0:
        return None
    lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_left = float(np.mean(valid <= 0))
    p_right = float(np.mean(valid >= 0))
    p_value = min(1.0, 2 * min(p_left, p_right))
    return {
        "delta_r2": float(point_delta), "ci_lower": float(lo), "ci_upper": float(hi),
        "p_value": p_value, "n_boot": int(valid.size), "n": n,
    }


def resolve_ckpt_and_prefix(state, variable, args, registry):
    """Returns (ckpt_dir_fn, prefix_fn) closures over the chosen discovery
    mode, so the main loop doesn't need to branch per cell."""
    if args.use_state_registry:
        return (
            lambda year, quarter: model_ckpt_dir(state, variable, year, quarter, args.version, registry),
            lambda year, quarter: imagery_prefix(state, variable, year, quarter, registry),
        )
    return (
        lambda year, quarter: template_ckpt_dir(state, variable, year, quarter, args.version,
                                                  args.data_root_template),
        lambda year, quarter: template_imagery_prefix(state, variable, year, quarter,
                                                        args.base_prefix_template),
    )


def build_significance_matrices(state, variable, quarters, ckpt_dir_fn, prefix_fn,
                                 n_iters, alpha, seed):
    labels = [f"Q{q} {y}" for y, q in quarters]
    n = len(quarters)
    delta_r2 = pd.DataFrame(np.nan, index=labels, columns=labels)
    p_value = pd.DataFrame(np.nan, index=labels, columns=labels)
    significant = pd.DataFrame(False, index=labels, columns=labels)

    # Load every cell's preds once, keyed by (row idx, col idx) -- a
    # diagonal cell (j, j) is loaded once and reused both as its own
    # column's baseline and (trivially) never compared to itself.
    cell_data = {}
    for i, (model_year, model_quarter) in enumerate(quarters):
        try:
            ckpt_dir = ckpt_dir_fn(model_year, model_quarter)
        except SystemExit as e:
            print(f"[{labels[i]}] no trained model found, skipping this row: {e}")
            continue
        for j, (imagery_year, imagery_quarter) in enumerate(quarters):
            prefix = prefix_fn(imagery_year, imagery_quarter)
            csv_path = find_preds_csv(ckpt_dir, prefix)
            if csv_path is None:
                continue
            cell_data[(i, j)] = load_cell_preds(csv_path)

    detail_rows = []
    for j, (eval_year, eval_quarter) in enumerate(quarters):
        diag = cell_data.get((j, j))
        if diag is None:
            print(f"[{labels[j]} diagonal] not validated -- can't test any cell in this column")
            continue
        for i, (model_year, model_quarter) in enumerate(quarters):
            if i == j:
                continue
            cell = cell_data.get((i, j))
            if cell is None:
                continue
            merged = cell.merge(diag, on="GEOID", how="inner", suffixes=("_transfer", "_matched"))
            if len(merged) < MIN_OVERLAP_TRACTS:
                print(f"[{labels[i]} -> {labels[j]}] only {len(merged)} overlapping tract(s) "
                      f"(< {MIN_OVERLAP_TRACTS}), skipping")
                continue

            result = paired_bootstrap_r2_delta(
                merged["label_transfer"].to_numpy(), merged["pred_transfer"].to_numpy(),
                merged["label_matched"].to_numpy(), merged["pred_matched"].to_numpy(),
                n_iters, alpha, seed,
            )
            if result is None:
                print(f"[{labels[i]} -> {labels[j]}] every bootstrap resample was degenerate "
                      f"(zero label variance), skipping")
                continue

            is_sig = result["p_value"] < alpha
            delta_r2.iloc[i, j] = result["delta_r2"]
            p_value.iloc[i, j] = result["p_value"]
            significant.iloc[i, j] = is_sig
            detail_rows.append({"model": labels[i], "imagery": labels[j], "significant": is_sig, **result})

            marker = "SIGNIFICANT" if is_sig else "not significant"
            direction = "drop" if result["delta_r2"] < 0 else "increase"
            print(f"[{labels[i]} model on {labels[j]} imagery] {direction} of "
                  f"{result['delta_r2']:+.4f} R^2 vs. the {labels[j]} model's own set, "
                  f"95% CI=[{result['ci_lower']:+.4f}, {result['ci_upper']:+.4f}], "
                  f"p={result['p_value']:.4g}  -- {marker} (n={result['n']})")

    return delta_r2, p_value, significant, pd.DataFrame(detail_rows)


def plot_significance_matrix(delta_r2: pd.DataFrame, significant: pd.DataFrame,
                              state: str, variable: str, alpha: float, out_path: str):
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
    ax.set_title(f"{state.upper()} {variable}: ΔR² vs. matched (diagonal) model\n"
                 f"(* = significant at paired-bootstrap p<{alpha}; diagonal is the baseline, not tested)")

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

    fig.colorbar(im, ax=ax, label="ΔR² (transfer minus matched)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_for_state(state, args, registry, base_out_dir: Path) -> "pd.DataFrame | None":
    quarters = [parse_imagery_quarter(t) for t in args.quarters]
    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)

    print(f"\n{'=' * 70}\n{state.upper()} {args.variable}\n{'=' * 70}")
    delta_r2, p_value, significant, detail_df = build_significance_matrices(
        state, args.variable, quarters, ckpt_dir_fn, prefix_fn,
        args.bootstrap_iters, args.alpha, args.seed,
    )

    out_dir = base_out_dir / f"{state}_{args.variable}"
    out_dir.mkdir(parents=True, exist_ok=True)
    delta_r2.to_csv(out_dir / "delta_r2_matrix.csv")
    p_value.to_csv(out_dir / "p_value_matrix.csv")
    significant.to_csv(out_dir / "significant_matrix.csv")
    detail_df.to_csv(out_dir / "significance_detail.csv", index=False)
    print(f"\nWrote {out_dir / 'delta_r2_matrix.csv'}, p_value_matrix.csv, significant_matrix.csv, "
          f"significance_detail.csv")

    if detail_df.empty:
        print(f"[{state.upper()}] No cells tested -- nothing to plot.")
        return detail_df

    n_sig = int(detail_df["significant"].sum())
    print(f"[{state.upper()}] {n_sig} / {len(detail_df)} off-diagonal cell(s) significant at "
          f"alpha={args.alpha}")
    plot_significance_matrix(delta_r2, significant, state, args.variable, args.alpha,
                              str(out_dir / "significance_matrix.png"))

    detail_df = detail_df.copy()
    detail_df.insert(0, "state", state)
    return detail_df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", required=True, nargs="+",
                    help="One or more state labels, e.g. --states oh, or --states oh az ga pa "
                         "to run all of them in one invocation")
    p.add_argument("--variable", required=True)
    p.add_argument("--quarters", required=True, nargs="+",
                    help="YYYYQN tokens used as BOTH rows and columns, e.g. 2018Q1 2018Q2 2018Q3 2018Q4 "
                         "-- applies to every state in --states")
    p.add_argument("--version", default=None, help="Trained version per row (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml (needs a full "
                         "entry there for every state in --states) instead of --data-root-template/"
                         "--base-prefix-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE,
                    help=f"Default: {DEFAULT_BASE_PREFIX_TEMPLATE!r}")
    p.add_argument("--bootstrap-iters", type=int, default=2000,
                    help="Paired bootstrap resamples per cell (default 2000)")
    p.add_argument("--alpha", type=float, default=0.05, help="Significance threshold (default 0.05)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None,
                    help="Default: ./out_cross_quarter_sig -- each state gets its own "
                         "<out-dir>/<state>_<variable>/ subfolder")
    args = p.parse_args()

    base_out_dir = Path(args.out_dir or "./out_cross_quarter_sig")
    base_out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registry() if args.use_state_registry else None

    states = [s.lower() for s in args.states]
    per_state_details = [run_for_state(state, args, registry, base_out_dir) for state in states]

    if len(states) > 1:
        combined = pd.concat([d for d in per_state_details if d is not None and not d.empty],
                              ignore_index=True)
        combined_path = base_out_dir / "significance_detail_all_states.csv"
        combined.to_csv(combined_path, index=False)
        n_sig = int(combined["significant"].sum()) if not combined.empty else 0
        print(f"\n{'=' * 70}\nALL STATES: {n_sig} / {len(combined)} off-diagonal cell(s) significant "
              f"at alpha={args.alpha} (see {combined_path})")


if __name__ == "__main__":
    main()
