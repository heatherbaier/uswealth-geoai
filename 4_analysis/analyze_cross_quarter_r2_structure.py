"""
Decompose an already-built cross-quarter R^2 matrix (see
build_cross_quarter_r2_matrix.py -- this reads its r2_matrix.csv output,
rows = training quarter, columns = evaluation quarter) into an additive
two-way model, an SVD of the grand-mean-centered matrix, and a
row-vs-transpose asymmetry check. Standalone, like every other script in
this folder -- takes r2_matrix.csv, produces its own outputs.

1. Additive two-way model: y_ij ~ mu + row_effect_i + col_effect_j (no
   interaction term). Fit via iterative mean-polish (converges to the
   exact least-squares solution; reduces to plain row/column means when
   the matrix is complete, and still works if a cell or two is missing).
   Reports mu, the row effects (does training quarter i just train a
   generally better/worse model, regardless of what it's evaluated on),
   the column effects (is evaluation quarter j just easier/harder to
   predict, regardless of which model is doing the predicting), the full
   fitted matrix, and the residuals. Also reports what fraction of the
   total sum of squares around mu the additive model explains vs. leaves
   in the residual -- a high fraction means "training quarter" and
   "evaluation quarter" mostly act independently (additively); a low
   fraction means there's a real interaction the additive model can't
   capture (e.g. a specific training-quarter/eval-quarter pair that's
   unusually good or bad beyond what either effect alone predicts).

2. SVD of the grand-mean-centered matrix (mu subtracted, nothing else --
   NOT the row/column-effect residuals from step 1): reports what share
   of the squared Frobenius norm the first singular value carries (a
   single dominant component means the centered matrix is close to
   rank-1 -- i.e. close to a pure row-effect-times-column-effect
   structure) plus the full singular value spectrum. Requires a complete
   matrix (no missing cells) -- SVD isn't well-posed with gaps filled by
   a guess, unlike the mean-polish fit above.

3. Asymmetry: A = M - M^T (only defined for a square matrix with the same
   quarters as both rows and columns, which this always is by
   construction). A_ij = 0 whenever training-quarter-i-eval'd-on-j scores
   the same as training-quarter-j-eval'd-on-i; the largest |A_ij| entries
   are the (training, eval) quarter pairs whose generalization is most
   lopsided in one direction. Also requires a complete matrix.

4. Two plots: an interaction/line plot (one line per training quarter,
   evaluation quarter on the x-axis) -- if the additive model fits well,
   these lines should look roughly parallel, just shifted up/down by each
   row's effect; visibly crossing or fanning lines are the visual
   signature of the interaction the additive model's residual is
   measuring. And a heatmap of the residuals themselves.

Usage:
    python 4_analysis/analyze_cross_quarter_r2_structure.py \
        --r2-matrix ./out_cross_quarter/az_wealth_index_sat/r2_matrix.csv
    # writes additive_model_summary.txt/.csv, singular_values.csv,
    # asymmetries.csv, interaction_lines.png, residual_heatmap.png
    # alongside the input file (override with --out-dir)
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm


def load_matrix(path: Path) -> pd.DataFrame:
    M = pd.read_csv(path, index_col=0)
    if list(M.index) != list(M.columns):
        raise SystemExit(
            f"{path}: row labels and column labels don't match "
            f"({list(M.index)} vs {list(M.columns)}) -- expected the same "
            f"quarters on both axes (see build_cross_quarter_r2_matrix.py)."
        )
    return M


def fit_additive_model(M: pd.DataFrame, max_iter: int = 500, tol: float = 1e-12):
    """
    Iterative mean-polish: converges to the exact least-squares additive
    fit (y_ij = mu + row_effect_i + col_effect_j), handling missing (NaN)
    cells gracefully -- reduces to plain row/column means when the matrix
    is complete. Returns (mu, row_effects, col_effects, fitted, residuals)
    as a scalar and three same-shaped Series/DataFrames.
    """
    vals = M.values.astype(float)
    n_rows, n_cols = vals.shape
    resid = vals.copy()
    row_eff = np.zeros(n_rows)
    col_eff = np.zeros(n_cols)

    for _ in range(max_iter):
        row_delta = np.nan_to_num(np.nanmean(resid, axis=1), nan=0.0)
        resid = resid - row_delta[:, None]
        row_eff += row_delta

        col_delta = np.nan_to_num(np.nanmean(resid, axis=0), nan=0.0)
        resid = resid - col_delta[None, :]
        col_eff += col_delta

        if max(np.abs(row_delta).max(initial=0.0), np.abs(col_delta).max(initial=0.0)) < tol:
            break

    # Re-express with the standard sum-to-zero identifiability constraint:
    # fitted = mu + row_effect_i + col_effect_j is unchanged by this --
    # it's just regrouping (row_eff.mean() + col_eff.mean()) into mu
    # instead of leaving it split arbitrarily between the two effect sets.
    mu = float(row_eff.mean() + col_eff.mean())
    row_effects = pd.Series(row_eff - row_eff.mean(), index=M.index, name="row_effect")
    col_effects = pd.Series(col_eff - col_eff.mean(), index=M.columns, name="col_effect")

    fitted = pd.DataFrame(
        mu + row_effects.values[:, None] + col_effects.values[None, :],
        index=M.index, columns=M.columns,
    )
    residuals = M - fitted
    return mu, row_effects, col_effects, fitted, residuals


def variance_decomposition(M: pd.DataFrame, mu: float, fitted: pd.DataFrame, residuals: pd.DataFrame) -> dict:
    observed = M.notna()
    ss_total = float(((M[observed] - mu) ** 2).sum().sum())
    ss_model = float(((fitted[observed] - mu) ** 2).sum().sum())
    ss_resid = float((residuals[observed] ** 2).sum().sum())
    return {
        "ss_total": ss_total,
        "ss_additive_model": ss_model,
        "ss_residual": ss_resid,
        "frac_explained_by_additive_model": ss_model / ss_total if ss_total else float("nan"),
        "frac_left_in_residual": ss_resid / ss_total if ss_total else float("nan"),
        "n_observed_cells": int(observed.sum().sum()),
    }


def require_complete(M: pd.DataFrame, for_what: str):
    if M.isna().any().any():
        missing = [(r, c) for r in M.index for c in M.columns if pd.isna(M.loc[r, c])]
        raise SystemExit(
            f"{for_what} needs a complete matrix (no missing cells) -- "
            f"{len(missing)} cell(s) not yet validated: {missing}. "
            f"Run the remaining pipeline_configs/generate_cross_validate_config.py "
            f"jobs first."
        )


def svd_decomposition(M: pd.DataFrame, mu: float):
    require_complete(M, "SVD decomposition")
    centered = M.values.astype(float) - mu
    _, s, _ = np.linalg.svd(centered)
    frob2 = float((s ** 2).sum())
    spectrum = pd.DataFrame({
        "singular_value": s,
        "squared_share_of_frobenius_norm": (s ** 2) / frob2 if frob2 else np.full_like(s, np.nan),
    })
    spectrum["cumulative_share"] = spectrum["squared_share_of_frobenius_norm"].cumsum()
    spectrum.index.name = "component"
    return spectrum


def asymmetry_matrix(M: pd.DataFrame, top_k: int):
    require_complete(M, "Asymmetry check")
    A = pd.DataFrame(M.values - M.values.T, index=M.index, columns=M.columns)

    rows = []
    labels = list(M.index)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j:
                continue
            rows.append({
                "train_quarter": labels[i],
                "eval_quarter": labels[j],
                "asymmetry": A.iloc[i, j],  # (i trained, eval j) - (j trained, eval i)
            })
    table = pd.DataFrame(rows)
    table["abs_asymmetry"] = table["asymmetry"].abs()
    table = table.sort_values("abs_asymmetry", ascending=False).drop(columns="abs_asymmetry")
    # Each unordered pair appears twice (once with each sign) -- keep both,
    # they're literally the two directions of the same generalization gap,
    # but only take the top_k UNIQUE pairs by magnitude to avoid double-
    # counting in the printed summary.
    seen_pairs = set()
    deduped = []
    for _, row in table.iterrows():
        pair = frozenset((row["train_quarter"], row["eval_quarter"]))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        deduped.append(row)
        if len(deduped) >= top_k:
            break
    return A, table.reset_index(drop=True), pd.DataFrame(deduped).reset_index(drop=True)


def plot_interaction_lines(M: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 6))
    for train_q in M.index:
        ax.plot(M.columns, M.loc[train_q], "-o", label=f"trained on {train_q}")
    ax.set_xlabel("Evaluated on imagery from")
    ax.set_ylabel("R²")
    ax.set_title("Cross-quarter R²: one line per training quarter\n"
                 "(parallel lines = additive fit; crossing/fanning = interaction)")
    ax.legend(title="Training quarter", loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_residual_heatmap(residuals: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(1.2 * len(residuals.columns) + 3, 1.2 * len(residuals.index) + 3))
    finite = residuals.values[np.isfinite(residuals.values)]
    vmin, vmax = float(finite.min()), float(finite.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    im = ax.imshow(residuals.values, cmap="RdBu", norm=norm,
                    vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(range(len(residuals.columns)))
    ax.set_xticklabels(residuals.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(residuals.index)))
    ax.set_yticklabels(residuals.index)
    ax.set_xlabel("Evaluated on imagery from")
    ax.set_ylabel("Model trained on")
    ax.set_title("Residuals from the additive (mu + row + col) fit\n"
                 "(0 = fully explained by row/column effects alone)")
    for i in range(len(residuals.index)):
        for j in range(len(residuals.columns)):
            ax.text(j, i, f"{residuals.iloc[i, j]:.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Residual R²")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--r2-matrix", required=True, type=Path,
                    help="Path to r2_matrix.csv from build_cross_quarter_r2_matrix.py")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Default: same directory as --r2-matrix")
    p.add_argument("--top-k-asymmetries", type=int, default=10,
                    help="How many largest (by |value|) unique training/eval quarter "
                         "pairs to report (default: 10)")
    args = p.parse_args()

    out_dir = args.out_dir or args.r2_matrix.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    M = load_matrix(args.r2_matrix)
    print(f"Loaded {M.shape[0]}x{M.shape[1]} matrix from {args.r2_matrix}")

    mu, row_effects, col_effects, fitted, residuals = fit_additive_model(M)
    var = variance_decomposition(M, mu, fitted, residuals)

    print(f"\nGrand mean (mu): {mu:.4f}")
    print("\nRow effects (training quarter):")
    print(row_effects.to_string(float_format=lambda x: f"{x:+.4f}"))
    print("\nColumn effects (evaluation quarter):")
    print(col_effects.to_string(float_format=lambda x: f"{x:+.4f}"))
    print(f"\nVariance explained by additive model: "
          f"{var['frac_explained_by_additive_model']:.1%} "
          f"(residual: {var['frac_left_in_residual']:.1%}), "
          f"over {var['n_observed_cells']} observed cell(s)")

    effects_path = out_dir / "additive_model_effects.csv"
    pd.concat([row_effects.rename("effect").to_frame().assign(kind="row"),
               col_effects.rename("effect").to_frame().assign(kind="col")]).to_csv(effects_path)
    fitted.to_csv(out_dir / "additive_model_fitted.csv")
    residuals.to_csv(out_dir / "additive_model_residuals.csv")
    pd.Series(var).to_csv(out_dir / "additive_model_variance_decomposition.csv", header=["value"])
    print(f"\nWrote {effects_path} and 3 more additive_model_*.csv files to {out_dir}")

    try:
        spectrum = svd_decomposition(M, mu)
        spectrum.to_csv(out_dir / "singular_values.csv")
        print(f"\nSVD of grand-mean-centered matrix:")
        print(spectrum.to_string(float_format=lambda x: f"{x:.4f}"))
        print(f"First singular value carries {spectrum['squared_share_of_frobenius_norm'].iloc[0]:.1%} "
              f"of the squared Frobenius norm")
        print(f"Wrote {out_dir / 'singular_values.csv'}")
    except SystemExit as e:
        print(f"\nSkipping SVD: {e}")

    try:
        A, full_asym, top_asym = asymmetry_matrix(M, args.top_k_asymmetries)
        A.to_csv(out_dir / "asymmetry_matrix.csv")
        top_asym.to_csv(out_dir / "top_asymmetries.csv", index=False)
        print(f"\nTop {len(top_asym)} asymmetries "
              f"(train_quarter,eval_quarter) - (eval_quarter,train_quarter), by |value|:")
        print(top_asym.to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        print(f"Wrote {out_dir / 'asymmetry_matrix.csv'} and {out_dir / 'top_asymmetries.csv'}")
    except SystemExit as e:
        print(f"\nSkipping asymmetry check: {e}")

    plot_interaction_lines(M, out_dir / "interaction_lines.png")
    plot_residual_heatmap(residuals, out_dir / "residual_heatmap.png")


if __name__ == "__main__":
    main()
