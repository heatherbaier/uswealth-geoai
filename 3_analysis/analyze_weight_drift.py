"""
Does fine-tuning move each quarter's model in a SEASON-SPECIFIC
direction in weight space, or do all quarters drift the same way
regardless of season?

For each trained (model_year, model_quarter) checkpoint: delta = that
checkpoint's BACKBONE weights minus torchvision's pretrained swin_v2_t
ImageNet backbone, expanded to this state's in_channels the exact same
way sail's SwinRegressor.build() does at the start of every training run
(build_backbone(), imported directly from analyze_fc_function_
similarity.py rather than reimplemented, so this is guaranteed to match
the real initialization bit-for-bit -- same reuse analyze_fc_function_
similarity.py itself already relies on). Cosine similarity between every
pair of delta vectors gives one model-by-model similarity matrix per
state.

Why NOT the head (head.0/head.3): the head is a FRESH, randomly
initialized regression layer at the start of every training run (see
sail/src/sail/models/swin.py::SwinRegressor.build()) -- there's no
"pretrained head" to subtract, and that random init isn't reproducible
without the exact training-time seed. Only the backbone (features.*,
norm.*) actually starts from a shared, known pretrained state, so only
it supports a meaningful delta. build_backbone() replaces the head with
nn.Identity() (zero parameters), so its state_dict() naturally has no
head.* keys to begin with; each fine-tuned checkpoint's own head.* keys
are explicitly dropped to match. (extract_fc_weights_tsne.py is the
complementary analysis that looks at just the head, where this same
backbone/head split is drawn and explained the other way around.)

Two checks on the resulting similarity matrix:

  1. Does it cluster by SEASON (quarter), independent of year? Computed
     as (mean cosine similarity among same-quarter model pairs) minus
     (mean among different-quarter pairs) -- and the same gap for
     same_year, as a secondary check (rules out "it's not season, it's
     just that models trained closer together in time/infra look more
     alike for unrelated reasons"). Tested against a null built by
     permuting which quarter/year label is attached to which MODEL
     (every pairwise similarity value stays fixed; only which model
     counts as "same season as which" is reshuffled) -- model pairs
     aren't independent observations (every model participates in many
     pairs) any more than cells were in analyze_cross_quarter_rs_gap.py,
     so this uses the same quarter-level-permutation logic for the same
     reason, applied here at the model level instead of the quarter
     level.

  2. Does weight similarity correlate with transfer accuracy? For every
     model pair (A, B), transfer(A, B) is pulled from the SAME
     off-diagonal cell source analyze_cross_quarter_rs_gap.py reads
     (--band-importance-csv and/or --cross-quarter-r2-csv, same loader
     function imported directly) -- model A evaluated on model B's own
     training quarter's imagery. ASSUMPTION (stated explicitly, since
     this wasn't specified either way): weight similarity has no
     direction, but transfer R^2 does -- transfer(A,B) != transfer(B,A)
     in general -- so the two directions are averaged into one symmetric
     "pair transfer" value before correlating against weight similarity
     (Pearson r). Both raw directions are kept in the long-format output
     too, so you can check whether that averaging is hiding an
     asymmetry. Skipped (with a printed note) if neither --band-
     importance-csv nor --cross-quarter-r2-csv is given -- the season/
     year clustering check above needs no transfer data at all.

     The overall r alone can't tell "weight similarity predicts transfer
     in general" apart from "same-season pairs are just both more
     weight-similar AND higher-transfer, so the correlation is really
     season identity counted twice." Three more rows split that apart:
     the correlation within same-season pairs only, within different-
     season pairs only, and a partial correlation controlling for both
     same_season and same_year (residualize both variables on those two
     dummies via OLS, correlate the residuals). If the partial
     correlation survives close to the overall r, weight similarity
     carries real information beyond season/year identity; if it
     collapses toward 0, the overall correlation was mostly that
     identity effect, not a continuous relationship.

Output (per state):
    <out-dir>/<state>_<variable>_weight_similarity_matrix.csv   full
        model x model cosine similarity matrix
    <out-dir>/<state>_<variable>_weight_similarity_long.csv   same
        matrix, long format (upper triangle only, no self-pairs): one
        row per (model_a, model_b) pair with both models' (year,
        quarter), same_season/same_year flags, and -- where available --
        both transfer directions plus the symmetrized pair_transfer_r2
    <out-dir>/<state>_<variable>_weight_clustering.csv   the same-vs-
        different gap test, one row each for season and year (observed
        gap, null mean/95th pct, p-value)
    <out-dir>/<state>_<variable>_weight_vs_transfer_corr.csv   Pearson r
        between weight similarity and symmetrized transfer R^2, n_pairs
    <fig-dir>/<state>_<variable>_weight_similarity_heatmap.png   models
        ordered by (year, quarter), quarter-colored tick labels
    <fig-dir>/<state>_<variable>_weight_vs_transfer_scatter.png   skipped
        if no transfer source was given

Usage:
    python 3_analysis/analyze_weight_drift.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019 \
        --cross-quarter-r2-csv ./out_cross_quarter/oh_wealth_index/r2_matrix.csv

    # From band_importance.py's output instead:
    python 3_analysis/analyze_weight_drift.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019 \
        --band-importance-csv 3_analysis/outputs/band_importance/oh_wealth_index_band_importance.csv
"""

import argparse
import gc
import itertools
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import torch
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_fc_weights_tsne import find_checkpoint, resolve_ckpt_dir_fn  # noqa: E402
from analyze_fc_function_similarity import build_backbone  # noqa: E402
from analyze_cross_quarter_rs_gap import load_offdiag_cells  # noqa: E402

QUARTER_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}


def collect_checkpoints(state, years, quarters, variable, args, registry):
    ckpt_dir_fn = resolve_ckpt_dir_fn(state, variable, args, registry)
    rows = []
    for year, quarter in itertools.product(years, quarters):
        try:
            ckpt_dir = ckpt_dir_fn(year, quarter)
        except SystemExit as e:
            print(f"[{state} Q{quarter} {year}] no trained model found, skipping: {e}")
            continue
        epoch, ckpt_path = find_checkpoint(ckpt_dir)
        if ckpt_path is None:
            print(f"[{state} Q{quarter} {year}] no model_epoch*.torch under {ckpt_dir}, skipping")
            continue
        rows.append({"year": year, "quarter": quarter, "epoch": epoch, "ckpt_path": ckpt_path})
        print(f"[{state} Q{quarter} {year}] epoch {epoch}: {ckpt_path}")
    if not rows:
        raise SystemExit("No checkpoints found -- nothing to compare.")
    return pd.DataFrame(rows)


def backbone_state(net: torch.nn.Module) -> dict:
    """state_dict() with any head.* keys dropped. build_backbone()'s net
    already has head=nn.Identity() (zero params) so this is a no-op for
    it; it's the fine-tuned checkpoint's raw state_dict that actually
    needs the filter, since that one DOES have trained head.0/head.3
    weights that must be excluded from the pretrained-delta comparison."""
    return {k: v for k, v in net.state_dict().items() if not k.startswith("head")}


def load_checkpoint_backbone_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    return {k: v for k, v in state_dict.items() if not k.startswith("head")}


def flatten_diff(finetuned_state: dict, pretrained_state: dict) -> np.ndarray:
    keys = sorted(pretrained_state.keys())
    missing = [k for k in keys if k not in finetuned_state]
    if missing:
        raise SystemExit(f"checkpoint is missing expected backbone key(s): {missing[:5]} "
                          f"(different architecture/in_channels than the pretrained backbone?)")
    parts = []
    for k in keys:
        ft = finetuned_state[k].detach().cpu().numpy().reshape(-1).astype(np.float32)
        pt = pretrained_state[k].detach().cpu().numpy().reshape(-1).astype(np.float32)
        if ft.shape != pt.shape:
            raise SystemExit(f"shape mismatch at {k!r}: checkpoint {ft.shape} vs pretrained {pt.shape}")
        parts.append(ft - pt)
    return np.concatenate(parts)


def collect_delta_vectors(meta: pd.DataFrame, pretrained_state: dict) -> np.ndarray:
    vectors = []
    for row in meta.itertuples():
        state_dict = load_checkpoint_backbone_state(row.ckpt_path)
        vectors.append(flatten_diff(state_dict, pretrained_state))
        del state_dict
        gc.collect()
        print(f"  [{row.year} Q{row.quarter}] delta vector: {vectors[-1].shape[0]:,} params")
    return np.stack(vectors)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.where(norms > 0, norms, 1.0)
    return normalized @ normalized.T


def clustering_test(sim_matrix: np.ndarray, labels: np.ndarray, n_perms: int, seed: int) -> dict:
    """Mean similarity among pairs sharing `labels` minus mean among
    pairs that don't, tested against a null that permutes which label is
    attached to which MODEL (pairwise similarity values never change,
    only the same/different classification of each pair does) -- the
    model-level analogue of analyze_cross_quarter_rs_gap.py's quarter-
    level permutation, for the same reason (model pairs aren't
    independent: every model appears in many pairs at once)."""
    n = len(labels)
    iu = np.triu_indices(n, k=1)
    sims = sim_matrix[iu]

    def gap_for(lab):
        same = lab[iu[0]] == lab[iu[1]]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan
        return float(sims[same].mean() - sims[~same].mean())

    observed = gap_for(labels)
    rng = np.random.default_rng(seed)
    null_gaps = np.array([gap_for(rng.permutation(labels)) for _ in range(n_perms)])
    valid_null = null_gaps[~np.isnan(null_gaps)]
    p_value = (1 + int(np.sum(valid_null >= observed))) / (len(valid_null) + 1) if len(valid_null) else np.nan
    return {
        "observed_gap": observed, "null_mean": float(np.nanmean(null_gaps)),
        "null_95pct": float(np.nanquantile(valid_null, 0.95)) if len(valid_null) else np.nan,
        "p_value": p_value, "n_models": n, "n_perms": n_perms,
    }


def build_pair_table(sim_matrix: np.ndarray, meta: pd.DataFrame) -> pd.DataFrame:
    n = len(meta)
    rows = []
    for i, j in itertools.combinations(range(n), 2):
        a, b = meta.iloc[i], meta.iloc[j]
        rows.append({
            "model_a_year": a.year, "model_a_quarter": a.quarter,
            "model_b_year": b.year, "model_b_quarter": b.quarter,
            "cosine_similarity": sim_matrix[i, j],
            "same_season": int(a.quarter == b.quarter), "same_year": int(a.year == b.year),
        })
    return pd.DataFrame(rows)


def attach_transfer_r2(pairs: pd.DataFrame, cells: "pd.DataFrame | None") -> pd.DataFrame:
    """For each model pair (A, B), looks up transfer(A->B) and
    transfer(B->A) from the off-diagonal cell table (model evaluated on
    the OTHER model's own training quarter), and adds their symmetrized
    mean as pair_transfer_r2 -- see module docstring's stated assumption
    on why this gets averaged rather than kept directional."""
    pairs = pairs.copy()
    if cells is None:
        pairs["r2_a_to_b"] = np.nan
        pairs["r2_b_to_a"] = np.nan
        pairs["pair_transfer_r2"] = np.nan
        return pairs

    lookup = {(row.model_year, row.model_quarter, row.eval_year, row.eval_quarter): row.r2
              for row in cells.itertuples(index=False)}
    r2_ab, r2_ba = [], []
    for row in pairs.itertuples(index=False):
        r2_ab.append(lookup.get((row.model_a_year, row.model_a_quarter, row.model_b_year, row.model_b_quarter),
                                 np.nan))
        r2_ba.append(lookup.get((row.model_b_year, row.model_b_quarter, row.model_a_year, row.model_a_quarter),
                                 np.nan))
    pairs["r2_a_to_b"] = r2_ab
    pairs["r2_b_to_a"] = r2_ba
    pairs["pair_transfer_r2"] = pairs[["r2_a_to_b", "r2_b_to_a"]].mean(axis=1, skipna=True)
    return pairs


def correlate_with_transfer(pairs: pd.DataFrame):
    sub = pairs.dropna(subset=["pair_transfer_r2"])
    if len(sub) < 4:
        return None
    r, p = pearsonr(sub["cosine_similarity"], sub["pair_transfer_r2"])
    return {"pearson_r": float(r), "p_value": float(p), "n_pairs": len(sub)}


def correlate_with_transfer_by_group(pairs: pd.DataFrame) -> pd.DataFrame:
    """The overall Pearson r alone can't tell "weight similarity predicts
    transfer accuracy in general" apart from "same-season pairs happen to
    be both more weight-similar AND higher-transfer, so the correlation
    is really just season identity showing up twice." Three additional
    views split that apart:
      - within same_season==1 pairs only
      - within same_season==0 pairs only
      - a partial correlation of cosine_similarity vs. pair_transfer_r2
        that controls for BOTH same_season and same_year: residualize
        each variable on those two dummies via OLS, then correlate the
        residuals (the standard partial-correlation construction) -- if
        this survives close to the overall r, weight similarity carries
        real information beyond season/year identity; if it collapses
        toward 0, the overall correlation was mostly that identity
        effect counted twice."""
    sub = pairs.dropna(subset=["pair_transfer_r2"])
    rows = []

    overall = correlate_with_transfer(pairs)
    if overall is not None:
        rows.append({"group": "overall", **overall})

    for label, mask in [("same_season", sub["same_season"] == 1), ("different_season", sub["same_season"] == 0)]:
        group = sub[mask]
        if len(group) < 4:
            print(f"  [{label}] only {len(group)} pair(s) after dropping NaNs -- skipping")
            continue
        r, p = pearsonr(group["cosine_similarity"], group["pair_transfer_r2"])
        rows.append({"group": label, "pearson_r": float(r), "p_value": float(p), "n_pairs": len(group)})

    if len(sub) >= 6:
        controls = sm.add_constant(sub[["same_season", "same_year"]].astype(float))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_resid = sm.OLS(sub["pair_transfer_r2"], controls).fit().resid
            x_resid = sm.OLS(sub["cosine_similarity"], controls).fit().resid
        r, p = pearsonr(x_resid, y_resid)
        rows.append({"group": "partial_controlling_for_season_and_year", "pearson_r": float(r),
                     "p_value": float(p), "n_pairs": len(sub)})
    else:
        print(f"  [partial_controlling_for_season_and_year] only {len(sub)} pair(s) -- skipping")

    return pd.DataFrame(rows)


def plot_heatmap(sim_matrix: np.ndarray, meta: pd.DataFrame, title: str, out_path: Path):
    order = meta.sort_values(["year", "quarter"]).index.to_numpy()
    ordered = sim_matrix[np.ix_(order, order)]
    labels = [f"{meta.loc[i, 'year']}Q{meta.loc[i, 'quarter']}" for i in order]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.45)))
    im = ax.imshow(ordered, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for tick, i in zip(ax.get_xticklabels(), order):
        tick.set_color(QUARTER_COLOR.get(meta.loc[i, "quarter"], "#52514e"))
    for tick, i in zip(ax.get_yticklabels(), order):
        tick.set_color(QUARTER_COLOR.get(meta.loc[i, "quarter"], "#52514e"))
    fig.colorbar(im, ax=ax, label="cosine similarity of (finetuned - pretrained) backbone weights")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_transfer_scatter(pairs: pd.DataFrame, title: str, out_path: Path):
    sub = pairs.dropna(subset=["pair_transfer_r2"])
    fig, ax = plt.subplots(figsize=(7, 6))

    # Per-group trend lines (not just color-coded points) -- this is the
    # same split correlate_with_transfer_by_group() reports numerically:
    # if the same-season and different-season lines have similar slopes,
    # the relationship holds beyond season identity; if one is flat,
    # that group's correlation is mostly/entirely the other one's.
    for label, mask, color in [("same season", sub["same_season"] == 1, "#1baf7a"),
                                ("different season", sub["same_season"] == 0, "#2a78d6")]:
        group = sub[mask]
        ax.scatter(group["cosine_similarity"], group["pair_transfer_r2"], c=color, s=40, alpha=0.8,
                   edgecolors="white", linewidths=0.5, label=label)
        if len(group) >= 4:
            slope, intercept = np.polyfit(group["cosine_similarity"], group["pair_transfer_r2"], 1)
            x_line = np.linspace(group["cosine_similarity"].min(), group["cosine_similarity"].max(), 50)
            ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, alpha=0.9)

    ax.set_xlabel("weight-delta cosine similarity")
    ax.set_ylabel("symmetrized transfer R^2")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", required=True, nargs="+", type=int)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--version", default=None, help="Trained version per checkpoint (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template")
    p.add_argument("--data-root-template", default=None,
                    help="Default: analyze_transfer_penalty_landcover.DEFAULT_DATA_ROOT_TEMPLATE")
    p.add_argument("--band-importance-csv", type=Path, default=None,
                    help="band_importance.py's long output, for the weight-vs-transfer correlation "
                         "check. Optional -- the clustering-by-season/year check needs neither this "
                         "nor --cross-quarter-r2-csv.")
    p.add_argument("--cross-quarter-r2-csv", type=Path, default=None,
                    help="build_cross_quarter_r2_matrix.py's r2_matrix.csv -- cheaper alternative "
                         "to --band-importance-csv for the same correlation check.")
    p.add_argument("--n-perms", type=int, default=999, help="Permutations for the clustering test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="3_analysis/outputs/weight_drift", type=Path)
    p.add_argument("--fig-dir", default="3_analysis/figures/weight_drift", type=Path)
    args = p.parse_args()
    if args.data_root_template is None:
        from analyze_transfer_penalty_landcover import DEFAULT_DATA_ROOT_TEMPLATE
        args.data_root_template = DEFAULT_DATA_ROOT_TEMPLATE

    state = args.state.lower()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry()
    st = resolve_state_settings(state, registry)
    in_channels = st["in_channels"]
    print(f"[{state}] in_channels={in_channels}")

    meta = collect_checkpoints(state, args.years, args.quarters, args.variable, args, registry)
    print(f"\n{len(meta)} checkpoint(s) found")

    print("\nBuilding pretrained backbone (torchvision swin_v2_t, ImageNet weights, "
          f"stem conv expanded to {in_channels} channels)...")
    pretrained_net = build_backbone(in_channels, device="cpu")
    pretrained_state = backbone_state(pretrained_net)
    print(f"Pretrained backbone: {len(pretrained_state)} parameter tensors")

    print("\nComputing (finetuned - pretrained) delta vectors:")
    vectors = collect_delta_vectors(meta, pretrained_state)
    del pretrained_net, pretrained_state
    gc.collect()

    sim_matrix = cosine_similarity_matrix(vectors)
    labels = [f"{r.year}Q{r.quarter}" for r in meta.itertuples()]
    state_variable = f"{state}_{args.variable}"

    matrix_path = args.out_dir / f"{state_variable}_weight_similarity_matrix.csv"
    pd.DataFrame(sim_matrix, index=labels, columns=labels).to_csv(matrix_path)
    print(f"\nWrote {matrix_path}")

    pairs = build_pair_table(sim_matrix, meta)

    if args.band_importance_csv is not None or args.cross_quarter_r2_csv is not None:
        cells = load_offdiag_cells(args.band_importance_csv, args.cross_quarter_r2_csv)
        print(f"Loaded {len(cells)} off-diagonal transfer cell(s) for the correlation check")
    else:
        cells = None
        print("No --band-importance-csv/--cross-quarter-r2-csv given -- skipping the "
              "weight-vs-transfer correlation check (clustering check below still runs)")
    pairs = attach_transfer_r2(pairs, cells)

    long_path = args.out_dir / f"{state_variable}_weight_similarity_long.csv"
    pairs.to_csv(long_path, index=False)
    print(f"Wrote {long_path} ({len(pairs)} pairs)")

    print("\nClustering test (999 perms, model-level label shuffle):")
    cluster_rows = []
    for label_name, label_vals in [("season", meta["quarter"].to_numpy()),
                                    ("year", meta["year"].to_numpy())]:
        result = clustering_test(sim_matrix, label_vals, args.n_perms, args.seed)
        result["clustering_by"] = label_name
        cluster_rows.append(result)
        print(f"  [{label_name}] observed gap={result['observed_gap']:+.4f} vs. null mean="
              f"{result['null_mean']:+.4f} (95th pct={result['null_95pct']:+.4f}) -> "
              f"p={result['p_value']:.4f}")
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_path = args.out_dir / f"{state_variable}_weight_clustering.csv"
    cluster_df.to_csv(cluster_path, index=False)
    print(f"Wrote {cluster_path}")

    print("\nWeight similarity vs. transfer R^2, by group:")
    corr_df = correlate_with_transfer_by_group(pairs)
    corr_path = args.out_dir / f"{state_variable}_weight_vs_transfer_corr.csv"
    if corr_df.empty:
        print("  Not enough pairs with transfer R^2 to correlate -- skipping (and no corr CSV written)")
    else:
        for row in corr_df.itertuples():
            print(f"  [{row.group}] pearson_r={row.pearson_r:+.4f} p={row.p_value:.4g} (n={row.n_pairs})")
        corr_df.to_csv(corr_path, index=False)
        print(f"Wrote {corr_path}")

    plot_heatmap(sim_matrix, meta, f"Weight-delta cosine similarity -- {state.upper()} {args.variable}",
                 args.fig_dir / f"{state_variable}_weight_similarity_heatmap.png")
    if not corr_df.empty:
        plot_transfer_scatter(pairs, f"Weight similarity vs. transfer R^2 -- {state.upper()} {args.variable}",
                               args.fig_dir / f"{state_variable}_weight_vs_transfer_scatter.png")


if __name__ == "__main__":
    main()
