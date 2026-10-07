"""
Functional (not parameter-space) comparison of each trained model's FC
head across seasons -- a follow-up to extract_fc_weights_tsne.py, which
compares raw head.0/head.3 WEIGHT VECTORS across models and found no
strong pattern. That's a real blind spot, not necessarily a null result:
the 256-unit hidden layer has no canonical ordering across independently
-trained networks, so two models computing the IDENTICAL function can
still end up with very different head.0 weight vectors purely from
hidden units landing in a different order during training (and the same
for any orthogonal rotation/sign flip/rescaling SGD happens to settle on)
-- noise that can swamp a real seasonal signal in both the PCA step and
the t-SNE embedding, independent of whether a true difference exists.

This compares what each head DOES instead of what its parameters ARE:
feed the SAME shared probe set of synthetic backbone-feature vectors
through every model's head (plain matrix ops, reading head.0/head.3
straight out of state_dict -- same minimal approach as
extract_fc_weights_tsne.py, no torchvision/full-model build needed) and
compare the resulting 256-dim hidden-layer ACTIVATIONS across models with
linear CKA (Centered Kernel Alignment -- Kornblith et al. 2019,
"Similarity of Neural Network Representations Revisited"). Linear CKA is
invariant to orthogonal transforms (including permutation) and isotropic
rescaling of either representation -- exactly the symmetries that make
raw weight comparison unreliable here -- and it's the standard tool in
deep learning interpretability for "do two differently-parameterized
networks compute the same function."

Probe set caveat: these are synthetic (standard normal) vectors in the
backbone's 768-dim feature space, not real features extracted from actual
imagery through each model's own (also fine-tuned, and so also somewhat
season-specific) backbone. That isolates the HEAD's function specifically
-- matching the original "FC weights" question -- at the cost of probing
regions of feature space real Swin features may never actually visit.
Treat a result here as "do these heads implement different functions on
a generic probe distribution," not "do they behave differently on real
imagery." Extending this to real extracted features is future work, not
done here.

For whether any apparent season clustering is more than chance, this
runs a permutation test directly on the pairwise CKA matrix (Mantel-test
style): the observed (mean within-quarter CKA) - (mean between-quarter
CKA) gap, against a null built by reshuffling quarter labels across the
SAME similarity matrix many times.

Usage:
    python 3_analysis/analyze_fc_function_similarity.py \
        --states az ga oh pa --years 2016 2017 2018 2019 --quarters 1 2 3 4 \
        --variable wealth_index --out-dir ./out_fc_function_similarity
"""

import argparse
import itertools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_quarter_r2_matrix import load_registry  # noqa: E402
from analyze_transfer_penalty_landcover import DEFAULT_DATA_ROOT_TEMPLATE  # noqa: E402
from extract_fc_weights_tsne import (  # noqa: E402
    find_checkpoint, resolve_ckpt_dir_fn, QUARTER_STYLE,
)


def load_head_params(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    missing = [k for k in ("head.0.weight", "head.0.bias", "head.3.weight", "head.3.bias")
               if k not in state_dict]
    if missing:
        raise SystemExit(f"{ckpt_path}: missing expected key(s) {missing}")
    return {
        "W0": state_dict["head.0.weight"].detach().cpu().numpy(),  # (256, in_features)
        "b0": state_dict["head.0.bias"].detach().cpu().numpy(),    # (256,)
        "W3": state_dict["head.3.weight"].detach().cpu().numpy(),  # (1, 256)
        "b3": state_dict["head.3.bias"].detach().cpu().numpy(),    # (1,)
    }


def head_forward(params: dict, probes: np.ndarray):
    """probes: (n_probes, in_features). Returns (hidden, output):
    hidden (n_probes, 256) is head.0's post-ReLU activation; output
    (n_probes, 1) is head.3's. Dropout is identity at eval time, so it's
    skipped entirely -- this matches nn.Sequential(Linear, ReLU, Dropout,
    Linear) run in eval mode exactly. Plain numpy matmul: nn.Linear
    computes y = x @ W^T + b."""
    hidden = probes @ params["W0"].T + params["b0"]
    hidden = np.maximum(hidden, 0.0)
    output = hidden @ params["W3"].T + params["b3"]
    return hidden, output


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Linear CKA between two activation matrices evaluated on the SAME
    n inputs: X (n, p), Y (n, q). Invariant to any orthogonal transform
    (incl. permutation) or isotropic rescaling applied to either matrix.
    Standard simplified form (Kornblith et al. 2019): center each
    matrix's columns (features) across the n samples, then
        CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
    equivalent to the Gram-matrix HSIC definition but O(n*p + n*q)
    instead of forming n x n matrices."""
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    hsic_xy = np.linalg.norm(Xc.T @ Yc, ord="fro") ** 2
    hsic_xx = np.linalg.norm(Xc.T @ Xc, ord="fro") ** 2
    hsic_yy = np.linalg.norm(Yc.T @ Yc, ord="fro") ** 2
    if hsic_xx == 0 or hsic_yy == 0:
        return float("nan")
    return float(hsic_xy / np.sqrt(hsic_xx * hsic_yy))


def build_probe_set(in_features: int, n_probes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n_probes, in_features)).astype(np.float32)


def collect_activations(states, years, quarters, variable, args, registry, probes: np.ndarray):
    in_features = probes.shape[1]
    rows, activations = [], []
    for state in states:
        ckpt_dir_fn = resolve_ckpt_dir_fn(state, variable, args, registry)
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

            params = load_head_params(ckpt_path)
            if params["W0"].shape[1] != in_features:
                print(f"[{state} Q{quarter} {year}] WARNING: head.0 expects "
                      f"{params['W0'].shape[1]}-dim input, probe set is {in_features}-dim "
                      f"(different backbone architecture?) -- skipping")
                continue

            hidden, _ = head_forward(params, probes)
            rows.append({"state": state, "year": year, "quarter": quarter, "epoch": epoch,
                        "ckpt_path": ckpt_path})
            activations.append(hidden)
            print(f"[{state} Q{quarter} {year}] epoch {epoch}: {ckpt_path}")

    if not rows:
        raise SystemExit("No checkpoints found -- nothing to compare.")
    return pd.DataFrame(rows), activations


def build_cka_matrix(activations: list) -> np.ndarray:
    n = len(activations)
    cka = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = linear_cka(activations[i], activations[j])
            cka[i, j] = cka[j, i] = c
    return cka


def quarter_similarity_test(cka: np.ndarray, quarters: np.ndarray, n_permutations: int, seed: int):
    """Mantel-style permutation test: is the observed gap between mean
    within-quarter and mean between-quarter CKA bigger than chance, given
    this exact similarity matrix? Reshuffles quarter LABELS (not the
    matrix) each draw, so the null respects the real similarity
    structure -- only whether quarter happens to explain any of it.

    Also reports a standardized effect size (gap / pooled SD of the two
    groups, i.e. Cohen's d) alongside the p-value. With --n-probes in the
    thousands, CKA estimates are precise enough that the permutation test
    can call a genuinely tiny gap "significant" (p<0.05 with within/
    between means that agree to 3 decimal places is a real failure mode,
    not hypothetical -- see this script's own first real run). p<0.05
    answers "is the gap exactly zero"; effect size answers "does the gap
    matter". Report both, trust neither alone."""
    n = len(quarters)
    iu = np.triu_indices(n, k=1)
    pairs = cka[iu]

    def gap_for(labels):
        same = labels[iu[0]] == labels[iu[1]]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan
        return pairs[same].mean() - pairs[~same].mean()

    observed = gap_for(quarters)
    rng = np.random.default_rng(seed)
    null = np.array([gap_for(rng.permutation(quarters)) for _ in range(n_permutations)])
    null = null[~np.isnan(null)]
    p_value = float(np.mean(null >= observed)) if len(null) else float("nan")

    same_mask = quarters[iu[0]] == quarters[iu[1]]
    same_vals, diff_vals = pairs[same_mask], pairs[~same_mask]

    effect_size = float("nan")
    if len(same_vals) > 1 and len(diff_vals) > 1:
        pooled_sd = np.sqrt((same_vals.var(ddof=1) + diff_vals.var(ddof=1)) / 2)
        # A numerically-near-zero (not just exactly-zero) pooled SD makes
        # this ratio explode into a meaningless huge number rather than a
        # real effect size -- guard with a small floor instead of `> 0`.
        if pooled_sd > 1e-6:
            effect_size = float(observed / pooled_sd)

    return {
        "observed_gap": float(observed),
        "mean_within_quarter_cka": float(same_vals.mean()),
        "mean_between_quarter_cka": float(diff_vals.mean()),
        "effect_size_cohens_d": effect_size,
        "min_pairwise_cka": float(pairs.min()),
        "p_value": p_value,
        "n_permutations": len(null),
    }


def plot_cka_heatmap(cka: pd.DataFrame, meta: pd.DataFrame, title: str, out_path: str):
    order = meta.sort_values(["quarter", "state", "year"]).index.to_numpy()
    ordered = cka.values[np.ix_(order, order)]
    labels = [f"{meta.loc[i, 'state']}{meta.loc[i, 'year']}Q{meta.loc[i, 'quarter']}" for i in order]

    fig, ax = plt.subplots(figsize=(max(8, 0.35 * len(order)), max(7, 0.35 * len(order))))
    # CKA is a bounded [0,1] similarity -- a pure magnitude with no
    # polarity around a meaningful zero -- so this is the sequential
    # (one hue, light->dark) case, not the diverging RdBu used for R^2/
    # delta-R^2 elsewhere in this project (those have real polarity:
    # negative R^2 means worse than the mean).
    im = ax.imshow(ordered, cmap="viridis", vmin=0, vmax=1)

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(title)

    # Mark quarter-block boundaries so same-quarter submatrices are easy
    # to compare against the off-block (cross-quarter) regions by eye.
    quarters_ordered = meta.loc[order, "quarter"].to_numpy()
    boundaries = np.where(np.diff(quarters_ordered) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=1.2)
        ax.axvline(b, color="white", lw=1.2)

    fig.colorbar(im, ax=ax, label="Linear CKA (0 = unrelated, 1 = identical function)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mds_embedding(meta: pd.DataFrame, cka: np.ndarray, title: str, out_path: str, seed: int):
    distance = np.clip(1 - cka, 0, None)
    np.fill_diagonal(distance, 0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              normalized_stress="auto")
    coords = mds.fit_transform(distance)

    fig, ax = plt.subplots(figsize=(8, 7))
    for q in sorted(meta["quarter"].unique()):
        style = QUARTER_STYLE.get(q, {"color": "#52514e", "marker": "x", "label": f"Q{q}"})
        idx = (meta["quarter"] == q).to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], c=style["color"], marker=style["marker"],
                   s=70, edgecolors="white", linewidths=0.6, label=style["label"], alpha=0.9)
    if len(meta) <= 40:
        for (x, y), row in zip(coords, meta.itertuples()):
            ax.annotate(f"{row.state}{row.year}", (x, y), fontsize=7, color="#52514e",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    ax.legend(title="Quarter", frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", required=True, nargs="+")
    p.add_argument("--years", required=True, nargs="+", type=int)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--variable", required=True)
    p.add_argument("--version", default=None, help="Trained version per checkpoint (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--n-probes", type=int, default=2000,
                    help="Synthetic probe vectors in backbone-feature space (default 2000)")
    p.add_argument("--n-permutations", type=int, default=2000,
                    help="Permutations for the quarter-similarity significance test (default 2000)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="Default: ./out_fc_function_similarity")
    args = p.parse_args()

    out_dir = Path(args.out_dir or "./out_fc_function_similarity")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry() if args.use_state_registry else None
    states = [s.lower() for s in args.states]

    # in_features is read from the first checkpoint found rather than
    # hardcoded (768 for swin_v2_t today, but this way a different
    # backbone just works): a quick first pass to discover ONE checkpoint
    # before building the real, shared probe set every model needs to be
    # evaluated on identically.
    probe_in_features = None
    for state in states:
        ckpt_dir_fn = resolve_ckpt_dir_fn(state, args.variable, args, registry)
        for year, quarter in itertools.product(args.years, args.quarters):
            try:
                ckpt_dir = ckpt_dir_fn(year, quarter)
            except SystemExit:
                continue
            _, ckpt_path = find_checkpoint(ckpt_dir)
            if ckpt_path is not None:
                probe_in_features = load_head_params(ckpt_path)["W0"].shape[1]
                break
        if probe_in_features is not None:
            break
    if probe_in_features is None:
        raise SystemExit("No checkpoints found -- can't even determine the probe dimensionality.")

    probes = build_probe_set(probe_in_features, args.n_probes, args.seed)
    print(f"Probe set: {args.n_probes} x {probe_in_features}-dim synthetic vectors (seed={args.seed})")

    meta, activations = collect_activations(states, args.years, args.quarters, args.variable,
                                             args, registry, probes)
    print(f"\nCollected {len(activations)} checkpoint(s)")

    cka = build_cka_matrix(activations)
    labels = [f"{r.state}_{r.year}_q{r.quarter}" for r in meta.itertuples()]
    cka_df = pd.DataFrame(cka, index=labels, columns=labels)
    cka_path = out_dir / "cka_matrix.csv"
    cka_df.to_csv(cka_path)
    print(f"Wrote {cka_path}")

    quarters = meta["quarter"].to_numpy()
    test = quarter_similarity_test(cka, quarters, args.n_permutations, args.seed)
    print(f"\nMean within-quarter CKA:  {test['mean_within_quarter_cka']:.4f}")
    print(f"Mean between-quarter CKA: {test['mean_between_quarter_cka']:.4f}")
    print(f"Observed gap: {test['observed_gap']:+.4f}  "
          f"(permutation p={test['p_value']:.4g}, {test['n_permutations']} permutations, "
          f"effect size d={test['effect_size_cohens_d']:+.3f})")

    # p<0.05 answers "is the gap exactly zero"; with --n-probes in the
    # thousands the test has enough power to call a trivial gap
    # "significant". Effect size (Cohen's d on the pooled within/between
    # distributions) answers "does the gap matter" -- gate the claim on
    # both, using the standard small/medium/large thresholds (0.2/0.5/0.8)
    # rather than reporting significance alone.
    d = abs(test["effect_size_cohens_d"])
    if test["p_value"] < 0.05 and d >= 0.2:
        size_word = "large" if d >= 0.8 else ("medium" if d >= 0.5 else "small")
        print(f"-> Statistically significant AND a {size_word} effect size (d={d:.2f}) -- same-quarter "
              f"models are more functionally similar to each other than to other quarters' models, "
              f"by a margin that isn't just a precise estimate of a near-zero gap.")
    elif test["p_value"] < 0.05:
        print(f"-> Statistically significant but the effect size is negligible (d={d:.2f}, within "
              f"{test['mean_within_quarter_cka']:.4f} vs. between {test['mean_between_quarter_cka']:.4f} "
              f"-- agree to 3 decimal places). With this many probes the permutation test has power to "
              f"detect a gap this small; treat 'p<0.05' here as 'nonzero', not 'meaningful'. This does "
              f"NOT support a real seasonal difference in the FC head's function.")
    else:
        print("-> No detectable difference between within-quarter and between-quarter functional "
              "similarity -- on this probe set, no evidence the FC head's function differs by "
              "season.")

    if test["min_pairwise_cka"] >= 0.95:
        print(f"NOTE: every pairwise CKA in this run is >= {test['min_pairwise_cka']:.3f}, including "
              f"across quarters -- all these heads look nearly identical on this synthetic probe set. "
              f"That's consistent with a real finding (the heads genuinely converge to similar "
              f"functions regardless of season), but it's equally consistent with the synthetic probe "
              f"set (standard-normal vectors, not real backbone features) not stressing real "
              f"functional differences -- see this script's docstring. If this result is surprising, "
              f"the next step is swapping in real backbone features extracted from actual imagery as "
              f"the probe set, not trusting a near-ceiling CKA score on its own.")

    pd.DataFrame([test]).to_csv(out_dir / "quarter_similarity_test.csv", index=False)

    meta_path = out_dir / "fc_function_meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path} and {out_dir / 'quarter_similarity_test.csv'}")

    title_suffix = f"{', '.join(s.upper() for s in states)}, {args.variable}"
    plot_cka_heatmap(cka_df, meta, f"FC head functional similarity (linear CKA)\n{title_suffix}",
                      str(out_dir / "cka_heatmap.png"))
    plot_mds_embedding(meta, cka, f"MDS of FC head functional (dis)similarity\n{title_suffix}",
                        str(out_dir / "mds_fc_function.png"), args.seed)


if __name__ == "__main__":
    main()
