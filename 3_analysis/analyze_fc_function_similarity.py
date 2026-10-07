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

Probe set: --probe-source synthetic (default) uses standard-normal
vectors in the backbone's 768-dim feature space -- no torchvision/real
imagery needed, isolates the HEAD's function specifically, at the cost of
probing regions of feature space real Swin features may never actually
visit. --probe-source real instead extracts REAL backbone features from
actual chips sampled across the whole --states/--years/--quarters grid,
run through a FROZEN pretrained swin_v2_t backbone (ImageNet weights,
with the stem conv rebuilt for the real band count the same way sail's
own SwinRegressor.build() does -- see build_backbone()) -- deliberately
NOT fine-tuned on any model's own task, so the probe distribution isn't
itself biased toward one season's backbone. This needs torchvision +
rasterio + band_mean/band_std filled in for every --states entry in
pipeline_configs/state_registry.yml (there's no other source of truth
for those in this project), and is substantially more expensive per
probe (a real backbone forward pass vs. a matrix lookup) -- consider a
smaller --n-probes (e.g. 200-500) unless running on a GPU node.

For whether any apparent clustering is more than chance, this runs a
permutation test directly on the pairwise CKA matrix (Mantel-test style):
the observed (mean within-group CKA) - (mean between-group CKA) gap,
against a null built by reshuffling group labels across the SAME
similarity matrix many times. --group-by {quarter,year} (default quarter)
picks which column defines the groups -- e.g. a state with a visually
spread-out CKA heatmap but a null "by quarter" result may still have a
real difference that's just organized by YEAR instead of season; rerun
with --group-by year to check that directly.

--cka-matrix/--meta: skip checkpoint discovery and the (expensive,
especially with --probe-source real) activation-collection step entirely
and rerun the significance test + plots directly against a previously
saved cka_matrix.csv + fc_function_meta.csv from an earlier run in the
same --out-dir. Both must be given together. This is the cheap way to
re-check a result under a different --group-by without recomputing
anything.

Usage:
    python 3_analysis/analyze_fc_function_similarity.py \
        --states az ga oh pa --years 2016 2017 2018 2019 --quarters 1 2 3 4 \
        --variable wealth_index --out-dir ./out_fc_function_similarity

    # re-check an existing result grouped by year instead of quarter,
    # without rerunning probe extraction:
    python 3_analysis/analyze_fc_function_similarity.py \
        --cka-matrix ./out_fc_function_similarity/cka_matrix.csv \
        --meta ./out_fc_function_similarity/fc_function_meta.csv \
        --group-by year --out-dir ./out_fc_function_similarity
"""

import argparse
import itertools
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import MDS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_quarter_r2_matrix import load_registry, resolve_state_settings  # noqa: E402
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


# ---------------------------------------------------------------------
# --probe-source real: extract real backbone features instead of using
# synthetic Gaussian vectors. Heavier (torchvision + rasterio + real
# imagery + band stats), so kept separate from the lightweight synthetic
# path above -- these imports stay local to the functions that need them
# so --probe-source synthetic never requires torchvision/rasterio.
# ---------------------------------------------------------------------
def resolve_data_root(state: str, year: int, quarter: int, args) -> str:
    """<data_root>/ for this (state, year, quarter) -- NOT the checkpoint
    dir. Chips live under <data_root>/chips/, independent of whether a
    trained model even exists for that cell (a probe image only needs to
    have been downloaded, not trained on)."""
    if args.use_state_registry:
        registry = load_registry()
        st = resolve_state_settings(state, registry)
        data_root = st["data_root_template"].format(state=state, year=year, quarter=quarter)
    else:
        data_root = args.data_root_template.format(state=state, year=year, quarter=quarter)
    return data_root if data_root.endswith("/") else data_root + "/"


def resolve_band_stats(state: str, registry: dict):
    """band_mean/band_std always come from state_registry.yml regardless
    of --use-state-registry (that flag only controls CHECKPOINT
    discovery) -- there's no other source of truth for them in this
    project."""
    entry = registry.get(state, {})
    band_mean, band_std = entry.get("band_mean"), entry.get("band_std")
    if band_mean is None or band_std is None:
        raise SystemExit(f"--probe-source real needs band_mean/band_std for {state!r} in "
                          f"pipeline_configs/state_registry.yml (run compute_shared_band_stats.py "
                          f"for it first) -- not filled in yet.")
    return band_mean, band_std


def sample_probe_chips(states, years, quarters, args, n_probes: int, seed: int):
    """Real chip paths (path, state), sampled across every (state, year,
    quarter) cell in the analysis grid -- the same quarters being
    compared, not one fixed reference quarter, so the probe distribution
    reflects the real diversity of inputs these specific models actually
    see rather than being anchored to a single season's idiosyncrasies."""
    import glob
    cells = list(itertools.product(states, years, quarters))
    rng = random.Random(seed)
    per_cell = max(1, n_probes // len(cells))
    chips = []
    for state, year, quarter in cells:
        chips_dir = os.path.join(resolve_data_root(state, year, quarter, args), "chips")
        available = sorted(glob.glob(os.path.join(chips_dir, "*.tif")))
        if not available:
            print(f"[probe chips] {state} Q{quarter} {year}: no chips under {chips_dir}, skipping")
            continue
        sample = available if len(available) <= per_cell else rng.sample(available, per_cell)
        chips.extend((p, state) for p in sample)
        print(f"[probe chips] {state} Q{quarter} {year}: sampled {len(sample)} / {len(available)} chips")
    if not chips:
        raise SystemExit("No probe chips found across the whole --states/--years/--quarters grid "
                          "-- check that imagery has actually been downloaded for these cells.")
    return chips


def build_backbone(in_channels: int, device: str):
    """torchvision's pretrained swin_v2_t (ImageNet weights only -- NOT
    fine-tuned on any model's own task), stem conv rebuilt for in_channels
    bands via sail's own warm-start logic (_expand_patch_embed_conv),
    head replaced with Identity so forward() returns exactly the pooled
    pre-head feature vector every checkpoint's head.0 actually receives
    in production (torchvision's SwinTransformer.forward does
    features -> norm -> permute -> avgpool -> flatten -> head, so
    Identity in place of head gives the flattened pre-head vector
    directly, no extra reshaping needed). Deliberately NOT fine-tuned on
    any checkpoint's own weights: a neutral, season-agnostic feature
    extractor, so the probe distribution isn't itself biased toward
    whichever season's backbone it came from."""
    import torch.nn as nn
    import torchvision as tv
    from sail.models.swin import _expand_patch_embed_conv

    backbone = tv.models.swin_v2_t(weights="IMAGENET1K_V1")
    if in_channels != 3:
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = _expand_patch_embed_conv(old_conv, in_channels)
    backbone.head = nn.Identity()
    backbone.eval()
    return backbone.to(device)


def load_chip_tensor(path: str, img_size, band_mean, band_std, scale_divisor: float = 10000.0):
    """Exact eval-time preprocessing sail's own dataloader uses for a
    TIFF chip (see sail/src/sail/data/adapters.py::_load_tiff_tensor +
    _TiffAugment.__call__ with train=False: resize then normalize, no
    augmentation). Reimplemented directly here rather than importing
    sail's dataset classes, since those pull in a lot more (coords.json/
    ys.json resolution, splitting, etc.) than loading one chip needs."""
    import rasterio
    import torchvision.transforms.functional as TF
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")
    img = torch.from_numpy(arr) / scale_divisor
    img = TF.resize(img, list(img_size))
    img = TF.normalize(img, mean=band_mean, std=band_std)
    return img


def extract_real_probe_features(chips, band_stats_by_state, in_channels: int, img_size, device: str,
                                 batch_size: int) -> np.ndarray:
    """chips: list of (path, state). Returns (n_probes, 768) real
    backbone features -- computed ONCE and shared across every
    checkpoint's head comparison, the same role build_probe_set() plays
    for the synthetic-probe path."""
    backbone = build_backbone(in_channels, device)
    all_features = []
    with torch.no_grad():
        for i in range(0, len(chips), batch_size):
            batch = chips[i:i + batch_size]
            tensors = [load_chip_tensor(path, img_size, *band_stats_by_state[state])
                       for path, state in batch]
            batch_tensor = torch.stack(tensors).to(device)
            feats = backbone(batch_tensor)
            all_features.append(feats.cpu().numpy())
            print(f"[probe features] {min(i + batch_size, len(chips))}/{len(chips)}")
    return np.concatenate(all_features, axis=0).astype(np.float32)


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


def group_similarity_test(cka: np.ndarray, groups: np.ndarray, n_permutations: int, seed: int):
    """Mantel-style permutation test: is the observed gap between mean
    within-group and mean between-group CKA bigger than chance, given
    this exact similarity matrix? `groups` can be quarters, years, or any
    other per-checkpoint label -- see --group-by. Reshuffles group LABELS
    (not the matrix) each draw, so the null respects the real similarity
    structure -- only whether this grouping happens to explain any of it.

    Also reports a standardized effect size (gap / pooled SD of the two
    groups, i.e. Cohen's d) alongside the p-value. With --n-probes in the
    thousands, CKA estimates are precise enough that the permutation test
    can call a genuinely tiny gap "significant" (p<0.05 with within/
    between means that agree to 3 decimal places is a real failure mode,
    not hypothetical -- see this script's own first real run). p<0.05
    answers "is the gap exactly zero"; effect size answers "does the gap
    matter". Report both, trust neither alone."""
    n = len(groups)
    iu = np.triu_indices(n, k=1)
    pairs = cka[iu]

    same_mask = groups[iu[0]] == groups[iu[1]]
    same_vals, diff_vals = pairs[same_mask], pairs[~same_mask]

    # With too few checkpoints per group (e.g. exactly 1 per quarter, as
    # in a single-year run), there may be literally ZERO within-group
    # PAIRS to compare at all -- not a null result, a sample-size problem.
    # Catch this explicitly rather than letting it fall through to NaN
    # means / NaN p-value, which main() would otherwise misreport as "no
    # detectable difference" (implying a real negative finding).
    if len(same_vals) == 0 or len(diff_vals) == 0:
        return {
            "observed_gap": float("nan"),
            "mean_within_group_cka": float(same_vals.mean()) if len(same_vals) else float("nan"),
            "mean_between_group_cka": float(diff_vals.mean()) if len(diff_vals) else float("nan"),
            "effect_size_cohens_d": float("nan"),
            "min_pairwise_cka": float(pairs.min()) if len(pairs) else float("nan"),
            "p_value": float("nan"),
            "n_permutations": 0,
            "insufficient_data": True,
        }

    def gap_for(labels):
        same = labels[iu[0]] == labels[iu[1]]
        if same.sum() == 0 or (~same).sum() == 0:
            return np.nan
        return pairs[same].mean() - pairs[~same].mean()

    observed = gap_for(groups)
    rng = np.random.default_rng(seed)
    null = np.array([gap_for(rng.permutation(groups)) for _ in range(n_permutations)])
    null = null[~np.isnan(null)]
    p_value = float(np.mean(null >= observed)) if len(null) else float("nan")

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
        "mean_within_group_cka": float(same_vals.mean()),
        "mean_between_group_cka": float(diff_vals.mean()),
        "effect_size_cohens_d": effect_size,
        "min_pairwise_cka": float(pairs.min()),
        "p_value": p_value,
        "n_permutations": len(null),
        "insufficient_data": False,
    }


# Dataviz-skill-validated categorical palette (hue + marker shape, so a
# 5th+ category is still distinguishable without relying on hue alone).
# QUARTER_STYLE uses the first four of these; non-quarter groupings (e.g.
# --group-by year) reuse the same palette via assign_group_styles below
# so every grouping gets a consistent, colorblind-aware look.
GROUP_PALETTE = [
    ("#2a78d6", "o"), ("#eb6834", "s"), ("#1baf7a", "^"), ("#eda100", "D"),
    ("#e87ba4", "v"), ("#4a3aa7", "P"), ("#e34948", "X"), ("#008300", "*"),
]


def assign_group_styles(values) -> dict:
    """Map each distinct value (sorted) to a {color, marker, label} style,
    cycling GROUP_PALETTE if there are more than 8 distinct values."""
    styles = {}
    for i, v in enumerate(sorted(values)):
        color, marker = GROUP_PALETTE[i % len(GROUP_PALETTE)]
        styles[v] = {"color": color, "marker": marker, "label": str(v)}
    return styles


def plot_cka_heatmap(cka: pd.DataFrame, meta: pd.DataFrame, title: str, out_path: str,
                      group_col: str = "quarter"):
    order = meta.sort_values([group_col, "state", "year", "quarter"]).index.to_numpy()
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

    # Mark group-block boundaries so same-group submatrices are easy to
    # compare against the off-block (cross-group) regions by eye.
    groups_ordered = meta.loc[order, group_col].to_numpy()
    boundaries = np.where(np.diff(groups_ordered) != 0)[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="white", lw=1.2)
        ax.axvline(b, color="white", lw=1.2)

    fig.colorbar(im, ax=ax, label="Linear CKA (0 = unrelated, 1 = identical function)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_mds_embedding(meta: pd.DataFrame, cka: np.ndarray, title: str, out_path: str, seed: int,
                        group_col: str = "quarter"):
    distance = np.clip(1 - cka, 0, None)
    np.fill_diagonal(distance, 0)
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=seed,
              normalized_stress="auto")
    coords = mds.fit_transform(distance)

    # Preserve exact prior behavior for the default grouping; reuse the
    # same palette (via assign_group_styles) for any other --group-by so
    # every grouping gets a consistent, colorblind-aware look.
    if group_col == "quarter":
        styles = QUARTER_STYLE
    else:
        styles = assign_group_styles(meta[group_col].unique())

    fig, ax = plt.subplots(figsize=(8, 7))
    for g in sorted(meta[group_col].unique()):
        style = styles.get(g, {"color": "#52514e", "marker": "x", "label": str(g)})
        idx = (meta[group_col] == g).to_numpy()
        ax.scatter(coords[idx, 0], coords[idx, 1], c=style["color"], marker=style["marker"],
                   s=70, edgecolors="white", linewidths=0.6, label=style["label"], alpha=0.9)
    if len(meta) <= 40:
        # Label points with whichever identifying info ISN'T already
        # encoded by color: quarter grouping colors by season, so label
        # by state+year; any other grouping colors by that column, so
        # label by state+quarter instead.
        if group_col == "quarter":
            point_label = lambda row: f"{row.state}{row.year}"
        else:
            point_label = lambda row: f"{row.state}Q{row.quarter}"
        for (x, y), row in zip(coords, meta.itertuples()):
            ax.annotate(point_label(row), (x, y), fontsize=7, color="#52514e",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    ax.set_title(title)
    ax.legend(title=group_col.capitalize(), frameon=False)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", help="Required unless --cka-matrix/--meta are given")
    p.add_argument("--years", nargs="+", type=int, help="Required unless --cka-matrix/--meta are given")
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--variable", help="Required unless --cka-matrix/--meta are given")
    p.add_argument("--group-by", choices=["quarter", "year"], default="quarter",
                    help="Grouping column for the significance test/plots (default: quarter). "
                         "A visually spread-out CKA heatmap with a null 'by quarter' result may "
                         "still have a real difference organized by year instead -- rerun with "
                         "--group-by year (ideally via --cka-matrix/--meta, see below) to check.")
    p.add_argument("--cka-matrix", default=None,
                    help="Path to a previously saved cka_matrix.csv. Must be given together with "
                         "--meta. Skips checkpoint discovery and activation collection entirely and "
                         "reruns the significance test + plots directly against these saved results "
                         "-- the cheap way to re-check a result under a different --group-by.")
    p.add_argument("--meta", default=None,
                    help="Path to a previously saved fc_function_meta.csv, matching --cka-matrix.")
    p.add_argument("--version", default=None, help="Trained version per checkpoint (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--probe-source", choices=["synthetic", "real"], default="synthetic",
                    help="'synthetic' (default): standard-normal probe vectors, no torchvision/real "
                         "imagery needed. 'real': extracts real backbone features from actual chips "
                         "across the --states/--years/--quarters grid through a frozen pretrained "
                         "swin_v2_t backbone -- see this script's docstring for the full tradeoff.")
    p.add_argument("--n-probes", type=int, default=2000,
                    help="Probe count (default 2000; with --probe-source real, each one is a real "
                         "backbone forward pass -- consider 200-500 unless on a GPU node)")
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256],
                    help="Only used with --probe-source real (default: 256 256, matching sail's own configs)")
    p.add_argument("--probe-device", default="cpu", help="Only used with --probe-source real")
    p.add_argument("--probe-batch-size", type=int, default=32, help="Only used with --probe-source real")
    p.add_argument("--n-permutations", type=int, default=2000,
                    help="Permutations for the quarter-similarity significance test (default 2000)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="Default: ./out_fc_function_similarity")
    args = p.parse_args()

    out_dir = Path(args.out_dir or "./out_fc_function_similarity")
    out_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.cka_matrix) != bool(args.meta):
        raise SystemExit("--cka-matrix and --meta must be given together.")

    if args.cka_matrix:
        # Reanalyze mode: skip checkpoint discovery and the (expensive)
        # activation-collection step entirely, reusing a previously saved
        # result -- e.g. to re-check --group-by year against a matrix
        # that was only ever computed once.
        cka_df = pd.read_csv(args.cka_matrix, index_col=0)
        meta = pd.read_csv(args.meta)
        if len(cka_df) != len(meta):
            raise SystemExit(f"--cka-matrix has {len(cka_df)} row(s) but --meta has {len(meta)} "
                              f"row(s) -- they must come from the same run.")
        cka = cka_df.values
        states = sorted(meta["state"].unique())
        print(f"Reanalyze mode: loaded {len(meta)} checkpoint(s) from {args.cka_matrix} and {args.meta} "
              f"(skipping checkpoint discovery/activation collection)")
    else:
        if not (args.states and args.years and args.variable):
            raise SystemExit("--states, --years, and --variable are required unless --cka-matrix/--meta "
                              "are given.")
        registry = load_registry() if args.use_state_registry else None
        states = [s.lower() for s in args.states]
        cka, meta = _collect_and_build_cka(states, args, registry, out_dir)

    group_similarity(cka, meta, args, out_dir, states)


def _collect_and_build_cka(states, args, registry, out_dir):
    # Peek at the first checkpoint found to learn in_features (768 for
    # swin_v2_t today, but this way a different backbone just works) and,
    # for --probe-source real, in_channels (needed to rebuild the probe
    # backbone's stem conv to match) -- both read straight from the raw
    # checkpoint rather than hardcoded.
    probe_in_features = None
    probe_in_channels = None
    for state in states:
        ckpt_dir_fn = resolve_ckpt_dir_fn(state, args.variable, args, registry)
        for year, quarter in itertools.product(args.years, args.quarters):
            try:
                ckpt_dir = ckpt_dir_fn(year, quarter)
            except SystemExit:
                continue
            _, ckpt_path = find_checkpoint(ckpt_dir)
            if ckpt_path is not None:
                raw_ckpt = torch.load(ckpt_path, map_location="cpu")
                probe_in_features = raw_ckpt["state_dict"]["head.0.weight"].shape[1]
                probe_in_channels = raw_ckpt.get("in_channels", 3)
                break
        if probe_in_features is not None:
            break
    if probe_in_features is None:
        raise SystemExit("No checkpoints found -- can't even determine the probe dimensionality.")

    if args.probe_source == "real":
        band_registry = load_registry()
        band_stats_by_state = {state: resolve_band_stats(state, band_registry) for state in states}
        chips = sample_probe_chips(states, args.years, args.quarters, args, args.n_probes, args.seed)
        probes = extract_real_probe_features(chips, band_stats_by_state, probe_in_channels,
                                              args.img_size, args.probe_device, args.probe_batch_size)
        print(f"\nProbe set: {probes.shape[0]} real backbone feature(s), {probes.shape[1]}-dim "
              f"(in_channels={probe_in_channels}, extracted via a frozen pretrained swin_v2_t)")
        if probes.shape[1] != probe_in_features:
            raise SystemExit(f"Real probe features are {probes.shape[1]}-dim but checkpoints expect "
                              f"{probe_in_features}-dim input -- architecture mismatch.")
    else:
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

    meta_path = out_dir / "fc_function_meta.csv"
    meta.to_csv(meta_path, index=False)
    print(f"Wrote {meta_path}")

    return cka, meta


def group_similarity(cka: np.ndarray, meta: pd.DataFrame, args, out_dir: Path, states: list):
    group_col = args.group_by
    groups = meta[group_col].to_numpy()
    test = group_similarity_test(cka, groups, args.n_permutations, args.seed)
    test["group_by"] = group_col
    cka_df = pd.DataFrame(cka)  # positional only; row order matches meta
    variable_label = args.variable or "?"
    title_suffix = f"{', '.join(s.upper() for s in states)}, {variable_label}"
    test_path = out_dir / "group_similarity_test.csv"

    if test["insufficient_data"]:
        print(f"\nCan't run the {group_col}-similarity test: no within-{group_col} OR no "
              f"between-{group_col} pairs exist in this set of {len(meta)} checkpoint(s) (need at "
              f"least 2 checkpoints in the SAME {group_col} to form a within-{group_col} pair). This "
              f"is a sample-size problem, not a null result -- add more data to get repeat "
              f"checkpoints per {group_col}.")
        pd.DataFrame([test]).to_csv(test_path, index=False)
        print(f"Wrote {test_path}")
        plot_cka_heatmap(cka_df, meta, f"FC head functional similarity (linear CKA)\n{title_suffix}",
                          str(out_dir / "cka_heatmap.png"), group_col=group_col)
        return

    print(f"\nMean within-{group_col} CKA:  {test['mean_within_group_cka']:.4f}")
    print(f"Mean between-{group_col} CKA: {test['mean_between_group_cka']:.4f}")
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
        print(f"-> Statistically significant AND a {size_word} effect size (d={d:.2f}) -- same-{group_col} "
              f"models are more functionally similar to each other than to other {group_col}s' models, "
              f"by a margin that isn't just a precise estimate of a near-zero gap.")
    elif test["p_value"] < 0.05:
        print(f"-> Statistically significant but the effect size is negligible (d={d:.2f}, within "
              f"{test['mean_within_group_cka']:.4f} vs. between {test['mean_between_group_cka']:.4f} "
              f"-- agree to 3 decimal places). With this many probes the permutation test has power to "
              f"detect a gap this small; treat 'p<0.05' here as 'nonzero', not 'meaningful'. This does "
              f"NOT support a real difference in the FC head's function organized by {group_col}.")
    else:
        print(f"-> No detectable difference between within-{group_col} and between-{group_col} "
              f"functional similarity -- on this probe set, no evidence the FC head's function "
              f"differs by {group_col}.")

    if test["min_pairwise_cka"] >= 0.95:
        print(f"NOTE: every pairwise CKA in this run is >= {test['min_pairwise_cka']:.3f}, including "
              f"across {group_col}s -- all these heads look nearly identical on this probe set. "
              f"That's consistent with a real finding (the heads genuinely converge to similar "
              f"functions regardless of {group_col}), but it's equally consistent with the probe "
              f"set not stressing real functional differences -- see this script's docstring. If "
              f"this result is surprising, the next step is swapping in real backbone features "
              f"(--probe-source real) as the probe set, not trusting a near-ceiling CKA score on "
              f"its own.")

    pd.DataFrame([test]).to_csv(test_path, index=False)
    print(f"Wrote {test_path}")

    plot_cka_heatmap(cka_df, meta, f"FC head functional similarity (linear CKA)\n{title_suffix}",
                      str(out_dir / "cka_heatmap.png"), group_col=group_col)
    plot_mds_embedding(meta, cka, f"MDS of FC head functional (dis)similarity\n{title_suffix}",
                        str(out_dir / "mds_fc_function.png"), args.seed, group_col=group_col)


if __name__ == "__main__":
    main()
