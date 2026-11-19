"""
Extract each trained model's FC (fully-connected) head weights and run
t-SNE across them, to see whether a season's (quarter's) models cluster
apart from other seasons' -- i.e. whether the learned mapping from
backbone features to wealth index actually differs by season, not just
how well it predicts.

What "the FC weights" means here: sail's SwinRegressor (sail/src/sail/
models/swin.py) replaces torchvision's swin_v2_t classification head with

    nn.Sequential(
        nn.Linear(in_features, 256),   # "head.0" in the saved state_dict
        nn.ReLU(),                     # "head.1" -- no weights
        nn.Dropout(0.3),               # "head.2" -- no weights
        nn.Linear(256, 1),             # "head.3" in the saved state_dict
    )

This is the only task-specific, non-pretrained part of the network --
everything upstream (features.*) is the frozen-at-init, then fine-tuned,
Swin Transformer backbone. "head.0"/"head.3" are what actually map the
backbone's learned features to a wealth-index scalar, so they're the
right place to look for "does the function differ by season" rather than
"does the backbone extract different textures by season" (a different,
noisier question this isn't trying to answer).

Reads checkpoints directly with torch.load() and pulls out only the
requested "head.*" keys from state_dict (default: both layers) --
does NOT build the full SwinRegressor/torchvision backbone (no
torchvision import, no pretrained-ImageNet-weights download, no need to
reconstruct in_channels/model_kwargs correctly), since the FC head's raw
tensors are all this needs. See sail/src/sail/models/swin.py::save() for
the exact checkpoint format this expects: {"state_dict": net.state_dict(),
"num_classes": ..., "in_channels": ...}.

Each checkpoint's flattened+concatenated FC weights become one row of a
(n_models x n_features) matrix. n_features is in the hundreds of
thousands (256*768 + 256 + 256 + 1 for both head layers on swin_v2_t) and
n_models is realistically a few dozen at most -- an extreme n << d case,
so this PCA-reduces to --pca-components dimensions (default: capped well
below n_models) before t-SNE, standard practice for exactly this
situation and NOT optional here the way it sometimes is for denser data
(skip it with --no-pca only to see why it matters).

Checkpoint discovery mirrors analyze_transfer_penalty_landcover.py /
analyze_cross_quarter_significance.py's dual approach: --data-root-template
(default: every state's actual on-disk convention, no state_registry.yml
entry needed) or --use-state-registry. Unlike those scripts, this reads
the MODEL's own checkpoint directory directly (model_epoch<N>.torch files
written by sail's Trainer), not a cross_quarter/ preds CSV -- the highest
epoch found in that directory is used as "the trained model" per
(state, year, quarter).

Usage:
    python 3_analysis/extract_fc_weights_tsne.py \
        --states az ga oh pa --years 2016 2017 2018 2019 --quarters 1 2 3 4 \
        --variable wealth_index --out-dir ./out_fc_weights_tsne

    # Only the final regression layer (256 -> 1), not the full head:
    python 3_analysis/extract_fc_weights_tsne.py \
        --states oh --years 2017 2018 2019 --quarters 1 2 3 4 \
        --variable wealth_index --layers head.3 --out-dir ./out_fc_weights_tsne
"""

import argparse
import itertools
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_quarter_r2_matrix import model_ckpt_dir, load_registry  # noqa: E402
from analyze_transfer_penalty_landcover import (  # noqa: E402
    template_ckpt_dir, DEFAULT_DATA_ROOT_TEMPLATE,
)

DEFAULT_LAYERS = ["head.0", "head.3"]

# Categorical color + marker per quarter -- a scatter plot puts every
# point pair on screen together (dataviz skill's "--pairs all" case), so
# only the palette's first 3 of 8 validated-for-scatter slots are safe on
# hue alone; a 4th category (quarters are a fixed set of 4, nothing to
# fold into "Other") gets a distinct MARKER too so identity is never
# color-alone, same mitigation the skill names for a palette slot that
# only clears the floor with secondary encoding.
QUARTER_STYLE = {
    1: {"color": "#2a78d6", "marker": "o", "label": "Q1"},  # blue
    2: {"color": "#eb6834", "marker": "s", "label": "Q2"},  # orange
    3: {"color": "#1baf7a", "marker": "^", "label": "Q3"},  # aqua
    4: {"color": "#eda100", "marker": "D", "label": "Q4"},  # yellow
}


def find_checkpoint(ckpt_dir: str):
    """Highest-epoch model_epoch<N>.torch directly under ckpt_dir (NOT
    cross_quarter/ -- that's cross-quarter validate preds, not a trained
    checkpoint). Same filename convention as sail's own
    highest_epoch()/Trainer, reimplemented here so this script only needs
    torch (no torchvision/full sail import chain) to run."""
    if not os.path.isdir(ckpt_dir):
        return None, None
    pat = re.compile(r"^model_epoch(\d+)\.torch$")
    best = None
    for name in os.listdir(ckpt_dir):
        m = pat.match(name)
        if m:
            epoch = int(m.group(1))
            if best is None or epoch > best[0]:
                best = (epoch, os.path.join(ckpt_dir, name))
    return best if best else (None, None)


def load_fc_vector(ckpt_path: str, layers: list):
    """Flattens + concatenates weight and bias for each requested
    "head.N" layer, in the order given, into one 1-D vector."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    parts = []
    shapes = {}
    for layer in layers:
        w_key, b_key = f"{layer}.weight", f"{layer}.bias"
        if w_key not in state_dict or b_key not in state_dict:
            head_keys = sorted(k for k in state_dict if k.startswith("head"))
            raise SystemExit(f"{ckpt_path}: expected keys {w_key!r}/{b_key!r} not in checkpoint "
                              f"state_dict (head.* keys present: {head_keys})")
        w = state_dict[w_key].detach().cpu().numpy().reshape(-1)
        b = state_dict[b_key].detach().cpu().numpy().reshape(-1)
        parts.append(w)
        parts.append(b)
        shapes[layer] = {"weight": tuple(state_dict[w_key].shape), "bias": tuple(state_dict[b_key].shape)}
    return np.concatenate(parts).astype(np.float32), shapes


def resolve_ckpt_dir_fn(state, variable, args, registry):
    if args.use_state_registry:
        return lambda year, quarter: model_ckpt_dir(state, variable, year, quarter, args.version, registry)
    return lambda year, quarter: template_ckpt_dir(state, variable, year, quarter, args.version,
                                                     args.data_root_template)


def collect_fc_vectors(states, years, quarters, variable, layers, args, registry):
    rows = []
    vectors = []
    expected_len = None
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

            vector, shapes = load_fc_vector(ckpt_path, layers)
            if expected_len is None:
                expected_len = len(vector)
                print(f"FC vector length: {expected_len} (layers={layers}, shapes={shapes})")
            elif len(vector) != expected_len:
                print(f"[{state} Q{quarter} {year}] WARNING: FC vector length {len(vector)} != "
                      f"{expected_len} from earlier checkpoints (different architecture/in_channels?) "
                      f"-- skipping so it doesn't corrupt the matrix")
                continue

            rows.append({"state": state, "year": year, "quarter": quarter, "epoch": epoch,
                         "ckpt_path": ckpt_path})
            vectors.append(vector)
            print(f"[{state} Q{quarter} {year}] epoch {epoch}: {ckpt_path}")

    if not vectors:
        raise SystemExit("No checkpoints found -- nothing to run t-SNE on.")
    return pd.DataFrame(rows), np.stack(vectors)


def run_tsne(matrix: np.ndarray, pca_components, perplexity, seed):
    n_samples = matrix.shape[0]
    if n_samples < 4:
        raise SystemExit(f"Only {n_samples} checkpoint(s) found -- t-SNE needs at least 4 "
                          f"points to be meaningful.")

    reduced = matrix
    if pca_components:
        n_components = min(pca_components, n_samples - 1, matrix.shape[1])
        pca = PCA(n_components=n_components, random_state=seed)
        reduced = pca.fit_transform(matrix)
        explained = float(pca.explained_variance_ratio_.sum())
        print(f"PCA: {matrix.shape[1]} -> {n_components} dims "
              f"({explained * 100:.1f}% of variance retained) before t-SNE")

    clamped_perplexity = min(perplexity, max(1, n_samples - 1))
    if clamped_perplexity != perplexity:
        print(f"--perplexity {perplexity} >= n_samples ({n_samples}) -- clamped to "
              f"{clamped_perplexity}")

    tsne = TSNE(n_components=2, perplexity=clamped_perplexity, random_state=seed, init="pca")
    embedding = tsne.fit_transform(reduced)
    return embedding


def plot_embedding(meta: pd.DataFrame, embedding: np.ndarray, title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 7))

    for q in sorted(meta["quarter"].unique()):
        style = QUARTER_STYLE.get(q, {"color": "#52514e", "marker": "x", "label": f"Q{q}"})
        idx = (meta["quarter"] == q).to_numpy()
        ax.scatter(embedding[idx, 0], embedding[idx, 1], c=style["color"], marker=style["marker"],
                   s=70, edgecolors="white", linewidths=0.6, label=style["label"], alpha=0.9)

    # Direct point labels only when there aren't too many to read --
    # beyond that the legend (color+marker, already present) carries
    # quarter identity and this just adds state/year for the curious.
    if len(meta) <= 40:
        for (x, y), row in zip(embedding, meta.itertuples()):
            ax.annotate(f"{row.state}{row.year}", (x, y), fontsize=7, color="#52514e",
                        xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
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
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS,
                    help=f"Which head sublayer(s) to extract, concatenated in order given "
                         f"(default: {DEFAULT_LAYERS}, the full head). Use e.g. --layers head.3 "
                         f"for just the final 256->1 regression layer.")
    p.add_argument("--version", default=None, help="Trained version per checkpoint (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--pca-components", type=int, default=50,
                    help="PCA-reduce to this many dims before t-SNE (default 50; auto-capped to "
                         "n_checkpoints-1). Pass 0 to disable (not recommended -- see docstring).")
    p.add_argument("--perplexity", type=float, default=30.0,
                    help="t-SNE perplexity (default 30, auto-clamped below n_checkpoints)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out-dir", default=None, help="Default: ./out_fc_weights_tsne")
    args = p.parse_args()

    out_dir = Path(args.out_dir or "./out_fc_weights_tsne")
    out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry() if args.use_state_registry else None
    states = [s.lower() for s in args.states]

    meta, matrix = collect_fc_vectors(states, args.years, args.quarters, args.variable,
                                       args.layers, args, registry)
    print(f"\nCollected {matrix.shape[0]} checkpoint(s), {matrix.shape[1]} FC weight(s) each")

    meta_path = out_dir / "fc_weights_meta.csv"
    meta.to_csv(meta_path, index=False)
    weights_path = out_dir / "fc_weights.npz"
    np.savez_compressed(weights_path, weights=matrix)
    print(f"Wrote {meta_path} and {weights_path} (raw weight matrix, not CSV -- "
          f"{matrix.nbytes / 1e6:.1f} MB uncompressed)")

    embedding = run_tsne(matrix, args.pca_components or None, args.perplexity, args.seed)
    meta = meta.copy()
    meta["tsne_x"] = embedding[:, 0]
    meta["tsne_y"] = embedding[:, 1]
    embedding_path = out_dir / "tsne_embedding.csv"
    meta.to_csv(embedding_path, index=False)
    print(f"Wrote {embedding_path}")

    layers_label = "+".join(args.layers)
    title = f"t-SNE of FC head weights ({layers_label})\n{', '.join(s.upper() for s in states)}, {args.variable}"
    plot_embedding(meta, embedding, title, str(out_dir / "tsne_fc_weights.png"))


if __name__ == "__main__":
    main()
