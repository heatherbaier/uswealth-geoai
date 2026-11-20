"""
Per-layer/stage breakdown of analyze_weight_drift.py's season-clustering
test: is season clustering concentrated in EARLY layers (patch
embedding, stage 1) -- meaning the model mostly adapted at the input/
low-level-feature stage to each season's different spectral
characteristics -- or does it hold or strengthen in LATE layers
(stage 3/4, the final norm) and the head -- the closest these weights
alone can get you to "the backbone-to-wealth MAPPING differs by
season," not just how it reads the raw input?

Reuses analyze_weight_drift.py's checkpoint discovery, pretrained-
backbone construction helpers, cosine_similarity_matrix(), and
clustering_test() directly (same model-level permutation null, same
reasoning for why it's needed -- model pairs aren't independent
observations) -- this script only adds the layer-by-layer split on top
of what that one already does for the whole backbone at once.

Layer groups (derived from the ACTUAL state_dict keys present in the
pretrained backbone, not a hardcoded torchvision-version-specific layer
count -- see discover_block_to_group()): patch_embed (features.0.*),
stage1..stage4 (features.<N>.* for that stage's own transformer blocks,
with the small merge/downsample block immediately BEFORE each stage
folded into it, since it's a tiny transition layer not worth its own
bucket), final_norm (norm.* -- the LayerNorm after the last stage), and
head.

The head is treated differently from every other group, and this
matters for reading its results: it is NOT a pretrained-minus-finetuned
delta. It's a freshly, randomly initialized regression layer at the
start of every training run (see analyze_weight_drift.py's docstring),
and sail's training pipeline doesn't fix a seed for model-weight
initialization -- checked directly: sail/src/sail/core/builders.py only
fixes the DATASET split's seed, nothing for the model's own init -- so
there's no reproducible "initial head" to subtract. This uses the RAW
finetuned head weights directly instead (head.0+head.3 concatenated,
same convention as extract_fc_weights_tsne.py) for cosine similarity.
Read that group's result with real caution in BOTH directions: a
256-unit hidden layer has no canonical ordering across independently-
trained networks, so two models computing the IDENTICAL function can
still have very different raw head.0 weight vectors just from hidden
units landing in a different order/rotation during training -- the
exact concern analyze_fc_function_similarity.py's own docstring raises
about this same kind of weight-space comparison. A WEAK head signal
here does NOT rule out "the mapping differs" the way it would for the
backbone's delta-based groups (it could just be that permutation/
rotation noise swamping a real signal); a STRONG one despite that noise
is still meaningful.

For each group: cosine similarity matrix among that group's own
vectors (delta for the backbone groups, raw for the head), then the
same season/year permutation-based clustering test analyze_weight_
drift.py runs on the whole backbone at once.

Output:
    <out-dir>/<state>_<variable>_weight_clustering_by_layer.csv   one
        row per (layer_group, clustering_by in {season, year}):
        observed gap, null mean/95th pct, p-value, n_params, and
        is_delta (False only for "head", see above)
    <fig-dir>/<state>_<variable>_season_clustering_by_layer.png   bar
        chart of the season-clustering gap per layer group, in
        anatomical order (patch_embed -> stage1 -> ... -> stage4 ->
        final_norm -> head) -- read left-to-right for the input-
        adaptation-vs-mapping-differs question this was built to answer

Usage:
    python 3_analysis/analyze_weight_drift_by_layer.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019
"""

import argparse
import gc
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_fc_function_similarity import build_backbone  # noqa: E402
from analyze_weight_drift import (  # noqa: E402
    collect_checkpoints, backbone_state, load_checkpoint_backbone_state,
    cosine_similarity_matrix, clustering_test,
)

FEATURE_BLOCK_RE = re.compile(r"^features\.(\d+)\.")
ANATOMICAL_ORDER = ["patch_embed", "stage1", "stage2", "stage3", "stage4", "final_norm", "head"]


def discover_block_to_group(keys) -> dict:
    """Maps each features.<N> block index to a stage-group name, derived
    from whichever indices are ACTUALLY present in the pretrained
    backbone's own state_dict (not a hardcoded layer count, in case the
    installed torchvision version numbers swin_v2_t's internal blocks
    differently). Block 0 is always the patch embed; each odd index
    afterward starts a new stage, and the even index right before it (a
    small merge/downsample block between stages) folds into that next
    stage rather than getting its own bucket."""
    indices = sorted({int(m.group(1)) for k in keys for m in [FEATURE_BLOCK_RE.match(k)] if m})
    block_to_group = {}
    current_stage = 0
    for idx in indices:
        if idx == 0:
            block_to_group[idx] = "patch_embed"
        elif idx % 2 == 1:
            current_stage = (idx + 1) // 2
            block_to_group[idx] = f"stage{current_stage}"
        else:
            block_to_group[idx] = f"stage{current_stage + 1}"
    return block_to_group


def assign_group(key: str, block_to_group: dict) -> str:
    m = FEATURE_BLOCK_RE.match(key)
    if m:
        return block_to_group.get(int(m.group(1)), "other")
    if key.startswith("norm."):
        return "final_norm"
    return "other"


def group_keys(keys, block_to_group: dict) -> dict:
    """{group_name: [sorted keys in that group]} -- sorted for a
    deterministic flatten order every time. Any key that matches neither
    features.<N>.* nor norm.* is dropped into "other" and printed as a
    warning rather than silently included or excluded -- backbone_state()
    should already have removed head.* before this is called, so "other"
    showing up at all would mean an unexpected architecture change."""
    groups = {}
    for k in keys:
        groups.setdefault(assign_group(k, block_to_group), []).append(k)
    for name in groups:
        groups[name] = sorted(groups[name])
    unexpected = groups.pop("other", [])
    if unexpected:
        print(f"  WARNING: {len(unexpected)} backbone key(s) didn't match any known layer group, "
              f"dropped from the per-layer breakdown: {unexpected[:5]}")
    return groups


def flatten_group_diff(finetuned_state: dict, pretrained_state: dict, keys: list) -> np.ndarray:
    parts = []
    for k in keys:
        ft = finetuned_state[k].detach().cpu().numpy().reshape(-1).astype(np.float32)
        pt = pretrained_state[k].detach().cpu().numpy().reshape(-1).astype(np.float32)
        if ft.shape != pt.shape:
            raise SystemExit(f"shape mismatch at {k!r}: checkpoint {ft.shape} vs pretrained {pt.shape}")
        parts.append(ft - pt)
    return np.concatenate(parts)


def flatten_group_raw(state: dict, keys: list) -> np.ndarray:
    return np.concatenate([state[k].detach().cpu().numpy().reshape(-1).astype(np.float32) for k in keys])


def load_checkpoint_head_state(ckpt_path: str) -> dict:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    head_state = {k: v for k, v in state_dict.items() if k.startswith("head")}
    if not head_state:
        raise SystemExit(f"{ckpt_path}: no head.* keys found in checkpoint")
    return head_state


def collect_group_vectors(meta: pd.DataFrame, pretrained_state: dict, backbone_groups: dict) -> dict:
    """{group_name: (n_models, n_params) array}, including "head" -- each
    checkpoint file is read ONCE and reused for every group (backbone
    delta groups + the raw head), not re-opened per group."""
    group_vectors = {g: [] for g in backbone_groups}
    group_vectors["head"] = []
    for row in meta.itertuples():
        finetuned_state = load_checkpoint_backbone_state(row.ckpt_path)
        for group_name, keys in backbone_groups.items():
            group_vectors[group_name].append(flatten_group_diff(finetuned_state, pretrained_state, keys))
        head_state = load_checkpoint_head_state(row.ckpt_path)
        group_vectors["head"].append(flatten_group_raw(head_state, sorted(head_state.keys())))
        del finetuned_state, head_state
        gc.collect()
        print(f"  [{row.year} Q{row.quarter}] loaded")
    return {g: np.stack(v) for g, v in group_vectors.items()}


def plot_by_layer(cluster_df: pd.DataFrame, title: str, out_path: Path):
    season_df = cluster_df[cluster_df.clustering_by == "season"].copy()
    order = [g for g in ANATOMICAL_ORDER if g in season_df["layer_group"].values]
    season_df["layer_group"] = pd.Categorical(season_df["layer_group"], categories=order, ordered=True)
    season_df = season_df.sort_values("layer_group")

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#1baf7a" if p < 0.05 else "#52514e" for p in season_df["p_value"]]
    ax.bar(season_df["layer_group"].astype(str), season_df["observed_gap"], color=colors)
    ax.axhline(0, color="#52514e", linewidth=0.8)
    ax.axhline(season_df["null_95pct"].mean(), color="#eb6834", linestyle="--", linewidth=1,
               label="mean null 95th pct across groups")
    ax.set_ylabel("same-season minus different-season\nmean cosine similarity")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")
    for i, (gap, pv) in enumerate(zip(season_df["observed_gap"], season_df["p_value"])):
        ax.annotate(f"p={pv:.3f}", (i, gap), textcoords="offset points",
                    xytext=(0, 4 if gap >= 0 else -14), ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def print_interpretation_guide(cluster_df: pd.DataFrame):
    season = cluster_df[cluster_df.clustering_by == "season"].set_index("layer_group")
    early = [g for g in ["patch_embed", "stage1"] if g in season.index]
    late = [g for g in ["stage3", "stage4", "final_norm", "head"] if g in season.index]
    early_sig = [g for g in early if season.loc[g, "p_value"] < 0.05]
    late_sig = [g for g in late if season.loc[g, "p_value"] < 0.05]

    print("\nInterpretation guide (your own call to make, not an automated verdict):")
    print(f"  Early (patch_embed/stage1) significant at p<0.05: {early_sig or 'none'}")
    print(f"  Late (stage3/stage4/final_norm/head) significant at p<0.05: {late_sig or 'none'}")
    if late_sig and not early_sig:
        print("  -> signal concentrated in LATE layers/head: the closest these weights get to "
              "\"the mapping differs by season\", not just input adaptation.")
    elif early_sig and not late_sig:
        print("  -> signal concentrated in EARLY layers: looks like input adaptation to each "
              "season's spectral characteristics, not a different learned mapping.")
    elif early_sig and late_sig:
        print("  -> both early AND late layers show clustering -- some input adaptation and "
              "some mapping difference both appear to be present; compare the gap sizes above "
              "to judge which is stronger.")
    else:
        print("  -> neither early nor late layers show significant clustering at this per-layer "
              "split -- if the whole-backbone test (analyze_weight_drift.py) was significant, "
              "the signal may be diffuse across many layers rather than concentrated anywhere "
              "specific.")


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
    p.add_argument("--n-perms", type=int, default=999, help="Permutations for the clustering test")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift/", type=Path)
    p.add_argument("--fig-dir", default="4_analysis/outputs/weight_drift/figures/", type=Path)
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

    print("\nBuilding pretrained backbone and discovering layer groups...")
    pretrained_net = build_backbone(in_channels, device="cpu")
    pretrained_state = backbone_state(pretrained_net)
    block_to_group = discover_block_to_group(pretrained_state.keys())
    backbone_groups = group_keys(pretrained_state.keys(), block_to_group)
    print(f"Backbone layer groups (n_params): "
          f"{ {g: sum(pretrained_state[k].numel() for k in ks) for g, ks in backbone_groups.items()} }")

    print("\nLoading every checkpoint once, building per-layer delta/raw vectors:")
    group_vectors = collect_group_vectors(meta, pretrained_state, backbone_groups)
    del pretrained_net, pretrained_state
    gc.collect()

    print("\nClustering test per layer group (999 perms, model-level label shuffle):")
    cluster_rows = []
    for group_name in ANATOMICAL_ORDER:
        if group_name not in group_vectors:
            continue
        vectors = group_vectors[group_name]
        sim_matrix = cosine_similarity_matrix(vectors)
        for label_name, label_vals in [("season", meta["quarter"].to_numpy()), ("year", meta["year"].to_numpy())]:
            result = clustering_test(sim_matrix, label_vals, args.n_perms, args.seed)
            result.update({"layer_group": group_name, "clustering_by": label_name,
                           "n_params": vectors.shape[1], "is_delta": group_name != "head"})
            cluster_rows.append(result)
            print(f"  [{group_name} / {label_name}] gap={result['observed_gap']:+.4f} vs. null mean="
                  f"{result['null_mean']:+.4f} -> p={result['p_value']:.4f} (n_params={vectors.shape[1]:,})")

    cluster_df = pd.DataFrame(cluster_rows)
    state_variable = f"{state}_{args.variable}"
    out_path = args.out_dir / f"{state_variable}_weight_clustering_by_layer.csv"
    cluster_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    plot_by_layer(cluster_df, f"Season clustering by layer -- {state.upper()} {args.variable}",
                  args.fig_dir / f"{state_variable}_season_clustering_by_layer.png")
    print_interpretation_guide(cluster_df)


if __name__ == "__main__":
    main()
