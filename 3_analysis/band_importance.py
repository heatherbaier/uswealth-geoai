"""
Spectral band importance for every trained wealth model, to test whether
the function mapping imagery -> wealth differs by season -- not just in
accuracy (see build_cross_quarter_r2_matrix.py), but in WHICH bands the
model actually relies on (e.g. red-edge/NIR in leaf-on quarters vs.
visible/SWIR in leaf-off quarters).

Reuses sail's own `task: band_importance` primitives directly (sail/src/
sail/explain/band_importance.py: _load_full_dataset, _predict_all,
permute_band, _resolve_group_indices) and its dataset/model loading
(sail.core.builders.build_model/build_validation_dataset/highest_epoch)
rather than reimplementing them -- the in-memory cfg dict built below has
the exact same shape generate_validate_config.py writes to YAML, so the
test set loaded here is byte-for-byte sail's own stable/spatial-split
test set for that model. No sail code changes, no YAML files written.

DIAGONAL (eval quarter == model's own quarter): dataset.new=False, which
intersects this quarter's items with THIS model's own test_indices.txt --
exactly sail's normal task: validate restriction.

OFF-DIAGONAL (--offdiag, eval quarter != model's own quarter, same
state): dataset.new=False pointed at a DIFFERENT quarter's imagery --
the exact same "restrict to the model's own held-out GEOIDs, which exist
in every quarter of this state" mechanism generate_cross_validate_
config.py already established for cross-quarter R^2 (see that script's
docstring for why this is leak-free). Off-diagonal cells compute GROUPED
permutation importance only (not single-band or mean-ablation) -- that's
the one number needed to see which band groups lose their usefulness
when a model is moved to another season, at a fraction of the cost of
every method on every cell.

Band order (12 channels, "_allbands" convention) is the authoritative
list from pipeline_configs/generate_download_config.py -- NOT guessed:
    B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
Groups (every band covered exactly once):
    visible=B02,B03,B04   red_edge=B05,B06,B07   nir=B08,B8A
    swir=B11,B12          atmospheric=B01,B09

Methods:
  permutation       Shuffle one band across test tracts, n_repeats times
                     (different seed each repeat) -- long format keeps
                     EVERY repeat as its own row (unlike sail's own
                     compute_band_importance, which only keeps the
                     mean+std -- this needs per-repeat rows).
  group_permutation Same, but shuffling a whole named group together
                     (one shared permutation across the group's bands) --
                     correlated bands can understate their importance
                     individually; this removes the fallback.
  mean_ablation     Deterministic secondary check: replace a band/group
                     with its mean over THIS eval quarter's loaded test
                     tracts (already-normalized values, since images are
                     normalized at load time) and re-score once (repeat=0).

Sign convention: delta_r2 = r2_base - r2_pert, delta_mae = mae_pert -
mae_base -- both positive = "this band/group mattered".

Caching: every (model_year, model_quarter, eval_year, eval_quarter)
cell's loaded test tensors + baseline predictions/metrics are cached to
--cache-dir, so a rerun (e.g. adding a method or a repeat) doesn't reload
imagery or redo the baseline forward pass. Skipped automatically unless
--overwrite.

A missing checkpoint or missing imagery for any (model, eval) cell is
skipped with a clear message, not a crash -- the --years/--quarters grid
is a wishlist, not a guarantee every cell exists yet.

Usage:
    # Full sweep, diagonal only:
    python 3_analysis/band_importance.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019 --quarters 1 2 3 4

    # Also off-diagonal (grouped-only) cells:
    python 3_analysis/band_importance.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019 --quarters 1 2 3 4 --offdiag

    # Sanity checks on one model before running everything (see this
    # script's docstring / the original request for exactly what these
    # check):
    python 3_analysis/band_importance.py --state oh --variable wealth_index \
        --years 2018 --quarters 1 --sanity-check --expected-r2 0.57
"""

import argparse
import itertools
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_quarter_r2_matrix import model_ckpt_dir, imagery_prefix  # noqa: E402
from analyze_transfer_penalty_landcover import (  # noqa: E402
    template_ckpt_dir, template_imagery_prefix,
    DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)

from sail.core.builders import build_model, build_validation_dataset, highest_epoch  # noqa: E402
from sail.explain.band_importance import (  # noqa: E402
    _load_full_dataset, _predict_all, permute_band, _resolve_group_indices,
)

# Authoritative 12-band order -- pipeline_configs/generate_download_config.py,
# NOT guessed. Every band importance channel index below is relative to this.
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B11", "B12"]

BAND_GROUPS = {
    "visible": ["B02", "B03", "B04"],
    "red_edge": ["B05", "B06", "B07"],
    "nir": ["B08", "B8A"],
    "swir": ["B11", "B12"],
    "atmospheric": ["B01", "B09"],
}

METHOD_CHOICES = ["permutation", "group_permutation", "mean_ablation"]


# -------------------------------------------------------------------------
# Checkpoint/imagery discovery -- same dual registry/template convention as
# analyze_cross_quarter_significance.py, imported rather than reimplemented.
# -------------------------------------------------------------------------
def resolve_ckpt_and_prefix(state, variable, args, registry):
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


def resolve_data_root_fn(args, registry):
    """<data_root>/ for a given (year, quarter) -- same convention as
    resolve_ckpt_and_prefix, independent of it since band_importance needs
    the imagery ROOT (not just its ckpt_dir/prefix) to build dataset cfg."""
    if args.use_state_registry:
        def fn(state, year, quarter):
            st = resolve_state_settings(state, registry)
            root = st["data_root_template"].format(state=state, year=year, quarter=quarter)
            return root if root.endswith("/") else root + "/"
        return fn

    def fn(state, year, quarter):
        root = args.data_root_template.format(state=state, year=year, quarter=quarter)
        return root if root.endswith("/") else root + "/"
    return fn


# -------------------------------------------------------------------------
# Model loading
# -------------------------------------------------------------------------
def load_model_wrapper(ckpt_dir: str, in_channels: int, device: str):
    model_wrapper, _, _ = build_model({"name": "swin", "params": {"in_channels": in_channels}})
    epoch, path = highest_epoch(ckpt_dir)
    if path is None:
        raise FileNotFoundError(f"No model_epoch*.torch checkpoint under {ckpt_dir}")
    model_wrapper.load(path)
    model_wrapper.net = model_wrapper.net.to(device).eval()
    return model_wrapper, epoch, path


# -------------------------------------------------------------------------
# Dataset loading + caching
# -------------------------------------------------------------------------
def cache_path(cache_dir: Path, state, variable, model_year, model_quarter, eval_year, eval_quarter):
    name = f"{state}_{variable}_model{model_year}q{model_quarter}_eval{eval_year}q{eval_quarter}.pt"
    return cache_dir / name


def load_eval_cell(state, variable, model_ckpt_dir_path, model_year, model_quarter,
                    eval_year, eval_quarter, data_root_fn, prefix_fn, band_mean, band_std,
                    img_size, batch_size, cache_dir, overwrite):
    """Returns (images, coords, labels, n_test) for this (model, eval
    quarter) cell -- the model's OWN held-out GEOIDs, as they appear in
    eval_year/eval_quarter's imagery (see module docstring). Cached to
    disk keyed by the full (model, eval) pair, since the restriction is a
    function of the MODEL's own test_indices.txt."""
    cpath = cache_path(cache_dir, state, variable, model_year, model_quarter, eval_year, eval_quarter)
    if cpath.exists() and not overwrite:
        d = torch.load(cpath, weights_only=False)
        return d["images"], d["coords"], d["labels"]

    data_root = data_root_fn(state, eval_year, eval_quarter)
    prefix = prefix_fn(eval_year, eval_quarter)
    cfg_dataset = {
        "data_root": data_root, "prefix": prefix, "img_size": list(img_size),
        "num_workers": 0, "seed": 1337, "temporal": False, "new": False,
        "band_mean": band_mean, "band_std": band_std,
    }
    cfg = {"dataset": cfg_dataset}
    ds = build_validation_dataset(cfg_dataset, cfg, model_ckpt_dir_path, temporal=False)
    images, coords, labels, _ = _load_full_dataset(ds, batch_size)

    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"images": images, "coords": coords, "labels": labels}, cpath)
    return images, coords, labels


def baseline_cache_path(cache_dir: Path, state, variable, model_year, model_quarter, eval_year, eval_quarter):
    name = f"{state}_{variable}_model{model_year}q{model_quarter}_eval{eval_year}q{eval_quarter}_baseline.pt"
    return cache_dir / name


def get_baseline(model_wrapper, images, coords, labels, device, batch_size,
                  cache_dir, state, variable, model_year, model_quarter, eval_year, eval_quarter, overwrite):
    bpath = baseline_cache_path(cache_dir, state, variable, model_year, model_quarter, eval_year, eval_quarter)
    if bpath.exists() and not overwrite:
        d = torch.load(bpath, weights_only=False)
        return d["preds"], d["r2"], d["mae"], d["r"]

    preds = _predict_all(model_wrapper, images, coords, device, batch_size).numpy()
    r2, mae, r = compute_metrics(preds, labels.numpy())
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"preds": preds, "r2": r2, "mae": mae, "r": r}, bpath)
    return preds, r2, mae, r


# -------------------------------------------------------------------------
# Metrics
# -------------------------------------------------------------------------
def compute_metrics(preds: np.ndarray, labels: np.ndarray):
    r2 = float(r2_score(labels, preds))
    mae = float(mean_absolute_error(labels, preds))
    if np.ptp(preds) == 0 or np.ptp(labels) == 0:
        r = float("nan")
    else:
        r = float(pearsonr(preds, labels)[0])
    return r2, mae, r


# -------------------------------------------------------------------------
# Importance methods -- long-format rows, exact schema from the spec
# -------------------------------------------------------------------------
def _row(context, method, unit, feature, repeat, r2_base, r2p, mae_base, maep, r_base, rp, n_test):
    return {
        **context, "method": method, "unit": unit, "feature": feature, "repeat": repeat,
        "r2_base": r2_base, "r2_pert": r2p, "delta_r2": r2_base - r2p,
        "mae_base": mae_base, "mae_pert": maep, "delta_mae": maep - mae_base,
        "r_base": r_base, "r_pert": rp, "n_test": n_test,
    }


def permutation_rows(model_wrapper, images, coords, labels, device, batch_size,
                      context, r2_base, mae_base, r_base, n_test, n_repeats, seed, log):
    rows = []
    labels_np = labels.numpy()
    for b, name in enumerate(BAND_NAMES):
        rng = np.random.RandomState(seed)
        deltas = []
        for rep in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(images.shape[0]))
            permuted = permute_band(images, b, perm)
            preds = _predict_all(model_wrapper, permuted, coords, device, batch_size).numpy()
            r2p, maep, rp = compute_metrics(preds, labels_np)
            rows.append(_row(context, "permutation", "band", name, rep,
                              r2_base, r2p, mae_base, maep, r_base, rp, n_test))
            deltas.append(r2_base - r2p)
        log(f"    [permutation band {name}] mean delta_r2={np.mean(deltas):+.4f}")
    return rows


def group_permutation_rows(model_wrapper, images, coords, labels, device, batch_size,
                            context, r2_base, mae_base, r_base, n_test, n_repeats, seed, log):
    rows = []
    labels_np = labels.numpy()
    group_idx = _resolve_group_indices(BAND_GROUPS, BAND_NAMES)
    for group_name, idxs in group_idx.items():
        rng = np.random.RandomState(seed)
        deltas = []
        for rep in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(images.shape[0]))
            permuted = permute_band(images, idxs, perm)
            preds = _predict_all(model_wrapper, permuted, coords, device, batch_size).numpy()
            r2p, maep, rp = compute_metrics(preds, labels_np)
            rows.append(_row(context, "group_permutation", "group", group_name, rep,
                              r2_base, r2p, mae_base, maep, r_base, rp, n_test))
            deltas.append(r2_base - r2p)
        log(f"    [group_permutation {group_name}] mean delta_r2={np.mean(deltas):+.4f}")
    return rows


def mean_ablation_rows(model_wrapper, images, coords, labels, device, batch_size,
                        context, r2_base, mae_base, r_base, n_test, log):
    rows = []
    labels_np = labels.numpy()
    band_means = images.mean(dim=0, keepdim=True)  # per-band mean over THIS eval cell's test tracts

    def _ablate(idxs, unit, feature):
        ablated = images.clone()
        for b in idxs:
            ablated[:, b] = band_means[:, b]
        preds = _predict_all(model_wrapper, ablated, coords, device, batch_size).numpy()
        r2p, maep, rp = compute_metrics(preds, labels_np)
        rows.append(_row(context, "mean_ablation", unit, feature, 0,
                          r2_base, r2p, mae_base, maep, r_base, rp, n_test))
        log(f"    [mean_ablation {feature}] delta_r2={r2_base - r2p:+.4f}")

    for b, name in enumerate(BAND_NAMES):
        _ablate([b], "band", name)
    group_idx = _resolve_group_indices(BAND_GROUPS, BAND_NAMES)
    for group_name, idxs in group_idx.items():
        _ablate(idxs, "group", group_name)
    return rows


# -------------------------------------------------------------------------
# Per-model driver
# -------------------------------------------------------------------------
def run_model(state, variable, model_year, model_quarter, args, registry,
              ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
              cache_dir, log):
    try:
        ckpt_dir = ckpt_dir_fn(model_year, model_quarter)
    except SystemExit as e:
        log(f"[{state} Q{model_quarter} {model_year}] no trained model found, skipping: {e}")
        return []

    try:
        model_wrapper, epoch, ckpt_path = load_model_wrapper(ckpt_dir, in_channels, args.device)
    except FileNotFoundError as e:
        log(f"[{state} Q{model_quarter} {model_year}] {e}, skipping")
        return []
    log(f"[{state} Q{model_quarter} {model_year}] loaded epoch {epoch}: {ckpt_path}")

    eval_cells = [(model_year, model_quarter)]
    if args.offdiag:
        eval_cells += [(y, q) for y, q in itertools.product(args.years, args.quarters)
                       if (y, q) != (model_year, model_quarter)]

    all_rows = []
    for eval_year, eval_quarter in eval_cells:
        is_diag = (eval_year, eval_quarter) == (model_year, model_quarter)
        try:
            images, coords, labels = load_eval_cell(
                state, variable, ckpt_dir, model_year, model_quarter, eval_year, eval_quarter,
                data_root_fn, prefix_fn, band_mean, band_std, args.img_size, args.batch_size,
                cache_dir, args.overwrite,
            )
        except Exception as e:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] couldn't load this cell, skipping: {e}")
            continue

        n_test = images.shape[0]
        if n_test == 0:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] 0 overlapping test tracts, skipping")
            continue

        _, r2_base, mae_base, r_base = get_baseline(
            model_wrapper, images, coords, labels, args.device, args.batch_size,
            cache_dir, state, variable, model_year, model_quarter, eval_year, eval_quarter, args.overwrite,
        )
        tag = "diagonal" if is_diag else "OFF-DIAGONAL"
        log(f"  [eval {state} Q{eval_quarter} {eval_year}] ({tag}) n_test={n_test} "
            f"r2_base={r2_base:.4f} mae_base={mae_base:.4f} r_base={r_base:.4f}")

        context = {"state": state, "variable": variable, "model_year": model_year,
                   "model_quarter": model_quarter, "eval_year": eval_year, "eval_quarter": eval_quarter}

        if not is_diag:
            # Off-diagonal: grouped permutation importance only -- see
            # module docstring. Runs regardless of --methods (that flag
            # only controls which methods run on the DIAGONAL cell).
            all_rows += group_permutation_rows(
                model_wrapper, images, coords, labels, args.device, args.batch_size,
                context, r2_base, mae_base, r_base, n_test, args.n_repeats, args.seed, log,
            )
            continue

        if "permutation" in args.methods:
            all_rows += permutation_rows(
                model_wrapper, images, coords, labels, args.device, args.batch_size,
                context, r2_base, mae_base, r_base, n_test, args.n_repeats, args.seed, log,
            )
        if "group_permutation" in args.methods:
            all_rows += group_permutation_rows(
                model_wrapper, images, coords, labels, args.device, args.batch_size,
                context, r2_base, mae_base, r_base, n_test, args.n_repeats, args.seed, log,
            )
        if "mean_ablation" in args.methods:
            all_rows += mean_ablation_rows(
                model_wrapper, images, coords, labels, args.device, args.batch_size,
                context, r2_base, mae_base, r_base, n_test, log,
            )

    return all_rows


# -------------------------------------------------------------------------
# Summary (mean/SD across repeats + normalized importance)
# -------------------------------------------------------------------------
def build_summary(panel: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["state", "variable", "model_year", "model_quarter", "eval_year", "eval_quarter",
                  "method", "unit", "feature"]
    summary = panel.groupby(group_cols, as_index=False).agg(
        delta_r2_mean=("delta_r2", "mean"), delta_r2_std=("delta_r2", "std"),
        delta_mae_mean=("delta_mae", "mean"), delta_mae_std=("delta_mae", "std"),
        r2_base=("r2_base", "first"), mae_base=("mae_base", "first"), r_base=("r_base", "first"),
        n_test=("n_test", "first"), n_repeats=("repeat", "nunique"),
    )

    # Normalized importance: per (model, eval, method, unit) cell, divide
    # each feature's delta_r2_mean by the sum of POSITIVE delta_r2_mean
    # across that cell's features -- lets profiles be compared across
    # models with different baseline R^2 (see module docstring).
    norm_keys = ["state", "variable", "model_year", "model_quarter", "eval_year", "eval_quarter",
                 "method", "unit"]

    def _normalize(g):
        pos_sum = g["delta_r2_mean"].clip(lower=0).sum()
        g["normalized_importance"] = g["delta_r2_mean"] / pos_sum if pos_sum > 0 else np.nan
        return g

    summary = summary.groupby(norm_keys, group_keys=False).apply(_normalize)
    return summary


# -------------------------------------------------------------------------
# Sanity-check mode
# -------------------------------------------------------------------------
def run_sanity_check(state, variable, model_year, model_quarter, args, registry,
                      ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
                      cache_dir, log):
    ckpt_dir = ckpt_dir_fn(model_year, model_quarter)
    model_wrapper, epoch, ckpt_path = load_model_wrapper(ckpt_dir, in_channels, args.device)
    log(f"[sanity-check] {state} Q{model_quarter} {model_year}: loaded epoch {epoch}: {ckpt_path}")

    images, coords, labels = load_eval_cell(
        state, variable, ckpt_dir, model_year, model_quarter, model_year, model_quarter,
        data_root_fn, prefix_fn, band_mean, band_std, args.img_size, args.batch_size,
        cache_dir, args.overwrite,
    )
    n_test = images.shape[0]
    preds_base, r2_base, mae_base, r_base = get_baseline(
        model_wrapper, images, coords, labels, args.device, args.batch_size,
        cache_dir, state, variable, model_year, model_quarter, model_year, model_quarter, args.overwrite,
    )

    print(f"\n{'=' * 70}\nSANITY CHECK 1: baseline matches the known cross-quarter diagonal?\n{'=' * 70}")
    print(f"n_test={n_test}  r2_base={r2_base:.4f}  mae_base={mae_base:.4f}  r_base={r_base:.4f}")
    if args.expected_r2 is not None:
        diff = abs(r2_base - args.expected_r2)
        status = "PASS" if diff < 0.02 else "FAIL"
        print(f"  --expected-r2 {args.expected_r2:.4f}  |diff|={diff:.4f}  -> {status}")
        if status == "FAIL":
            print("  STOP: baseline R^2 doesn't match the expected cross-quarter diagonal value. "
                  "Don't trust anything downstream until this is resolved -- check that the "
                  "checkpoint/version/dataset resolved here are really the same run that "
                  "produced the expected number (--version, --use-state-registry, data root).")

    print(f"\n{'=' * 70}\nSANITY CHECK 2: permuting nothing == baseline; permuting all 12 bands "
          f"drives R^2 toward 0 or below\n{'=' * 70}")
    preds_identity = _predict_all(model_wrapper, images, coords, args.device, args.batch_size).numpy()
    max_abs_diff = float(np.max(np.abs(preds_identity - preds_base)))
    print(f"  permute-nothing max|pred diff| vs baseline: {max_abs_diff:.8f} "
          f"-> {'PASS' if max_abs_diff < 1e-5 else 'FAIL'}")

    rng = np.random.RandomState(args.seed)
    perm = torch.from_numpy(rng.permutation(images.shape[0]))
    all_bands_permuted = permute_band(images, list(range(len(BAND_NAMES))), perm)
    preds_all_perm = _predict_all(model_wrapper, all_bands_permuted, coords, args.device, args.batch_size).numpy()
    r2_all_perm, _, _ = compute_metrics(preds_all_perm, labels.numpy())
    print(f"  all-12-bands-permuted R^2: {r2_all_perm:.4f} "
          f"-> {'PASS (near 0 or negative)' if r2_all_perm < 0.15 else 'WARN (higher than expected)'}")

    print(f"\n{'=' * 70}\nSANITY CHECK 3: atmospheric group (B01, B09) reliance\n{'=' * 70}")
    group_idx = _resolve_group_indices(BAND_GROUPS, BAND_NAMES)
    atmo_idx = group_idx["atmospheric"]
    rng = np.random.RandomState(args.seed)
    deltas = []
    for _ in range(args.n_repeats):
        perm = torch.from_numpy(rng.permutation(images.shape[0]))
        permuted = permute_band(images, atmo_idx, perm)
        preds = _predict_all(model_wrapper, permuted, coords, args.device, args.batch_size).numpy()
        r2p, _, _ = compute_metrics(preds, labels.numpy())
        deltas.append(r2_base - r2p)
    mean_delta = float(np.mean(deltas))
    print(f"  atmospheric group delta_r2 (mean over {args.n_repeats} repeats): {mean_delta:+.4f}")
    if mean_delta > 0.05:
        print(f"  FLAG: atmospheric group (B01=coastal aerosol, B09=water vapor) reliance is "
              f"large ({mean_delta:+.4f}). B09 is strongly seasonal on its own (water vapor), "
              f"so a large chunk of this model's apparent 'seasonal function' could be "
              f"atmospheric rather than landscape -- worth a closer look before trusting the "
              f"grouped seasonality results.")
    else:
        print(f"  Atmospheric group reliance looks small ({mean_delta:+.4f}) -- not flagged.")


# -------------------------------------------------------------------------
# main
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", required=True, nargs="+", type=int)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--methods", nargs="+", choices=METHOD_CHOICES, default=METHOD_CHOICES,
                    help="Methods to run on the DIAGONAL cell (default: all three). Off-diagonal "
                         "cells (--offdiag) always run group_permutation only, regardless of this.")
    p.add_argument("--n-repeats", type=int, default=5)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--offdiag", action="store_true",
                    help="Also evaluate every model against every OTHER (year, quarter) in the "
                         "--years/--quarters grid (grouped permutation only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--version", default=None, help="Trained version per model (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Discover checkpoints via pipeline_configs/state_registry.yml instead of "
                         "--data-root-template/--base-prefix-template")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE,
                    help=f"Default: {DEFAULT_BASE_PREFIX_TEMPLATE!r}")
    p.add_argument("--out-dir", default="3_analysis/outputs/band_importance")
    p.add_argument("--cache-dir", default=None,
                    help="Default: <out-dir>/_cache")
    p.add_argument("--overwrite", action="store_true",
                    help="Recompute everything for this run instead of reusing cached "
                         "tensors/baselines or skipping rows already in the output CSV")
    p.add_argument("--sanity-check", action="store_true",
                    help="Run only the 3 sanity checks against the FIRST (year, quarter) in "
                         "--years/--quarters, print PASS/FAIL, and exit -- no CSV written")
    p.add_argument("--expected-r2", type=float, default=None,
                    help="--sanity-check only: compare the baseline R^2 against this value")
    args = p.parse_args()

    state = args.state.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else out_dir / "_cache"

    def log(msg):
        print(msg, flush=True)

    log("Band channel index -> name mapping (pipeline_configs/generate_download_config.py):")
    for i, name in enumerate(BAND_NAMES):
        log(f"  [{i}] {name}")
    log("Band groups:")
    for group_name, members in BAND_GROUPS.items():
        log(f"  {group_name}: {members}")

    registry = load_registry()
    band_registry_entry = resolve_state_settings(state, registry)
    band_mean, band_std, in_channels = (band_registry_entry["band_mean"], band_registry_entry["band_std"],
                                         band_registry_entry["in_channels"])
    if in_channels != len(BAND_NAMES):
        raise SystemExit(f"state_registry.yml says {state} has in_channels={in_channels}, but "
                          f"BAND_NAMES has {len(BAND_NAMES)} entries -- this script assumes the "
                          f"12-band _allbands convention throughout, not a different band count.")

    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    if args.sanity_check:
        run_sanity_check(state, args.variable, args.years[0], args.quarters[0], args, registry,
                          ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
                          cache_dir, log)
        return

    csv_path = out_dir / f"{state}_{args.variable}_band_importance.csv"
    existing = pd.read_csv(csv_path) if (csv_path.exists() and not args.overwrite) else None

    # A model is skippable only if the CSV already has EVERY (eval cell,
    # method) pair THIS run would produce for it -- not just "something"
    # for that model. Otherwise a rerun that adds --offdiag or a new
    # --methods entry would silently skip the model and never compute the
    # newly-requested rows. "Mostly done" still means "recompute fully";
    # this trades a little repeated work for never shipping an
    # incomplete-but-skipped cell.
    def expected_keys(model_year, model_quarter):
        diag_methods = set(args.methods)
        keys = {(model_year, model_quarter, model_year, model_quarter, m) for m in diag_methods}
        if args.offdiag:
            for y, q in itertools.product(args.years, args.quarters):
                if (y, q) != (model_year, model_quarter):
                    keys.add((model_year, model_quarter, y, q, "group_permutation"))
        return keys

    all_rows = []
    for model_year, model_quarter in itertools.product(args.years, args.quarters):
        if existing is not None:
            already = existing[(existing.model_year == model_year) & (existing.model_quarter == model_quarter)]
            have_keys = set(already[["model_year", "model_quarter", "eval_year", "eval_quarter", "method"]]
                             .drop_duplicates().itertuples(index=False, name=None))
            if expected_keys(model_year, model_quarter) <= have_keys:
                log(f"[{state} Q{model_quarter} {model_year}] already fully in {csv_path}, skipping "
                    f"(pass --overwrite to redo)")
                all_rows.append(already)
                continue
        rows = run_model(state, args.variable, model_year, model_quarter, args, registry,
                          ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
                          cache_dir, log)
        if rows:
            all_rows.append(pd.DataFrame(rows))

    if not all_rows:
        raise SystemExit("No model produced any rows -- nothing to write.")
    panel = pd.concat(all_rows, ignore_index=True)
    panel.to_csv(csv_path, index=False)
    log(f"\nWrote {csv_path} ({len(panel):,} rows)")

    summary = build_summary(panel)
    summary_path = out_dir / f"{state}_{args.variable}_band_importance_summary.csv"
    summary.to_csv(summary_path, index=False)
    log(f"Wrote {summary_path} ({len(summary):,} rows)")


if __name__ == "__main__":
    main()
