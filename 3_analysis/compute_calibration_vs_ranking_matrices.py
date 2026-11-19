"""
For every (model_year, model_quarter) x (eval_year, eval_quarter) cell
across --years (default 2017-2019), for EVERY state in --states, FOUR
matrices per state:

  1. R^2               Raw predictions vs. true labels, on the model's
                        own held-out test tracts, as they appear in the
                        eval quarter's imagery -- same leak-free
                        mechanism build_cross_quarter_r2_matrix.py/
                        band_importance.py use.
  2. Pearson r          Same predictions/labels, linear correlation
                        only -- insensitive to a pure scale/offset
                        miscalibration, so this isolates RANKING quality
                        from CALIBRATION the way raw R^2 can't (R^2
                        penalizes a correct ranking that's just shifted
                        or rescaled exactly as harshly as a genuinely
                        scrambled one).
  3. R^2 recalibrated   Fit a 1-D linear recalibration (label ~ a*pred +
                        b, ordinary least squares) on the EVAL quarter's
                        OWN TRAINING tracts -- the complement of ITS
                        held-out test set, NOT the model's own training
                        quarter -- using the SAME model's predictions on
                        those training tracts. Apply (a, b) to the
                        model's test predictions and recompute R^2.
                        Answers "if the model's underlying ranking is
                        basically fine but its OUTPUT SCALE is wrong for
                        this quarter, does a cheap post-hoc linear fix
                        on a handful of in-domain examples recover most
                        of the lost R^2?"
  4. Bias (mean signed)  mean(pred - label) over the SAME test tracts
                        R^2/Pearson r use (raw predictions, no
                        recalibration) -- a cell can have a terrible R^2
                        purely from DIRECTION (every prediction too high
                        or too low by about the same amount) rather than
                        a ranking failure; this number says which way and
                        by how much, distinct from R^2/r (which are both
                        blind to the sign/magnitude of a consistent
                        offset) and from recal_intercept (that's the
                        offset a linear fit WOULD apply, fit on training
                        tracts; bias is the offset actually observed on
                        the test tracts, no fitting involved).

Together, these four give you the calibration-vs-ranking split
everywhere: high Pearson r + high recalibrated R^2 + low raw R^2 means a
pure calibration problem (fixable cheaply, no architecture change or
retraining needed -- the model already "knows" relative wealth, it's
just reporting it on the wrong scale for this quarter). Low Pearson r
AND low recalibrated R^2 means the model's actual discrimination breaks
down for that quarter pair -- no amount of rescaling fixes a ranking
failure.

Training-tracts loading: SimbaJSONDataset (imported directly, not
reimplemented) with validate=False loads EVERY tract for the eval
quarter unfiltered; this script then excludes whichever ones are listed
in the recipient model's own ckpt_dir/test_indices.txt (the same file
load_eval_cell's validate=True path uses to keep ONLY those) to get the
complementary training set. Deterministic eval transform (train=False,
augment=False) is used even for these "training" tracts -- they feed a
simple linear regression here, not backprop, so the random augmentation
pipeline would just add irrelevant noise to the recalibration fit.

Full matrix (diagonal included, not just off-diagonal) -- the diagonal
is each model on its own quarter, where recalibration should do little
(it's already well-calibrated there), a useful built-in sanity check.

A cell's recalibration is skipped (R^2 recalibrated left NaN, with a
printed note) if the eval quarter's training tracts have fewer than 5
usable tracts, or if the model's predictions on them have zero variance
(a degenerate linear fit) -- never silently reported as a real number.

Output, per state in --states:
    <out-dir>/<state>_<variable>_calibration_vs_ranking_long.csv   one
        row per cell: model/eval year+quarter, r2, pearson_r,
        r2_recalibrated, bias, recal_slope, recal_intercept, n_test,
        n_train
    <out-dir>/<state>_<variable>_r2_matrix.csv
    <out-dir>/<state>_<variable>_pearson_r_matrix.csv
    <out-dir>/<state>_<variable>_r2_recalibrated_matrix.csv
    <out-dir>/<state>_<variable>_bias_matrix.csv
    <fig-dir>/<state>_<variable>_r2_matrix.png
    <fig-dir>/<state>_<variable>_pearson_r_matrix.png
    <fig-dir>/<state>_<variable>_r2_recalibrated_matrix.png
    <fig-dir>/<state>_<variable>_bias_matrix.png

Usage:
    python 3_analysis/compute_calibration_vs_ranking_matrices.py --states oh pa az ga --variable wealth_index
"""

import argparse
import gc
import itertools
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_importance import (  # noqa: E402
    resolve_ckpt_and_prefix, resolve_data_root_fn, load_model_wrapper, load_eval_cell,
    compute_metrics, DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)

from sail.data.adapters import SimbaJSONDataset, resolve_json_paths  # noqa: E402
from sail.data.splitting import item_key  # noqa: E402
from sail.explain.band_importance import _load_full_dataset, _predict_all  # noqa: E402

MIN_TRAIN_TRACTS = 5


def load_train_cell(state, variable, model_ckpt_dir_path, eval_year, eval_quarter,
                     data_root_fn, prefix_fn, band_mean, band_std, img_size, batch_size):
    """Returns (images, coords, labels, names) for the eval quarter's
    TRAINING tracts -- the complement of load_eval_cell's held-out test
    set, read from the SAME model_ckpt_dir_path/test_indices.txt file
    load_eval_cell uses (every quarter shares the same stable/spatial
    split for a given state, so which model's ckpt_dir is passed doesn't
    change which GEOIDs are held out -- but this still reads it from the
    SAME ckpt_dir load_eval_cell was given for this cell, rather than
    assuming that equivalence, so nothing here depends on an unverified
    cross-model assumption)."""
    data_root = data_root_fn(state, eval_year, eval_quarter)
    prefix = prefix_fn(eval_year, eval_quarter)
    ys_path, coords_path, dup_path = resolve_json_paths(data_root=data_root, prefix=prefix, with_neighbors=False)

    # validate=False: loads every tract in this quarter unfiltered (no
    # test_indices.txt filtering at all), so this script can invert the
    # filter itself below rather than getting the held-out set directly.
    ds = SimbaJSONDataset(
        root_dir=data_root, ys_path=ys_path, coords_path=coords_path, dup_path=dup_path,
        ckpt_dir=model_ckpt_dir_path, validate=False, new=True,
        train=False, augment=False,
        img_size=tuple(img_size), band_mean=band_mean, band_std=band_std,
    )
    test_indices_path = os.path.join(model_ckpt_dir_path, "test_indices.txt")
    with open(test_indices_path) as f:
        test_names = f.read().splitlines()
    test_keys = {item_key(t) for t in test_names}
    ds.items = [it for it in ds.items if item_key(it) not in test_keys]
    if len(ds.items) == 0:
        raise RuntimeError(f"No training tracts left after excluding the held-out test set for "
                            f"{state} Q{eval_quarter} {eval_year} (test_indices.txt at {model_ckpt_dir_path})")

    return _load_full_dataset(ds, batch_size)


def fit_recalibration(preds_train: np.ndarray, labels_train: np.ndarray):
    """OLS label ~ a*pred + b on the training tracts. Returns (a, b, ok)
    -- ok=False (a=1.0, b=0.0, a no-op) if there aren't enough training
    tracts or the predictions have zero variance (a degenerate 1-D fit),
    so the caller can skip reporting a recalibrated R^2 for this cell
    rather than silently returning a meaningless one."""
    if len(preds_train) < MIN_TRAIN_TRACTS or np.ptp(preds_train) == 0:
        return 1.0, 0.0, False
    a, b = np.polyfit(preds_train, labels_train, 1)
    return float(a), float(b), True


def run_model(state, variable, model_year, model_quarter, years, quarters, args, ckpt_dir_fn,
              prefix_fn, data_root_fn, band_mean, band_std, in_channels, log):
    try:
        ckpt_dir = ckpt_dir_fn(model_year, model_quarter)
    except SystemExit as e:
        log(f"[{state} Q{model_quarter} {model_year}] no trained model found, skipping: {e}")
        return []
    try:
        model_wrapper, epoch, _ = load_model_wrapper(ckpt_dir, in_channels, args.device)
    except FileNotFoundError as e:
        log(f"[{state} Q{model_quarter} {model_year}] {e}, skipping")
        return []
    log(f"[{state} Q{model_quarter} {model_year}] loaded epoch {epoch}")

    rows = []
    for eval_year, eval_quarter in itertools.product(years, quarters):
        try:
            images_test, coords_test, labels_test, _ = load_eval_cell(
                state, variable, ckpt_dir, model_year, model_quarter, eval_year, eval_quarter,
                data_root_fn, prefix_fn, band_mean, band_std, args.img_size, args.batch_size,
            )
        except Exception as e:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] couldn't load test cell, skipping: {e}")
            continue
        n_test = images_test.shape[0]
        if n_test == 0:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] 0 overlapping test tracts, skipping")
            continue

        preds_test = _predict_all(model_wrapper, images_test, coords_test, args.device, args.batch_size).numpy()
        labels_test_np = labels_test.numpy()
        r2_raw, _, r = compute_metrics(preds_test, labels_test_np)
        bias = float(np.mean(preds_test - labels_test_np))

        r2_recal, recal_a, recal_b, n_train = np.nan, np.nan, np.nan, 0
        try:
            images_train, coords_train, labels_train, _ = load_train_cell(
                state, variable, ckpt_dir, eval_year, eval_quarter, data_root_fn, prefix_fn,
                band_mean, band_std, args.img_size, args.batch_size,
            )
            n_train = images_train.shape[0]
            preds_train = _predict_all(model_wrapper, images_train, coords_train, args.device,
                                        args.batch_size).numpy()
            recal_a, recal_b, ok = fit_recalibration(preds_train, labels_train.numpy())
            if ok:
                recal_preds_test = recal_a * preds_test + recal_b
                r2_recal = r2_score(labels_test_np, recal_preds_test)
            else:
                log(f"  [eval {state} Q{eval_quarter} {eval_year}] recalibration skipped "
                    f"(n_train={n_train} or zero-variance predictions)")
            del images_train, coords_train, labels_train
        except Exception as e:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] couldn't load training cell, "
                f"recalibration skipped: {e}")

        marker = " (diagonal)" if (model_year, model_quarter) == (eval_year, eval_quarter) else ""
        log(f"  [eval {state} Q{eval_quarter} {eval_year}]{marker} r2={r2_raw:.4f} r={r:.4f} "
            f"r2_recal={r2_recal:.4f} bias={bias:+.4f} (n_test={n_test}, n_train={n_train})")

        rows.append({
            "model_year": model_year, "model_quarter": model_quarter,
            "eval_year": eval_year, "eval_quarter": eval_quarter,
            "r2": r2_raw, "pearson_r": r, "r2_recalibrated": r2_recal, "bias": bias,
            "recal_slope": recal_a, "recal_intercept": recal_b,
            "n_test": n_test, "n_train": n_train,
        })

        del images_test, coords_test, labels_test
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def build_matrix(df: pd.DataFrame, value_col: str, quarters: list) -> pd.DataFrame:
    labels = [f"Q{q} {y}" for y, q in quarters]
    matrix = pd.DataFrame(np.nan, index=labels, columns=labels)
    for row in df.itertuples(index=False):
        i_label, j_label = f"Q{row.model_quarter} {row.model_year}", f"Q{row.eval_quarter} {row.eval_year}"
        if i_label in matrix.index and j_label in matrix.columns:
            matrix.loc[i_label, j_label] = getattr(row, value_col)
    return matrix


def plot_matrix(matrix: pd.DataFrame, title: str, out_path: Path, vrange: "tuple | None" = None):
    fig, ax = plt.subplots(figsize=(1.2 * len(matrix.columns) + 3, 1.2 * len(matrix.index) + 3))
    finite = matrix.values[np.isfinite(matrix.values)]
    if finite.size == 0:
        print(f"  no finite values to plot for {out_path.name}, skipping")
        plt.close(fig)
        return

    if vrange is not None:
        vmin, vmax = vrange
    else:
        vmin, vmax = float(finite.min()), float(finite.max())
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax) if vmin < 0 < vmax else None

    im = ax.imshow(matrix.values, cmap="RdBu", norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)
    ax.set_xlabel("Evaluated on imagery from")
    ax.set_ylabel("Model trained on")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_for_state(state, args, registry):
    def log(msg):
        print(msg, flush=True)

    st = resolve_state_settings(state, registry if registry is not None else load_registry())
    band_mean, band_std, in_channels = st["band_mean"], st["band_std"], st["in_channels"]
    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    all_rows = []
    for model_year, model_quarter in itertools.product(args.years, args.quarters):
        rows = run_model(state, args.variable, model_year, model_quarter, args.years, args.quarters,
                          args, ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels, log)
        all_rows.extend(rows)

    if not all_rows:
        log(f"[{state}] No cells produced -- nothing to write for this state.")
        return

    df = pd.DataFrame(all_rows)
    state_variable = f"{state}_{args.variable}"
    long_path = args.out_dir / f"{state_variable}_calibration_vs_ranking_long.csv"
    df.to_csv(long_path, index=False)
    print(f"\nWrote {long_path} ({len(df)} rows)")

    quarters_grid = list(itertools.product(args.years, args.quarters))
    for value_col, matrix_name, title, vrange in [
        ("r2", "r2", "R^2 (raw)", None),
        ("pearson_r", "pearson_r", "Pearson r", (-1.0, 1.0)),
        ("r2_recalibrated", "r2_recalibrated", "R^2 (recalibrated on target quarter's training tracts)", None),
        ("bias", "bias", "Mean signed prediction bias (pred - label, raw)", None),
    ]:
        matrix = build_matrix(df, value_col, quarters_grid)
        matrix_path = args.out_dir / f"{state_variable}_{matrix_name}_matrix.csv"
        matrix.to_csv(matrix_path)
        print(f"Wrote {matrix_path}")
        plot_matrix(matrix, f"{title} -- {state.upper()} {args.variable}",
                    args.fig_dir / f"{state_variable}_{matrix_name}_matrix.png", vrange)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019])
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--version", default=None, help="Trained version per model (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE)
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--out-dir", default="3_analysis/outputs/calibration_vs_ranking", type=Path)
    p.add_argument("--fig-dir", default="3_analysis/figures/calibration_vs_ranking", type=Path)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry() if args.use_state_registry else None

    for state in [s.lower() for s in args.states]:
        print(f"\n{'=' * 70}\n{state.upper()}\n{'=' * 70}")
        run_for_state(state, args, registry)


if __name__ == "__main__":
    main()
