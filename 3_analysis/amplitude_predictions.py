"""
Step 1 of the seasonal-amplitude pipeline: for every trained model in
--years/--quarters, get its predictions on ITS OWN held-out test tracts,
as they appear in EVERY OTHER quarter's imagery in that same grid (not
just its own quarter) -- the full (model, eval) grid, diagonal and
off-diagonal both. This is a thin, GEOID-capturing specialization of
band_importance.py's model/dataset-loading path (load_model_wrapper,
load_eval_cell, same leak-free "restrict to the model's own held-out
GEOIDs" mechanism, same reasoning for why that's valid cross-quarter --
see that script's docstring) -- no permutation/ablation here, just the
baseline prediction, with GEOID kept this time instead of discarded.

Why every model x every quarter, not just the diagonal: amplitude_
covariates.py (step 2) needs, for each tract, every (model_quarter,
imagery_quarter) combination to compute seasonal amplitude (same model,
imagery varies by quarter), model disagreement (same-year models applied
each to its OWN quarter), and the noise floor (same model+quarter,
varies by year) -- see that script's docstring for the exact definitions.
All of those are slices of this one (model, eval) grid, so it's collected
once here rather than three separate times.

Output: one long CSV per state --
    <out-dir>/<state>_<variable>_amplitude_predictions.csv
columns: state, GEOID, model_year, model_quarter, eval_year, eval_quarter,
pred, label.

Resume: a (model_year, model_quarter, eval_year, eval_quarter) cell
already present in the output CSV is skipped unless --overwrite -- same
per-cell granularity as band_importance.py, same reasoning (a cell's
rows are always written as a complete unit, so "present" means "done").
A missing checkpoint or missing imagery for any cell is skipped with a
message, not a crash.

Incremental saving: the output CSV is rewritten after EVERY trained
model finishes its full eval sweep (i.e. after each (model_year,
model_quarter) in the outer loop, not just once at the end) -- so you
can watch 3_analysis/outputs/amplitude/<state>_<variable>_amplitude_
predictions.csv grow while this is still running, and a killed/crashed
run only loses the one model it was partway through, not everything.

Usage:
    python 3_analysis/amplitude_predictions.py --state oh --variable wealth_index \
        --years 2017 2018 2019 --use-state-registry
"""

import argparse
import gc
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_importance import (  # noqa: E402
    resolve_ckpt_and_prefix, resolve_data_root_fn, load_model_wrapper, load_eval_cell,
    DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)
from sail.explain.band_importance import _predict_all  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from update_ys_labels import normalize_geoid  # noqa: E402


def run_model(state, variable, model_year, model_quarter, args, ckpt_dir_fn, prefix_fn,
              data_root_fn, band_mean, band_std, in_channels, existing_cells, log):
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

    rows = []
    for eval_year, eval_quarter in itertools.product(args.years, args.quarters):
        cell_key = (model_year, model_quarter, eval_year, eval_quarter)
        if cell_key in existing_cells:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] already in output CSV, skipping")
            continue

        try:
            images, coords, labels, names = load_eval_cell(
                state, variable, ckpt_dir, model_year, model_quarter, eval_year, eval_quarter,
                data_root_fn, prefix_fn, band_mean, band_std, args.img_size, args.batch_size,
            )
        except Exception as e:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] couldn't load this cell, skipping: {e}")
            continue

        n_test = images.shape[0]
        if n_test == 0:
            log(f"  [eval {state} Q{eval_quarter} {eval_year}] 0 overlapping test tracts, skipping")
            continue

        preds = _predict_all(model_wrapper, images, coords, args.device, args.batch_size).numpy()
        labels_np = labels.numpy()
        geoids = [normalize_geoid(Path(n).stem) for n in names]

        for geoid, pred, label in zip(geoids, preds, labels_np):
            rows.append({"state": state, "GEOID": geoid, "model_year": model_year,
                         "model_quarter": model_quarter, "eval_year": eval_year,
                         "eval_quarter": eval_quarter, "pred": float(pred), "label": float(label)})
        log(f"  [eval {state} Q{eval_quarter} {eval_year}] n_test={n_test} rows added")

        del images, coords, labels
        gc.collect()
        if args.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", nargs="+", type=int, default=[2017, 2018, 2019])
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--version", default=None, help="Trained version per model (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE,
                    help=f"Default: {DEFAULT_DATA_ROOT_TEMPLATE!r}")
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE,
                    help=f"Default: {DEFAULT_BASE_PREFIX_TEMPLATE!r}")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--out-dir", default="3_analysis/outputs/amplitude")
    p.add_argument("--overwrite", action="store_true",
                    help="Recompute every cell instead of skipping ones already in the output CSV")
    args = p.parse_args()

    state = args.state.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(msg, flush=True)

    registry = load_registry()
    st = resolve_state_settings(state, registry)
    band_mean, band_std, in_channels = st["band_mean"], st["band_std"], st["in_channels"]

    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    csv_path = out_dir / f"{state}_{args.variable}_amplitude_predictions.csv"
    existing_df = pd.read_csv(csv_path) if (csv_path.exists() and not args.overwrite) else None
    existing_cells = set()
    if existing_df is not None:
        existing_cells = set(existing_df[["model_year", "model_quarter", "eval_year", "eval_quarter"]]
                              .drop_duplicates().itertuples(index=False, name=None))

    combined = existing_df if existing_df is not None else pd.DataFrame()
    for model_year, model_quarter in itertools.product(args.years, args.quarters):
        rows = run_model(state, args.variable, model_year, model_quarter, args,
                          ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
                          existing_cells, log)
        if not rows:
            continue
        new_df = pd.DataFrame(rows)
        combined = pd.concat([combined, new_df], ignore_index=True) if not combined.empty else new_df
        combined.to_csv(csv_path, index=False)
        log(f"  [{state} Q{model_quarter} {model_year}] saved progress: {len(combined):,} rows so far "
            f"in {csv_path}")

    if combined.empty:
        raise SystemExit("No predictions produced -- nothing to write.")
    log(f"\nWrote {csv_path} ({len(combined):,} rows, "
        f"{combined[['model_year', 'model_quarter', 'eval_year', 'eval_quarter']].drop_duplicates().shape[0]} cells, "
        f"{combined['GEOID'].nunique():,} unique GEOIDs)")


if __name__ == "__main__":
    main()
