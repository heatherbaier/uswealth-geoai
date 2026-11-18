"""
The causal follow-up to analyze_weight_drift_by_layer.py's correlational
finding: if patch_embed + stage1 carry the strongest season-clustering
signal, does TRANSPLANTING those layers from a model that natively
handles a given quarter's imagery actually recover performance on a
model that otherwise fails on it -- without any retraining?

Procedure (recipient = the model being modified and evaluated; donor =
the model whose layers get transplanted in; both share the same
pretrained initialization, which is what makes a direct weight swap
between two independently fine-tuned checkpoints even well-defined --
every layer still means the same thing architecturally in both):
  1. Load the recipient's checkpoint twice: once left untouched
     (baseline), once to be modified (intervention).
  2. Load the donor's checkpoint (raw state_dict only -- no need to
     build its model) and pull out the tensors belonging to
     --swap-groups (default: patch_embed stage1 -- see analyze_weight_
     drift_by_layer.py's ANATOMICAL_ORDER/discover_block_to_group,
     imported directly rather than reimplemented).
  3. Overwrite ONLY those tensors on the intervention copy
     (net.load_state_dict(..., strict=False) with a dict containing only
     the swapped keys -- everything else on the intervention copy stays
     exactly as the recipient trained it).
  4. Evaluate baseline and intervention on the SAME eval cell: the
     recipient's own held-out test tracts (same leak-free mechanism
     band_importance.py/amplitude_predictions.py use), as they appear in
     --eval-year/--eval-quarter's imagery (default: the donor's own
     quarter -- the natural "does borrowing the donor's early layers
     recover its own-quarter performance" test).
  5. For context (NOT an apples-to-apples number, since it uses the
     DONOR's own held-out split rather than the recipient's -- read it
     as a rough ceiling, not a fourth directly-comparable row): also
     reports the donor's own diagonal performance on its own quarter.

Output:
    <out-dir>/<state>_<variable>_swap_<ry>q<rq>_from_<dy>q<dq>_<groups>.csv
    columns: condition (baseline/intervention/donor_own_diagonal_context),
    r2, mae, r, n_test, plus the run's full identity (state, variable,
    recipient/donor/eval year+quarter, swap_groups)

Usage (the motivating case: OH's worst cell, Q1 2018 model on Q3 2018
imagery, patched with Q3 2018's own patch_embed + stage1):
    python 3_analysis/weight_swap_intervention.py --state oh --variable wealth_index \
        --recipient-year 2018 --recipient-quarter 1 \
        --donor-year 2018 --donor-quarter 3

    # Try a different/larger set of transplanted layers:
    python 3_analysis/weight_swap_intervention.py --state oh --variable wealth_index \
        --recipient-year 2018 --recipient-quarter 1 --donor-year 2018 --donor-quarter 3 \
        --swap-groups patch_embed stage1 stage2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_importance import (  # noqa: E402
    resolve_ckpt_and_prefix, resolve_data_root_fn, load_model_wrapper, load_eval_cell,
    compute_metrics, DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)
from analyze_weight_drift_by_layer import discover_block_to_group, assign_group  # noqa: E402

from sail.core.builders import highest_epoch  # noqa: E402
from sail.explain.band_importance import _predict_all  # noqa: E402


def load_donor_state_dict(donor_ckpt_dir: str):
    epoch, path = highest_epoch(donor_ckpt_dir)
    if path is None:
        raise SystemExit(f"No model_epoch*.torch checkpoint under {donor_ckpt_dir} (donor)")
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    return state_dict, epoch, path


def swap_keys_for_groups(donor_state: dict, swap_groups: list) -> list:
    block_to_group = discover_block_to_group(donor_state.keys())
    keys = sorted(k for k in donor_state if assign_group(k, block_to_group) in swap_groups)
    if not keys:
        raise SystemExit(f"No donor key(s) matched --swap-groups {swap_groups} -- check the "
                          f"group name(s) against analyze_weight_drift_by_layer.py's "
                          f"ANATOMICAL_ORDER (patch_embed, stage1..stage4, final_norm, head).")
    return keys


def apply_swap(net: torch.nn.Module, donor_state: dict, keys: list, device: str):
    recipient_state = net.state_dict()
    missing = [k for k in keys if k not in recipient_state]
    if missing:
        raise SystemExit(f"Donor has key(s) the recipient doesn't: {missing[:5]} -- different "
                          f"architecture/in_channels between donor and recipient?")
    shape_mismatch = [k for k in keys if recipient_state[k].shape != donor_state[k].shape]
    if shape_mismatch:
        raise SystemExit(f"Shape mismatch on swapped key(s): "
                          f"{[(k, tuple(recipient_state[k].shape), tuple(donor_state[k].shape)) for k in shape_mismatch[:5]]}")
    swap_dict = {k: donor_state[k] for k in keys}
    net.load_state_dict(swap_dict, strict=False)
    return net.to(device).eval()


def evaluate(model_wrapper, images, coords, labels, device, batch_size):
    preds = _predict_all(model_wrapper, images, coords, device, batch_size).numpy()
    r2, mae, r = compute_metrics(preds, labels.numpy())
    return r2, mae, r, preds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--recipient-year", required=True, type=int)
    p.add_argument("--recipient-quarter", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--donor-year", required=True, type=int)
    p.add_argument("--donor-quarter", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--eval-year", type=int, default=None, help="Default: donor's own year")
    p.add_argument("--eval-quarter", type=int, default=None, choices=[1, 2, 3, 4],
                    help="Default: donor's own quarter")
    p.add_argument("--swap-groups", nargs="+", default=["patch_embed", "stage1"],
                    help="Layer group(s) (analyze_weight_drift_by_layer.py's ANATOMICAL_ORDER) to "
                         "transplant from the donor into the recipient. Default: patch_embed stage1.")
    p.add_argument("--version", default=None, help="Trained version per checkpoint (default: latest existing)")
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE)
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--out-dir", default="3_analysis/outputs/weight_drift", type=Path)
    args = p.parse_args()

    if args.eval_year is None:
        args.eval_year = args.donor_year
    if args.eval_quarter is None:
        args.eval_quarter = args.donor_quarter

    state = args.state.lower()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    registry = load_registry() if args.use_state_registry else None
    st = resolve_state_settings(state, registry if registry is not None else load_registry())
    band_mean, band_std, in_channels = st["band_mean"], st["band_std"], st["in_channels"]

    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    recipient_ckpt_dir = ckpt_dir_fn(args.recipient_year, args.recipient_quarter)
    donor_ckpt_dir = ckpt_dir_fn(args.donor_year, args.donor_quarter)
    print(f"Recipient: {state} Q{args.recipient_quarter} {args.recipient_year} -> {recipient_ckpt_dir}")
    print(f"Donor:     {state} Q{args.donor_quarter} {args.donor_year} -> {donor_ckpt_dir}")
    print(f"Eval imagery: {state} Q{args.eval_quarter} {args.eval_year}")
    print(f"Swapping group(s) from donor into recipient: {args.swap_groups}")

    print("\nLoading recipient's eval cell (recipient's own held-out test tracts, eval imagery)...")
    images, coords, labels, _ = load_eval_cell(
        state, args.variable, recipient_ckpt_dir, args.recipient_year, args.recipient_quarter,
        args.eval_year, args.eval_quarter, data_root_fn, prefix_fn, band_mean, band_std,
        args.img_size, args.batch_size,
    )
    n_test = images.shape[0]
    if n_test == 0:
        raise SystemExit("0 overlapping test tracts for this (recipient, eval) cell -- nothing to evaluate.")
    print(f"n_test={n_test}")

    print("\n[1/2] BASELINE -- recipient model, unmodified:")
    baseline_wrapper, baseline_epoch, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, args.device)
    baseline_r2, baseline_mae, baseline_r, _ = evaluate(baseline_wrapper, images, coords, labels,
                                                         args.device, args.batch_size)
    print(f"  epoch {baseline_epoch}: R^2={baseline_r2:.4f} MAE={baseline_mae:.4f} r={baseline_r:.4f}")

    print(f"\n[2/2] INTERVENTION -- recipient with {'+'.join(args.swap_groups)} from donor:")
    intervention_wrapper, intervention_epoch, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, args.device)
    donor_state, donor_epoch, donor_path = load_donor_state_dict(donor_ckpt_dir)
    swap_keys = swap_keys_for_groups(donor_state, args.swap_groups)
    n_swapped_params = sum(donor_state[k].numel() for k in swap_keys)
    print(f"  donor epoch {donor_epoch} ({donor_path}): transplanting {len(swap_keys)} tensor(s), "
          f"{n_swapped_params:,} params")
    intervention_wrapper.net = apply_swap(intervention_wrapper.net, donor_state, swap_keys, args.device)
    intervention_r2, intervention_mae, intervention_r, _ = evaluate(intervention_wrapper, images, coords,
                                                                     labels, args.device, args.batch_size)
    print(f"  epoch {intervention_epoch} (patched): R^2={intervention_r2:.4f} MAE={intervention_mae:.4f} "
          f"r={intervention_r:.4f}")
    print(f"\nDelta R^2 (intervention - baseline): {intervention_r2 - baseline_r2:+.4f}")

    rows = [
        {"condition": "baseline", "r2": baseline_r2, "mae": baseline_mae, "r": baseline_r, "n_test": n_test},
        {"condition": "intervention", "r2": intervention_r2, "mae": intervention_mae, "r": intervention_r,
         "n_test": n_test},
    ]

    print("\nFor context (NOT apples-to-apples -- uses the DONOR's own held-out split): "
          "donor model's own performance on its own quarter's imagery...")
    donor_images, donor_coords, donor_labels, _ = load_eval_cell(
        state, args.variable, donor_ckpt_dir, args.donor_year, args.donor_quarter,
        args.donor_year, args.donor_quarter, data_root_fn, prefix_fn, band_mean, band_std,
        args.img_size, args.batch_size,
    )
    donor_wrapper, donor_model_epoch, _ = load_model_wrapper(donor_ckpt_dir, in_channels, args.device)
    donor_r2, donor_mae, donor_r, _ = evaluate(donor_wrapper, donor_images, donor_coords, donor_labels,
                                                args.device, args.batch_size)
    print(f"  epoch {donor_model_epoch}: R^2={donor_r2:.4f} MAE={donor_mae:.4f} r={donor_r:.4f} "
          f"(n_test={donor_images.shape[0]}, donor's own held-out split)")
    rows.append({"condition": "donor_own_diagonal_context", "r2": donor_r2, "mae": donor_mae, "r": donor_r,
                 "n_test": donor_images.shape[0]})

    df = pd.DataFrame(rows)
    df["state"], df["variable"] = state, args.variable
    df["recipient_year"], df["recipient_quarter"] = args.recipient_year, args.recipient_quarter
    df["donor_year"], df["donor_quarter"] = args.donor_year, args.donor_quarter
    df["eval_year"], df["eval_quarter"] = args.eval_year, args.eval_quarter
    df["swap_groups"] = "+".join(args.swap_groups)

    swap_label = "+".join(args.swap_groups)
    out_path = args.out_dir / (f"{state}_{args.variable}_swap_{args.recipient_year}q{args.recipient_quarter}"
                                f"_from_{args.donor_year}q{args.donor_quarter}_{swap_label}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
