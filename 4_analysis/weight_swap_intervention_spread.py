"""
Runs the layer-swap intervention across a SPREAD of (recipient, donor)
pairs, BOTH directions, with two causal controls added alongside the
original intervention from weight_swap_intervention.py:

  intervention              donor's --early-groups (default: patch_embed
                             stage1) swapped into the recipient -- the
                             original test.
  same_season_control        a THIRD model's --early-groups swapped in
                             instead -- a model sharing the RECIPIENT's
                             own quarter (same season) but a DIFFERENT
                             year, so it does NOT match the eval season
                             at all. If this does NOT recover
                             performance the way the real donor does,
                             that's evidence it's specifically EVAL-
                             SEASON-MATCHING early layers that help, not
                             "any early-layer transplant perturbs the
                             model into doing better."
  late_layers_complement     the SAME real donor's --late-groups
                             (default: stage3 stage4 final_norm) swapped
                             in INSTEAD of the early groups. If this does
                             NOT recover performance the way the early-
                             layer swap does, that reinforces "it's
                             specifically the early layers doing the
                             work," not late layers/the head.

Plus baseline (no swap) and the donor's own diagonal performance -- five
conditions total per directed pair. Baseline/intervention/same_season_
control/late_layers_complement are all evaluated on the SAME loaded eval
cell (one load, four forward passes, each on its OWN freshly-loaded
model copy so no condition's weight swap leaks into another's); only the
donor-context row needs the donor's own separately-loaded held-out
split.

Pairs: give --pairs explicitly as UNORDERED "yearA:quarterA:yearB:
quarterB" strings -- list each unordered pair ONCE, both directions are
run automatically. This is meant for a deliberately diverse, hand-picked
spread (different recipients, different quarter combinations), not an
auto-discovered "worst N" (that's weight_swap_intervention_batch.py's
job, and still available unchanged).

Same-season control selection: for a given recipient (ry, rq), tries
every OTHER year in --years sharing quarter rq, closest-year-first,
skipping any that doesn't actually have a trained checkpoint, until one
works. A directed pair's control is left out (not fatal) if no usable
other year exists.

Output:
    <out-dir>/<state>_<variable>_swap_spread_long.csv   every condition
        row for every directed pair
    <out-dir>/<state>_<variable>_swap_spread_summary.csv   one row per
        directed pair: baseline/intervention/same_season_control/
        late_layers_complement/donor r2, plus recovery_fraction
        (relative to the SAME donor_r2/baseline_r2 denominator) for the
        intervention and each control -- the real test of the
        hypothesis is whether intervention's recovery is clearly higher
        than either control's, not just whether it's positive

Usage:
    python 3_analysis/weight_swap_intervention_spread.py --state oh --variable wealth_index \
        --years 2016 2017 2018 2019 \
        --pairs 2018:1:2018:3 2017:2:2017:4 2016:4:2016:2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from band_importance import (  # noqa: E402
    resolve_ckpt_and_prefix, resolve_data_root_fn, load_model_wrapper, load_eval_cell,
    DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)
from weight_swap_intervention import load_donor_state_dict, swap_keys_for_groups, apply_swap, evaluate  # noqa: E402
from weight_swap_intervention_batch import recovery_fraction  # noqa: E402


def parse_unordered_pairs(pair_strs: list) -> list:
    pairs = []
    for s in pair_strs:
        parts = s.split(":")
        if len(parts) != 4:
            raise SystemExit(f"--pairs entry {s!r} must be 'yearA:quarterA:yearB:quarterB'")
        ya, qa, yb, qb = (int(p) for p in parts)
        pairs.append((ya, qa, yb, qb))
    return pairs


def same_season_control_candidates(recipient_year, recipient_quarter, donor_year, donor_quarter, years):
    """Every OTHER year sharing the recipient's own quarter, closest-
    year-first -- the caller tries each in turn until one has an actual
    trained checkpoint."""
    return [y for y in sorted(set(years), key=lambda y: abs(y - recipient_year))
            if y != recipient_year and (y, recipient_quarter) != (donor_year, donor_quarter)]


def load_same_season_control(recipient_year, recipient_quarter, donor_year, donor_quarter, years, ckpt_dir_fn):
    for candidate_year in same_season_control_candidates(recipient_year, recipient_quarter, donor_year,
                                                          donor_quarter, years):
        try:
            candidate_ckpt_dir = ckpt_dir_fn(candidate_year, recipient_quarter)
            candidate_state, _, _ = load_donor_state_dict(candidate_ckpt_dir)
        except (SystemExit, FileNotFoundError, RuntimeError) as e:
            print(f"  same_season_control candidate {candidate_year}Q{recipient_quarter} unavailable: {e}")
            continue
        return candidate_state, candidate_year
    return None, None


def run_direction(state, variable, recipient_year, recipient_quarter, donor_year, donor_quarter, years,
                   early_groups, late_groups, ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std,
                   in_channels, device, batch_size, img_size):
    eval_year, eval_quarter = donor_year, donor_quarter
    recipient_ckpt_dir = ckpt_dir_fn(recipient_year, recipient_quarter)
    donor_ckpt_dir = ckpt_dir_fn(donor_year, donor_quarter)
    print(f"\nRecipient {recipient_year}Q{recipient_quarter} <- donor {donor_year}Q{donor_quarter} "
          f"(eval {eval_year}Q{eval_quarter})")

    images, coords, labels, _ = load_eval_cell(
        state, variable, recipient_ckpt_dir, recipient_year, recipient_quarter, eval_year, eval_quarter,
        data_root_fn, prefix_fn, band_mean, band_std, img_size, batch_size,
    )
    n_test = images.shape[0]
    if n_test == 0:
        print("  0 overlapping test tracts -- skipping this direction")
        return None

    rows = []

    baseline_wrapper, _, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, device)
    r2, mae, r, _ = evaluate(baseline_wrapper, images, coords, labels, device, batch_size)
    rows.append({"condition": "baseline", "r2": r2, "mae": mae, "r": r, "n_test": n_test})
    print(f"  baseline: R^2={r2:.4f}")

    donor_state, donor_epoch, donor_path = load_donor_state_dict(donor_ckpt_dir)

    early_keys = swap_keys_for_groups(donor_state, early_groups)
    int_wrapper, _, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, device)
    int_wrapper.net = apply_swap(int_wrapper.net, donor_state, early_keys, device)
    r2, mae, r, _ = evaluate(int_wrapper, images, coords, labels, device, batch_size)
    rows.append({"condition": "intervention", "r2": r2, "mae": mae, "r": r, "n_test": n_test})
    print(f"  intervention ({'+'.join(early_groups)} from donor): R^2={r2:.4f}")

    control_state, control_year = load_same_season_control(recipient_year, recipient_quarter, donor_year,
                                                             donor_quarter, years, ckpt_dir_fn)
    if control_state is not None:
        control_keys = swap_keys_for_groups(control_state, early_groups)
        ctrl_wrapper, _, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, device)
        ctrl_wrapper.net = apply_swap(ctrl_wrapper.net, control_state, control_keys, device)
        r2, mae, r, _ = evaluate(ctrl_wrapper, images, coords, labels, device, batch_size)
        rows.append({"condition": "same_season_control", "r2": r2, "mae": mae, "r": r, "n_test": n_test,
                     "control_year": control_year, "control_quarter": recipient_quarter})
        print(f"  same_season_control ({control_year}Q{recipient_quarter}'s {'+'.join(early_groups)}): "
              f"R^2={r2:.4f}")
    else:
        print(f"  same_season_control: no usable other year at Q{recipient_quarter} in --years, skipped")

    late_keys = swap_keys_for_groups(donor_state, late_groups)
    late_wrapper, _, _ = load_model_wrapper(recipient_ckpt_dir, in_channels, device)
    late_wrapper.net = apply_swap(late_wrapper.net, donor_state, late_keys, device)
    r2, mae, r, _ = evaluate(late_wrapper, images, coords, labels, device, batch_size)
    rows.append({"condition": "late_layers_complement", "r2": r2, "mae": mae, "r": r, "n_test": n_test})
    print(f"  late_layers_complement ({'+'.join(late_groups)} from donor): R^2={r2:.4f}")

    donor_images, donor_coords, donor_labels, _ = load_eval_cell(
        state, variable, donor_ckpt_dir, donor_year, donor_quarter, donor_year, donor_quarter,
        data_root_fn, prefix_fn, band_mean, band_std, img_size, batch_size,
    )
    donor_wrapper, _, _ = load_model_wrapper(donor_ckpt_dir, in_channels, device)
    r2, mae, r, _ = evaluate(donor_wrapper, donor_images, donor_coords, donor_labels, device, batch_size)
    rows.append({"condition": "donor_own_diagonal_context", "r2": r2, "mae": mae, "r": r,
                 "n_test": donor_images.shape[0]})
    print(f"  donor_own_diagonal_context: R^2={r2:.4f}")

    for row in rows:
        row.update({
            "state": state, "variable": variable,
            "recipient_year": recipient_year, "recipient_quarter": recipient_quarter,
            "donor_year": donor_year, "donor_quarter": donor_quarter,
            "eval_year": eval_year, "eval_quarter": eval_quarter,
            "early_groups": "+".join(early_groups), "late_groups": "+".join(late_groups),
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--years", nargs="+", type=int, required=True,
                    help="Every year that could serve as a same-season control candidate -- include "
                         "every year you have a trained model for, not just the years in --pairs")
    p.add_argument("--pairs", nargs="+", required=True,
                    help="One or more UNORDERED 'yearA:quarterA:yearB:quarterB' pairs -- both "
                         "directions run automatically for each one given")
    p.add_argument("--early-groups", nargs="+", default=["patch_embed", "stage1"])
    p.add_argument("--late-groups", nargs="+", default=["stage3", "stage4", "final_norm"])
    p.add_argument("--version", default=None)
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE)
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--out-dir", default="3_analysis/outputs/weight_drift", type=Path)
    args = p.parse_args()

    state = args.state.lower()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    unordered_pairs = parse_unordered_pairs(args.pairs)
    directed_pairs = []
    for ya, qa, yb, qb in unordered_pairs:
        directed_pairs.append((ya, qa, yb, qb))
        directed_pairs.append((yb, qb, ya, qa))

    registry = load_registry() if args.use_state_registry else None
    st = resolve_state_settings(state, registry if registry is not None else load_registry())
    band_mean, band_std, in_channels = st["band_mean"], st["band_std"], st["in_channels"]
    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    all_rows, summary_rows = [], []
    for i, (ry, rq, dy, dq) in enumerate(directed_pairs):
        print(f"\n{'=' * 70}\nDirected pair {i + 1}/{len(directed_pairs)}: "
              f"recipient {ry}Q{rq} <- donor {dy}Q{dq}\n{'=' * 70}")
        rows = run_direction(state, args.variable, ry, rq, dy, dq, args.years, args.early_groups,
                              args.late_groups, ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std,
                              in_channels, args.device, args.batch_size, args.img_size)
        if rows is None:
            continue
        all_rows.extend(rows)

        by_cond = {r["condition"]: r for r in rows}
        baseline_r2 = by_cond["baseline"]["r2"]
        donor_r2 = by_cond["donor_own_diagonal_context"]["r2"]
        summary_row = {
            "recipient_year": ry, "recipient_quarter": rq, "donor_year": dy, "donor_quarter": dq,
            "baseline_r2": baseline_r2, "donor_r2": donor_r2,
            "intervention_r2": by_cond["intervention"]["r2"],
            "intervention_recovery": recovery_fraction(baseline_r2, by_cond["intervention"]["r2"], donor_r2),
            "late_layers_r2": by_cond["late_layers_complement"]["r2"],
            "late_layers_recovery": recovery_fraction(baseline_r2, by_cond["late_layers_complement"]["r2"],
                                                       donor_r2),
        }
        if "same_season_control" in by_cond:
            summary_row["same_season_control_r2"] = by_cond["same_season_control"]["r2"]
            summary_row["same_season_control_recovery"] = recovery_fraction(
                baseline_r2, by_cond["same_season_control"]["r2"], donor_r2)
            summary_row["control_year"] = by_cond["same_season_control"]["control_year"]
        else:
            summary_row["same_season_control_r2"] = np.nan
            summary_row["same_season_control_recovery"] = np.nan
            summary_row["control_year"] = None
        summary_rows.append(summary_row)

    if not summary_rows:
        raise SystemExit("Every directed pair was skipped -- nothing to summarize.")

    state_variable = f"{state}_{args.variable}"
    long_path = args.out_dir / f"{state_variable}_swap_spread_long.csv"
    pd.DataFrame(all_rows).to_csv(long_path, index=False)
    print(f"\nWrote {long_path} ({len(all_rows)} rows)")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / f"{state_variable}_swap_spread_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path} ({len(summary_df)} directed pairs)")

    print("\nMean recovery fraction across directed pairs (excluding NaN denominators) -- the "
          "hypothesis predicts intervention_recovery should be clearly higher than both controls':")
    for col in ["intervention_recovery", "same_season_control_recovery", "late_layers_recovery"]:
        valid = summary_df[col].dropna()
        if len(valid):
            print(f"  {col}: mean={valid.mean():.1%} median={valid.median():.1%} (n={len(valid)})")
        else:
            print(f"  {col}: no usable pairs")


if __name__ == "__main__":
    main()
