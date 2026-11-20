"""
Runs weight_swap_intervention.py's baseline/intervention/donor-context
comparison across SEVERAL (recipient, donor) pairs instead of just one,
and summarizes how much of each pair's transfer gap the swap recovers
-- the natural follow-up once a single pair (e.g. OH's worst cell, Q1
2018 model on Q3 2018 imagery) showed a large recovery: does that
generalize, or was it a feature of that one pair?

Pairs come from one of three sources:
  --top-n-worst (default)   auto-discovers the N worst off-diagonal
      cells from --band-importance-csv/--cross-quarter-r2-csv (same
      loader analyze_cross_quarter_rs_gap.py uses), sorted by R^2
      ascending. For each worst cell (model_year, model_quarter,
      eval_year, eval_quarter, r2): recipient = (model_year,
      model_quarter) (the model that failed), donor = (eval_year,
      eval_quarter) (the model that NATIVELY handles that eval quarter
      -- the same natural pairing the original single-pair request
      used, since donor_year/donor_quarter default to eval_year/
      eval_quarter in weight_swap_intervention.py too).
  --all-cells                every off-diagonal cell in the source CSV,
      not just the worst N -- the same (recipient, donor) pairing as
      above, just unfiltered. Expensive: a full 16x16 grid has 240
      off-diagonal cells, each needing 2+ model loads and forward
      passes (baseline, intervention, donor context), on top of
      whatever --swap-groups requires.
  --pairs                   explicit override: one or more
      "recipient_year:recipient_quarter:donor_year:donor_quarter"
      strings, for full manual control instead of auto-discovery.

Incremental saving + resume (specifically for --all-cells, where a run
can be long): both output files are rewritten after EVERY pair
finishes, and a pair already present in an existing summary CSV is
skipped on a rerun unless --overwrite is given.

Recovery fraction per pair: (intervention_r2 - baseline_r2) /
(donor_r2 - baseline_r2) -- 0 means the swap did nothing, 1 means it
fully closed the gap to the donor's own native ceiling, >1 means it
overshot that ceiling. Left as NaN (not clipped or hidden) when the
denominator is <= 0 (donor's own R^2 isn't actually better than the
recipient's baseline on this pair -- a sign the pair itself may not be
a meaningful "worse than its donor" comparison, worth a second look
rather than silently averaging in a nonsensical ratio).

Reuses weight_swap_intervention.py's run_pair() directly (one
implementation, not two copies) -- this script only adds the pair
selection and the across-pairs summary on top.

Output:
    <out-dir>/<state>_<variable>_swap_batch_long.csv   every row from
        every pair's run_pair() call, concatenated (same columns as a
        single-pair run's output)
    <out-dir>/<state>_<variable>_swap_batch_summary.csv   one row per
        pair: recipient/donor/eval identity, baseline/intervention/
        donor r2, delta_r2, recovery_fraction
    printed: mean/median recovery_fraction across pairs (excluding any
        NaN pairs, with a count of how many were excluded)

Usage:
    python 4_analysis/weight_swap_intervention_batch.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv ./out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --top-n-worst 5

    # Explicit pairs instead of auto-discovery:
    python 4_analysis/weight_swap_intervention_batch.py --state oh --variable wealth_index \
        --pairs 2018:1:2018:3 2017:2:2017:4 2016:4:2016:2

    # Every off-diagonal cell (expensive -- see above):
    python 4_analysis/weight_swap_intervention_batch.py --state oh --variable wealth_index \
        --cross-quarter-r2-csv ./out_cross_quarter/oh_wealth_index/r2_matrix.csv \
        --all-cells
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
    resolve_ckpt_and_prefix, resolve_data_root_fn, DEFAULT_DATA_ROOT_TEMPLATE, DEFAULT_BASE_PREFIX_TEMPLATE,
)
from weight_swap_intervention import run_pair  # noqa: E402
from analyze_cross_quarter_rs_gap import load_offdiag_cells  # noqa: E402


def parse_pairs(pair_strs: list) -> list:
    pairs = []
    for s in pair_strs:
        parts = s.split(":")
        if len(parts) != 4:
            raise SystemExit(f"--pairs entry {s!r} must be 'recipient_year:recipient_quarter:"
                              f"donor_year:donor_quarter'")
        ry, rq, dy, dq = (int(p) for p in parts)
        pairs.append((ry, rq, dy, dq))
    return pairs


def discover_worst_pairs(band_importance_csv, cross_quarter_r2_csv, top_n: int) -> list:
    cells = load_offdiag_cells(band_importance_csv, cross_quarter_r2_csv)
    worst = cells.sort_values("r2").head(top_n)
    print(f"Worst {len(worst)} off-diagonal cell(s) by R^2:")
    pairs = []
    for row in worst.itertuples(index=False):
        print(f"  model {row.model_year}Q{row.model_quarter} -> eval {row.eval_year}Q{row.eval_quarter}: "
              f"R^2={row.r2:.4f}")
        pairs.append((row.model_year, row.model_quarter, row.eval_year, row.eval_quarter))
    return pairs


def discover_all_pairs(band_importance_csv, cross_quarter_r2_csv) -> list:
    cells = load_offdiag_cells(band_importance_csv, cross_quarter_r2_csv)
    print(f"All {len(cells)} off-diagonal cell(s) (--all-cells, unsorted/unfiltered):")
    pairs = [(row.model_year, row.model_quarter, row.eval_year, row.eval_quarter)
              for row in cells.itertuples(index=False)]
    return pairs


def recovery_fraction(baseline_r2, intervention_r2, donor_r2):
    denom = donor_r2 - baseline_r2
    if denom <= 0:
        return np.nan
    return (intervention_r2 - baseline_r2) / denom


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--variable", default="wealth_index")
    p.add_argument("--pairs", nargs="+", default=None,
                    help="Explicit 'ry:rq:dy:dq' pair(s), overriding auto-discovery")
    p.add_argument("--band-importance-csv", type=Path, default=None,
                    help="For auto-discovering the worst cells (ignored if --pairs is given)")
    p.add_argument("--cross-quarter-r2-csv", type=Path, default=None,
                    help="Cheaper alternative to --band-importance-csv for the same purpose")
    p.add_argument("--top-n-worst", type=int, default=3)
    p.add_argument("--all-cells", action="store_true",
                    help="Run EVERY off-diagonal cell instead of just --top-n-worst -- expensive, "
                         "see module docstring. Ignores --top-n-worst.")
    p.add_argument("--swap-groups", nargs="+", default=["patch_embed", "stage1"])
    p.add_argument("--version", default=None)
    p.add_argument("--use-state-registry", action="store_true")
    p.add_argument("--data-root-template", default=DEFAULT_DATA_ROOT_TEMPLATE)
    p.add_argument("--base-prefix-template", default=DEFAULT_BASE_PREFIX_TEMPLATE)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, nargs=2, default=[256, 256])
    p.add_argument("--out-dir", default="4_analysis/outputs/weight_drift", type=Path)
    p.add_argument("--overwrite", action="store_true",
                    help="Recompute every pair instead of resuming from an existing summary CSV")
    args = p.parse_args()

    state = args.state.lower()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.pairs is not None:
        pairs = parse_pairs(args.pairs)
    elif args.all_cells:
        if args.band_importance_csv is None and args.cross_quarter_r2_csv is None:
            raise SystemExit("--all-cells needs --band-importance-csv/--cross-quarter-r2-csv")
        pairs = discover_all_pairs(args.band_importance_csv, args.cross_quarter_r2_csv)
    else:
        if args.band_importance_csv is None and args.cross_quarter_r2_csv is None:
            raise SystemExit("Pass --pairs explicitly, or --band-importance-csv/--cross-quarter-r2-csv "
                              "to auto-discover the worst cells")
        pairs = discover_worst_pairs(args.band_importance_csv, args.cross_quarter_r2_csv, args.top_n_worst)
    if not pairs:
        raise SystemExit("No pairs to run.")

    registry = load_registry() if args.use_state_registry else None
    st = resolve_state_settings(state, registry if registry is not None else load_registry())
    band_mean, band_std, in_channels = st["band_mean"], st["band_std"], st["in_channels"]
    ckpt_dir_fn, prefix_fn = resolve_ckpt_and_prefix(state, args.variable, args, registry)
    data_root_fn = resolve_data_root_fn(args, registry)

    state_variable = f"{state}_{args.variable}"
    long_path = args.out_dir / f"{state_variable}_swap_batch_long.csv"
    summary_path = args.out_dir / f"{state_variable}_swap_batch_summary.csv"

    existing_summary = pd.read_csv(summary_path) if (summary_path.exists() and not args.overwrite) else None
    existing_long = pd.read_csv(long_path) if (long_path.exists() and not args.overwrite) else None
    done_pairs = set()
    if existing_summary is not None and not existing_summary.empty:
        done_pairs = set(existing_summary[["recipient_year", "recipient_quarter", "donor_year", "donor_quarter"]]
                          .itertuples(index=False, name=None))
        print(f"Resuming from {summary_path} ({len(done_pairs)} pair(s) already done)")

    all_rows = list(existing_long.to_dict("records")) if existing_long is not None else []
    summary_rows = list(existing_summary.to_dict("records")) if existing_summary is not None else []

    for i, (ry, rq, dy, dq) in enumerate(pairs):
        if (ry, rq, dy, dq) in done_pairs:
            print(f"\nPair {i + 1}/{len(pairs)}: recipient {ry}Q{rq} <- donor {dy}Q{dq} -- already done, skipping")
            continue
        print(f"\n{'=' * 70}\nPair {i + 1}/{len(pairs)}: recipient {ry}Q{rq} <- donor {dy}Q{dq}\n{'=' * 70}")
        rows = run_pair(state, args.variable, ry, rq, dy, dq, dy, dq, args.swap_groups,
                         ckpt_dir_fn, prefix_fn, data_root_fn, band_mean, band_std, in_channels,
                         args.device, args.batch_size, args.img_size)
        if rows is None:
            print(f"  skipping pair {i + 1} (0 overlapping test tracts)")
            continue
        all_rows.extend(rows)

        by_condition = {r["condition"]: r for r in rows}
        baseline_r2 = by_condition["baseline"]["r2"]
        intervention_r2 = by_condition["intervention"]["r2"]
        donor_r2 = by_condition["donor_own_diagonal_context"]["r2"]
        frac = recovery_fraction(baseline_r2, intervention_r2, donor_r2)
        summary_rows.append({
            "recipient_year": ry, "recipient_quarter": rq, "donor_year": dy, "donor_quarter": dq,
            "baseline_r2": baseline_r2, "intervention_r2": intervention_r2, "donor_r2": donor_r2,
            "delta_r2": intervention_r2 - baseline_r2, "recovery_fraction": frac,
        })
        frac_str = f"{frac:.1%}" if pd.notna(frac) else "n/a (donor_r2 <= baseline_r2)"
        print(f"  recovery_fraction={frac_str}")

        pd.DataFrame(all_rows).to_csv(long_path, index=False)
        pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
        print(f"  saved progress: {len(summary_rows)} pair(s) so far in {summary_path}")

    if not summary_rows:
        raise SystemExit("Every pair was skipped -- nothing to summarize.")

    summary_df = pd.DataFrame(summary_rows)
    print(f"\nWrote {long_path} ({len(all_rows)} rows)")
    print(f"Wrote {summary_path} ({len(summary_df)} pairs)")

    valid = summary_df["recovery_fraction"].dropna()
    n_excluded = len(summary_df) - len(valid)
    print(f"\nRecovery fraction across {len(valid)} pair(s){f' ({n_excluded} excluded, donor_r2 <= baseline_r2)' if n_excluded else ''}:")
    if len(valid):
        print(f"  mean={valid.mean():.1%}  median={valid.median():.1%}  "
              f"min={valid.min():.1%}  max={valid.max():.1%}")
    else:
        print("  (no pair had a usable denominator -- nothing to average)")


if __name__ == "__main__":
    main()
