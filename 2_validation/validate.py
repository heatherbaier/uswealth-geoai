"""
Sanity-check a state's trained checkpoints: R^2, mean signed error (bias),
and error std dev per quarter, loaded straight from sail's `task: validate`
output (epoch<N>_valset_<dataset-prefix>_preds.csv /
epoch<N>_full_<dataset-prefix>_preds.csv -- see
sail/src/sail/engine.py::run_validation; the dataset-prefix segment was
added in sail PR #23, so a config generated/validated before that PR
produced the shorter epoch<N>_valset_preds.csv name instead -- re-run
task: validate once to pick up the new name if --preds-glob below finds
nothing for an older quarter).

Only ever finds ONE match per quarter directory: this glob is
non-recursive and looks directly under artifacts/<experiment>/, not
artifacts/<experiment>/cross_quarter/ -- cross-quarter generalization
runs (generate_cross_validate_config.py) deliberately write there instead
(sail PR #25's validator.output_subdir) specifically so they don't show
up here and break the "exactly 1 match = this model's own held-out test
set" assumption this script depends on.

Once more than one target variable has been trained for a state, each
quarter's artifacts/ has one experiment dir PER variable (exp_base =
f"{state}_q{quarter}_{year}_{variable}", same convention as
build_cross_quarter_r2_matrix.py/analyze_transfer_penalty_landcover.py),
so the default glob's `artifacts/*/` now matches all of them --
pass --variable to pick one (narrows the glob to that experiment dir
specifically) instead of silently trying to validate every variable's
preds at once.

Replaces the per-state validation_oh.ipynb / validation_az.ipynb notebooks
(which were the same ~35 cells copy-pasted with different hardcoded paths,
and a validation_az.ipynb left half-finished mid-copy) with one script
driven by --state / --imagery-root. Also fixes a bug those notebooks had:
their x-axis tick labels were a hardcoded 15-entry list assuming every
state has exactly q2_2016..q4_2019 trained -- wrong for any state with a
different quarter range or fewer quarters trained (which is every state
except OH right now). Ticks here are built from whatever quarters were
actually found.

Usage:
    python validate.py --state oh --imagery-root /data/hbaier/new_data/tlag/imagery

    # Multiple variables trained for this state -- pick one:
    python validate.py --state az --imagery-root /data/hbaier/new_data/tlag/az_imagery \
        --variable wealth_index_sat --out-dir ./out_validation/az_wealth_index_sat
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Experiment dir naming convention shared with build_cross_quarter_r2_matrix.py /
# analyze_transfer_penalty_landcover.py: artifacts/<state>_q<quarter>_<year>_<variable>_v<N>/
EXP_DIR_RE = re.compile(r"^.+_q\d_\d{4}_(?P<variable>.+)_v\d+$")


def discover_quarter_preds(imagery_root: Path, preds_glob: str, variable: "str | None"):
    quarters = []
    for qdir in sorted(imagery_root.glob("q[1-4]_????_s2*")):
        m = re.match(r"q([1-4])_(\d{4})_s2", qdir.name)
        if not m:
            continue
        quarter, year = int(m.group(1)), int(m.group(2))
        matches = sorted(qdir.glob(preds_glob))
        if len(matches) > 1 and not variable:
            # artifacts/*/ matched one experiment dir per trained variable
            # (see EXP_DIR_RE) -- list what's actually available instead
            # of guessing, or silently mixing variables' preds together.
            found_vars = sorted({
                vm.group("variable") for p in matches
                if (vm := EXP_DIR_RE.match(p.parent.name))
            })
            hint = f" ({', '.join(found_vars)})" if found_vars else ""
            print(f"Skipping {qdir.name}: found {len(matches)} preds CSVs matching "
                  f"'{preds_glob}' -- multiple variables trained here{hint}; "
                  f"pass --variable to pick one")
            continue
        if len(matches) != 1:
            print(f"Skipping {qdir.name}: found {len(matches)} preds CSVs "
                  f"matching '{preds_glob}' (expected exactly 1) -- run "
                  f"sail's task: validate for this quarter first")
            continue
        quarters.append({"year": year, "quarter": quarter, "preds_csv": matches[0]})
    quarters.sort(key=lambda q: (q["year"], q["quarter"]))
    return quarters


def validate_state(state, imagery_root, preds_glob, out_dir, variable=None):
    quarters = discover_quarter_preds(imagery_root, preds_glob, variable)
    if not quarters:
        raise SystemExit(f"No quarters found under {imagery_root} matching '{preds_glob}'")
    print(f"[{state}] Found {len(quarters)} validated quarters")

    rows = []
    for q in quarters:
        df = pd.read_csv(q["preds_csv"])
        r2 = r2_score(df["label"], df["pred"])
        bias = float(np.mean(df["pred"] - df["label"]))
        std = float(np.std(df["pred"] - df["label"]))
        n = len(df)
        label = f"q{q['quarter']}_{q['year']}"
        print(f"  {label}: r2={r2:.4f} bias={bias:+.4f} std={std:.4f} n={n} ({q['preds_csv']})")
        rows.append({"year": q["year"], "quarter": q["quarter"], "label": label,
                      "r2": r2, "bias": bias, "std": std, "n": n})

    summary = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"validation_summary_{state.lower()}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[{state}] Wrote {summary_path}")

    x = range(len(summary))
    for col, ylabel, fname in [
        ("r2", "R^2", "r2_by_quarter.png"),
        ("bias", "Mean signed error (pred - truth)", "bias_by_quarter.png"),
        ("std", "Std dev of signed error", "std_by_quarter.png"),
    ]:
        fig, ax = plt.subplots(figsize=(15, 8))
        if col == "bias":
            ax.axhline(0, color="black", lw=1, alpha=0.6)
        ax.plot(x, summary[col], "-o")
        ax.set_xticks(list(x))
        ax.set_xticklabels(summary["label"], rotation=45, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{state.upper()}: {ylabel} by quarter")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=150)
        plt.close()
    print(f"[{state}] Wrote r2/bias/std plots to {out_dir}")

    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State label, e.g. oh/az/ca/ga/pa (used for output naming/titles only)")
    p.add_argument("--imagery-root", required=True, type=Path,
                    help="geoetl tlag imagery root for this state, e.g. .../tlag/az_imagery")
    p.add_argument("--variable", default=None,
                    help="Target variable to validate, e.g. wealth_index/wealth_index_sat -- "
                         "required once more than one variable has been trained for this state "
                         "(each gets its own artifacts/<...>_<variable>_v<N>/ experiment dir, so "
                         "the default --preds-glob then matches more than one)")
    p.add_argument("--preds-glob", default=None,
                    help="Glob (relative to each q<n>_<year>_s2*/ dir) for the validate-task "
                         "output CSV. Default: artifacts/*/epoch*_valset_*_preds.csv, narrowed to "
                         "artifacts/*_<variable>_v*/epoch*_valset_*_preds.csv when --variable is given")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: ./out_validation/<state>, or "
                         "./out_validation/<state>_<variable> when --variable is given)")
    args = p.parse_args()

    preds_glob = args.preds_glob
    if preds_glob is None:
        preds_glob = (f"artifacts/*_{args.variable}_v*/epoch*_valset_*_preds.csv" if args.variable
                      else "artifacts/*/epoch*_valset_*_preds.csv")

    if args.out_dir is not None:
        out_dir = args.out_dir
    elif args.variable:
        out_dir = Path(f"./out_validation/{args.state.lower()}_{args.variable}")
    else:
        out_dir = Path(f"./out_validation/{args.state.lower()}")

    validate_state(args.state, args.imagery_root, preds_glob, out_dir, variable=args.variable)


if __name__ == "__main__":
    main()
