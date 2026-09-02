"""
Sanity-check a state's trained checkpoints: R^2, mean signed error (bias),
and error std dev per quarter, loaded straight from sail's `task: validate`
output (epoch<N>_preds.csv / epoch<N>_valset_preds.csv / epoch<N>_full_preds.csv
-- see sail/src/sail/engine.py::run_validation).

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
    python validate.py --state az --imagery-root /data/hbaier/new_data/tlag/az_imagery \
        --out-dir ./out_validation/az
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def discover_quarter_preds(imagery_root: Path, preds_glob: str):
    quarters = []
    for qdir in sorted(imagery_root.glob("q[1-4]_????_s2*")):
        m = re.match(r"q([1-4])_(\d{4})_s2", qdir.name)
        if not m:
            continue
        quarter, year = int(m.group(1)), int(m.group(2))
        matches = sorted(qdir.glob(preds_glob))
        if len(matches) != 1:
            print(f"Skipping {qdir.name}: found {len(matches)} preds CSVs "
                  f"matching '{preds_glob}' (expected exactly 1) -- run "
                  f"sail's task: validate for this quarter first")
            continue
        quarters.append({"year": year, "quarter": quarter, "preds_csv": matches[0]})
    quarters.sort(key=lambda q: (q["year"], q["quarter"]))
    return quarters


def validate_state(state, imagery_root, preds_glob, out_dir):
    quarters = discover_quarter_preds(imagery_root, preds_glob)
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
    p.add_argument("--preds-glob", default="artifacts/*/epoch*preds.csv",
                    help="Glob (relative to each q<n>_<year>_s2*/ dir) for the validate-task output CSV")
    p.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir (default: ./out_validation/<state>)")
    args = p.parse_args()
    out_dir = args.out_dir or Path(f"./out_validation/{args.state.lower()}")
    validate_state(args.state, args.imagery_root, args.preds_glob, out_dir)


if __name__ == "__main__":
    main()
