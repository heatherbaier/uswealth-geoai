"""
Collects copies of every cross-quarter preds CSV (the per-tract
GEOID/pred/label files sail's task: validate already wrote, same ones
build_cross_quarter_r2_matrix.py reads) into one organized folder, so
they're easy to hand off to a collaborator -- no recomputation, nothing
deleted from where they already live.

Layout written under --out-dir:
    <out-dir>/<state>/q<quarter>_<year>/epoch*_valset_*_preds.csv

one subfolder per (state, model_year, model_quarter) -- i.e. per
trained model -- containing every one of that model's own cross-quarter
preds CSVs (one per imagery quarter it was evaluated against, found
under that model's own ckpt_dir/cross_quarter/ subfolder -- see
build_cross_quarter_r2_matrix.py's docstring for why they live there).
A model with no trained checkpoint, or no cross-quarter CSVs yet, is
skipped with a message, not a crash.

Usage:
    python 4_analysis/collect_cross_quarter_csvs.py \
        --states az ca oh pa --variable wealth_index \
        --quarters 2016Q1 2016Q2 2016Q3 2016Q4 2017Q1 2017Q2 2017Q3 2017Q4 \
                   2018Q1 2018Q2 2018Q3 2018Q4 2019Q1 2019Q2 2019Q3 2019Q4 \
        --out-dir ./cross_quarter_csvs
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry  # noqa: E402
from generate_cross_validate_config import parse_imagery_quarter, CROSS_QUARTER_SUBDIR  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_cross_quarter_r2_matrix import model_ckpt_dir  # noqa: E402


def collect_state(state, variable, quarters, version, registry, out_dir: Path):
    n_copied_total = 0
    for year, quarter in quarters:
        try:
            ckpt_dir = Path(model_ckpt_dir(state, variable, year, quarter, version, registry))
        except SystemExit as e:
            print(f"[{state} Q{quarter} {year}] no trained model found, skipping: {e}")
            continue

        csv_dir = ckpt_dir / CROSS_QUARTER_SUBDIR
        csvs = sorted(csv_dir.glob("epoch*_preds.csv")) if csv_dir.is_dir() else []
        if not csvs:
            print(f"[{state} Q{quarter} {year}] no cross-quarter CSVs under {csv_dir}, skipping")
            continue

        dest_dir = out_dir / state / f"q{quarter}_{year}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in csvs:
            shutil.copy2(csv_path, dest_dir / csv_path.name)
        print(f"[{state} Q{quarter} {year}] copied {len(csvs)} CSV(s) -> {dest_dir}")
        n_copied_total += len(csvs)

    return n_copied_total


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--states", nargs="+", required=True)
    p.add_argument("--variable", required=True)
    p.add_argument("--quarters", required=True, nargs="+",
                    help="YYYYQN tokens, e.g. 2016Q1 2016Q2 2016Q3 2016Q4 -- same set for every state")
    p.add_argument("--version", default=None,
                    help="Trained version to use for every model (default: latest existing, per model)")
    p.add_argument("--out-dir", default="./cross_quarter_csvs", type=Path)
    args = p.parse_args()

    quarters = [parse_imagery_quarter(t) for t in args.quarters]
    registry = load_registry()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    for state in [s.lower() for s in args.states]:
        grand_total += collect_state(state, args.variable, quarters, args.version, registry, args.out_dir)

    print(f"\nDone. {grand_total} CSV(s) copied into {args.out_dir}")


if __name__ == "__main__":
    main()
