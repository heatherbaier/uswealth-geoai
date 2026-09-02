"""
Assemble the wide-format preds.csv that analyze_seasonality_lag.py needs
(GEOID, q{n}_{year}_pred, q{n}_{year}_error columns) from a state's
per-quarter sail `task: validate` output CSVs (long format: name/pred/label,
one row per chip -- see sail/src/sail/engine.py::run_validation).

This is the "no script assembles preds.csv" gap noted in REPLICATION.md.
GEOID comes directly from the chip filename stem (geoetl writes chips to
<output.root>/chips/{GEOID}.{tif,png}, see compute_pixel_diversity.py's
docstring for the same convention) -- no separate join needed.

Usage:
    python assemble_preds_wide.py --state az \
        --imagery-root /data/hbaier/new_data/tlag/az_imagery \
        --out ./preds_az.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


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
                  f"matching '{preds_glob}' (expected exactly 1)")
            continue
        quarters.append({"year": year, "quarter": quarter, "preds_csv": matches[0]})
    quarters.sort(key=lambda q: (q["year"], q["quarter"]))
    return quarters


def assemble(state, imagery_root, preds_glob, out_csv):
    quarters = discover_quarter_preds(imagery_root, preds_glob)
    if not quarters:
        raise SystemExit(f"No quarters found under {imagery_root} matching '{preds_glob}'")
    print(f"[{state}] Found {len(quarters)} validated quarters")

    wide = None
    for q in quarters:
        df = pd.read_csv(q["preds_csv"])
        col_prefix = f"q{q['quarter']}_{q['year']}"
        part = pd.DataFrame({
            "GEOID": df["name"].apply(lambda p: Path(p).stem),
            f"{col_prefix}_pred": df["pred"],
            f"{col_prefix}_error": df["pred"] - df["label"],
        })
        n_before = len(part)
        part = part.drop_duplicates(subset="GEOID")
        if len(part) != n_before:
            print(f"  {col_prefix}: dropped {n_before - len(part)} duplicate GEOID rows")

        wide = part if wide is None else wide.merge(part, on="GEOID", how="outer")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_csv, index=False)
    print(f"[{state}] Wrote {out_csv} ({len(wide):,} tracts x {len(quarters)} quarters)")
    return wide


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State label (used for logging only)")
    p.add_argument("--imagery-root", required=True, type=Path,
                    help="geoetl tlag imagery root for this state, e.g. .../tlag/az_imagery")
    p.add_argument("--preds-glob", default="artifacts/*/epoch*preds.csv",
                    help="Glob (relative to each q<n>_<year>_s2*/ dir) for the validate-task output CSV")
    p.add_argument("--out", required=True, type=Path, help="Output wide-format preds CSV path")
    args = p.parse_args()
    assemble(args.state, args.imagery_root, args.preds_glob, args.out)


if __name__ == "__main__":
    main()
