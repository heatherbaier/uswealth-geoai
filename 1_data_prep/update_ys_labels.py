"""
Regenerate sail-format <prefix>_ys.json (and a matching <prefix>_coords.json)
for one or more already-downloaded quarters, using a new wealth-index CSV
from a partner instead of whatever geoetl originally wrote to label_column
-- without re-downloading any imagery.

Why this is needed: sail's dataset.prefix convention requires a matching
<prefix>_ys.json and <prefix>_coords.json pair in the same data_root (see
sail/src/sail/data/adapters.py::resolve_json_paths). Coordinates don't
change per target variable -- same chip, same location -- only labels do.
So one new prefix (one new ys.json + a copied coords.json) gets written
per wealth_index_* target, giving you a separate, directly-trainable
dataset per variable without touching imagery or coords at all.

For each --data-root given:
  1. Auto-discovers the existing prefix (the one <data_root>/*_ys.json
     present) -- used only to read the chip path list + coords, not the
     old label values (those get replaced entirely).
  2. Extracts each chip's GEOID from its filename (this project's
     established convention: a chip is named "{GEOID}.tif").
  3. Joins against --wealth-csv by GEOID, both sides normalized (stripped,
     digits only, zero-padded to 11 -- the standard state+county+tract
     GEOID width). A raw-string join has already bitten this project once
     via a dropped leading zero on AZ's state FIPS "04" -- don't repeat it.
  4. For each --target column, writes <data_root>/<old_prefix>_<target>_ys.json
     (chip_path -> float(value), explicitly cast so a stray string/NaN in
     the source CSV can't repeat sail's "must be real number, not str"
     crash) and a copy of coords.json under the same new prefix.

Chips whose GEOID doesn't match any CSV row are dropped (reported, not
silently included) -- same "absent = not in this quarter's dataset"
convention SimbaJSONDataset already uses for missing imagery.

Usage:
    python update_ys_labels.py \
        --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/ \
        --wealth-csv ./partner_acs_wealth.csv

Then point a sail train config's dataset.prefix at e.g.
"az_2016_q1_s2_allbands_wealth_index_sat" for that target --
dataset.data_root stays exactly the same.
"""

import argparse
import csv
import glob
import json
import os

DEFAULT_TARGETS = [
    "wealth_index",
    "wealth_index_sat",
    "wealth_index_housing_core",
    "wealth_index_transport",
    "wealth_index_financial",
    "wealth_index_utilities",
]


def item_key(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name.strip()))[0]


def normalize_geoid(raw) -> "str | None":
    """Digits only, zero-padded to 11 (state+county+tract). Handles a
    stray trailing '.0' from a CSV column that got read/re-saved as a
    float somewhere upstream, and a dropped leading zero on single-digit
    state FIPS codes (AZ=04, CA=06, etc.)."""
    s = str(raw).strip()
    if s.endswith(".0"):
        s = s[:-2]
    s = "".join(ch for ch in s if ch.isdigit())
    if not s:
        return None
    return s.zfill(11)


def load_wealth_csv(path: str, targets: list) -> dict:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing_cols = [t for t in targets if t not in fieldnames]
        if missing_cols:
            raise SystemExit(f"--wealth-csv is missing columns: {missing_cols}\n"
                              f"Available columns: {fieldnames}")
        if "geoid" not in fieldnames:
            raise SystemExit(f"--wealth-csv has no 'geoid' column. Available: {fieldnames}")
        rows = {}
        n_bad_geoid = 0
        for row in reader:
            g = normalize_geoid(row["geoid"])
            if g is None:
                n_bad_geoid += 1
                continue
            rows[g] = row
    if n_bad_geoid:
        print(f"  ({n_bad_geoid} CSV rows had an unparseable geoid, skipped)")
    return rows


def discover_prefix(data_root: str) -> str:
    matches = sorted(glob.glob(os.path.join(data_root, "*_ys.json")))
    if len(matches) != 1:
        raise SystemExit(f"{data_root}: expected exactly 1 *_ys.json, found {len(matches)}: {matches}")
    return os.path.basename(matches[0])[: -len("_ys.json")]


def process_data_root(data_root: str, wealth_rows: dict, targets: list):
    old_prefix = discover_prefix(data_root)
    with open(os.path.join(data_root, f"{old_prefix}_ys.json")) as f:
        old_ys = json.load(f)
    with open(os.path.join(data_root, f"{old_prefix}_coords.json")) as f:
        old_coords = json.load(f)

    chip_paths = sorted(set(old_ys) & set(old_coords))

    matched, unmatched = {}, []
    for path in chip_paths:
        g = normalize_geoid(item_key(path))
        if g in wealth_rows:
            matched[path] = wealth_rows[g]
        else:
            unmatched.append(path)

    print(f"[{old_prefix}] {len(chip_paths)} chips, {len(matched)} matched to --wealth-csv, "
          f"{len(unmatched)} unmatched (dropped)")
    if unmatched:
        print(f"  sample unmatched GEOIDs: {[item_key(p) for p in unmatched[:5]]}")

    for target in targets:
        ys_out, coords_out = {}, {}
        n_bad_value = 0
        for path, row in matched.items():
            raw_val = row.get(target, "")
            try:
                val = float(raw_val)
            except (TypeError, ValueError):
                n_bad_value += 1
                continue
            ys_out[path] = val
            coords_out[path] = old_coords[path]

        new_prefix = f"{old_prefix}_{target}"
        ys_path = os.path.join(data_root, f"{new_prefix}_ys.json")
        coords_path = os.path.join(data_root, f"{new_prefix}_coords.json")
        with open(ys_path, "w") as f:
            json.dump(ys_out, f)
        with open(coords_path, "w") as f:
            json.dump(coords_out, f)

        vals = list(ys_out.values())
        if vals:
            print(f"  [{target}] wrote {len(ys_out)} chips (dropped {n_bad_value} non-numeric) "
                  f"-- min={min(vals):.4f} mean={sum(vals)/len(vals):.4f} max={max(vals):.4f}  "
                  f"prefix={new_prefix!r}")
        else:
            print(f"  [{target}] WARNING: 0 chips written -- check the column name / CSV values")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-roots", nargs="+", required=True,
                    help="One or more already-downloaded quarters' dataset.data_root dirs")
    p.add_argument("--wealth-csv", required=True, help="Partner's new ACS-derived wealth CSV")
    p.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                    help=f"Column(s) to generate a ys.json for (default: all 6 wealth_index* "
                         f"variants: {DEFAULT_TARGETS})")
    args = p.parse_args()

    print(f"Loading {args.wealth_csv}...")
    wealth_rows = load_wealth_csv(args.wealth_csv, args.targets)
    print(f"Loaded {len(wealth_rows)} GEOIDs\n")

    for data_root in args.data_roots:
        process_data_root(data_root, wealth_rows, args.targets)
        print()


if __name__ == "__main__":
    main()
