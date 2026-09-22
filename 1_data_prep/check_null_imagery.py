"""
Count "bad" (null/empty) chips in a downloaded state/year/quarter of
imagery -- the geoetl MPC pipeline (geoetl/io/mpc.py + geoetl/io/base.py)
writes chips as uint16 "reflectance x 10000" with nodata=0 (see
4_analysis/compute_pixel_diversity.py's docstring for the same convention),
so a chip that's nodata=0 in EVERY band, every pixel is a chip with no
real imagery in it at all -- exactly what "downloading as null data" looks
like on disk.

Why chips like this can still exist even after the fix in geoetl (PR #14:
raise instead of writing an all-NaN composite as a clean-looking all-zero
GeoTIFF; also stop a partial/corrupt write from masquerading as a valid
cache): pipeline.py's AOI loop skips any AOI whose chip path already
exists (`if os.path.exists(clip_path): continue`) -- it never re-checks
whether that existing chip is actually good. A chip written by a run from
BEFORE that fix, or one left over from any other write failure, sits there
forever looking "downloaded" and gets silently reused by every later run
and by sail's dataloader. This script is the audit step: find those, then
delete them so the next geoetl run (same config, same idempotent
skip-if-exists behavior) rebuilds only the actually-missing/bad ones
instead of the whole quarter.

Two "bad" categories, checked per chip:
  unreadable    rasterio can't open it at all (e.g. truncated by a killed
                process mid-write).
  null          opens fine, but the fraction of nonzero pixels (pooled
                across every band) is below --min-nonzero-frac. Default 0
                catches only a chip that's genuinely 100% nodata; raise
                --min-nonzero-frac (e.g. 0.01) to also flag chips that are
                almost entirely empty except a stray sliver of real data.

Usage:
    # Resolve the chip directory from pipeline_configs/state_registry.yml's
    # data_root_template, same convention every other pipeline_configs/
    # 4_analysis script uses:
    python 1_data_prep/check_null_imagery.py --state pa --year 2016 --quarter 1

    # Or point it straight at a chip directory (e.g. imagery downloaded
    # outside the state_registry.yml convention):
    python 1_data_prep/check_null_imagery.py --chip-dir /data/hbaier/new_data/tlag/pa_imagery/q1_2016_s2_allbands/chips

    # Write the full per-chip manifest and a bad-paths-only list (for
    # `xargs rm` cleanup) instead of just printing the summary:
    python 1_data_prep/check_null_imagery.py --state pa --year 2016 --quarter 1 \
        --out ./out_imagery_qc/pa_2016_q1.csv --bad-paths-out ./out_imagery_qc/pa_2016_q1_bad_paths.txt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from tqdm.auto import tqdm

PIPELINE_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "pipeline_configs"
REGISTRY_PATH = PIPELINE_CONFIGS_DIR / "state_registry.yml"


def resolve_chip_dir(state: str, year: int, quarter: int, registry_path=REGISTRY_PATH) -> Path:
    """Only needs data_root_template -- deliberately NOT
    generate_train_config.resolve_state_settings, which also requires
    spatial_block_deg/band_mean/band_std to be filled in. Those come from
    running find_spatial_block_deg.py/compute_shared_band_stats.py against
    already-downloaded imagery, so requiring them here would make it
    impossible to QC a state's imagery before that's done."""
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    if state not in registry:
        raise SystemExit(f"--state {state!r} not in {registry_path} (known states: {sorted(registry)})")
    template = registry[state].get("data_root_template")
    if not template:
        raise SystemExit(f"--state {state!r} has no data_root_template set in {registry_path}")
    data_root = template.format(state=state, year=year, quarter=quarter)
    return Path(data_root) / "chips"


def classify_chip(path: Path, min_nonzero_frac: float):
    """Returns (status, nonzero_frac_or_None, detail). status is one of
    'ok' / 'unreadable' / 'null'."""
    try:
        with rasterio.open(path) as src:
            data = src.read()
    except Exception as e:
        return "unreadable", None, str(e)

    total = data.size
    if total == 0:
        return "null", 0.0, "zero-size raster"

    nonzero_frac = float(np.count_nonzero(data)) / total
    if nonzero_frac <= min_nonzero_frac:
        return "null", nonzero_frac, f"nonzero pixel fraction {nonzero_frac:.6f} <= threshold {min_nonzero_frac}"
    return "ok", nonzero_frac, None


def check_directory(chip_dir: Path, pattern: str, min_nonzero_frac: float):
    paths = sorted(chip_dir.glob(pattern))
    if not paths:
        raise SystemExit(f"No chips matching {pattern!r} found under {chip_dir}")

    rows = []
    for p in tqdm(paths, desc=chip_dir.name):
        status, nonzero_frac, detail = classify_chip(p, min_nonzero_frac)
        rows.append({
            "path": str(p),
            "GEOID": p.stem,
            "status": status,
            "nonzero_frac": nonzero_frac,
            "detail": detail,
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", help="Used with --year/--quarter to resolve the chip dir from state_registry.yml")
    p.add_argument("--year", type=int)
    p.add_argument("--quarter", type=int, choices=[1, 2, 3, 4])
    p.add_argument("--chip-dir", help="Explicit chip directory, instead of --state/--year/--quarter")
    p.add_argument("--pattern", default="*.tif", help="Glob pattern for chip files (default: *.tif)")
    p.add_argument("--min-nonzero-frac", type=float, default=0.0,
                    help="Flag a chip as null if its nonzero-pixel fraction is <= this (default 0.0: "
                         "only fully-empty chips). Raise to e.g. 0.01 to also catch near-empty chips.")
    p.add_argument("--out", default=None, help="Optional: write the full per-chip manifest CSV here")
    p.add_argument("--bad-paths-out", default=None,
                    help="Optional: write one bad chip path per line here (unreadable + null), "
                         "e.g. for `xargs rm` cleanup")
    args = p.parse_args()

    if args.chip_dir:
        chip_dir = Path(args.chip_dir)
    elif args.state and args.year and args.quarter:
        chip_dir = resolve_chip_dir(args.state.lower(), args.year, args.quarter)
    else:
        raise SystemExit("Pass either --chip-dir, or all of --state/--year/--quarter")

    if not chip_dir.is_dir():
        raise SystemExit(f"Chip directory not found: {chip_dir}")

    df = check_directory(chip_dir, args.pattern, args.min_nonzero_frac)

    n_total = len(df)
    n_unreadable = int((df["status"] == "unreadable").sum())
    n_null = int((df["status"] == "null").sum())
    n_ok = n_total - n_unreadable - n_null

    print(f"\n[{chip_dir}]")
    print(f"  total:      {n_total}")
    print(f"  ok:         {n_ok} ({100 * n_ok / n_total:.2f}%)")
    print(f"  null:       {n_null} ({100 * n_null / n_total:.2f}%)")
    print(f"  unreadable: {n_unreadable} ({100 * n_unreadable / n_total:.2f}%)")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"  Wrote per-chip manifest: {out_path}")

    bad = df[df["status"] != "ok"]
    if args.bad_paths_out:
        bad_path = Path(args.bad_paths_out)
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_text("\n".join(bad["path"]) + ("\n" if len(bad) else ""))
        print(f"  Wrote {len(bad)} bad chip path(s): {bad_path}")
    elif len(bad):
        print(f"\n  Bad chips ({len(bad)}):")
        for _, row in bad.iterrows():
            print(f"    {row['status']:<10} {row['path']}  {row['detail'] or ''}")

    return 1 if (n_unreadable + n_null) else 0


if __name__ == "__main__":
    sys.exit(main())
