"""
Compute per-tract ESA WorldCover land-cover fractions for a state, ready to
use as the `lc.csv` input to 4_analysis/analyze_seasonality_lag.py,
analyze_pixel_diversity_accuracy.py (H1c), and analyze_band_importance_spatial.py.

Generalizes what used to be OH-only lc.py / compute_landcover_oh.py:
  - --state selects the tract shapefile instead of a hardcoded OH path.
  - Downloads *every* ESA WorldCover grid tile that intersects the state's
    AOI (the old script opened exactly one hardcoded tile filename after
    downloading all of them -- fine by luck for a state that happens to
    fit in one 3x3-degree tile, silently wrong/incomplete for any state
    whose tracts span more than one tile, which is most of them). Each
    tract is now clipped against whichever tile(s) it actually falls in
    and counts are summed across tiles.
  - Converts ESA's numeric class codes directly to the named fraction
    columns (Cropland, Built-up, ...) the analysis scripts expect, instead
    of leaving that as a manual follow-up step (this was the "no script
    does the JSON -> lc.csv conversion" gap noted in REPLICATION.md).

Usage:
    python compute_landcover.py --state oh
    python compute_landcover.py --state az --tract-shp ./data/tracts2019/AZ/tl_2019_04_tract_wi.shp

Output: ./lc_<state>.csv -- GEOID index, one fraction column per ESA
WorldCover class actually observed in this state (columns are Cropland,
Built-up, Tree cover, etc; a class with zero pixels anywhere is omitted).
Also writes the raw per-class pixel counts to <out-dir>/raw_counts_<state>.json
for debugging / re-deriving with different class groupings later.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import geopandas as gpd
import numpy as np
import rioxarray as rxr
import requests
from tqdm.auto import tqdm

S3_URL_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# ESA WorldCover 10m class legend (https://esa-worldcover.org/en/data-access
# -- "Product validation and specification" documents the class codes).
ESA_CLASS_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# Default tract shapefile per state -- matches what clean_wealth_index.py
# writes (for az/ca/ga/pa) or the pre-existing OH file (no _wi equivalent
# for OH -- see REPLICATION.md).
STATE_FIPS = {"OH": "39", "AZ": "04", "CA": "06", "GA": "13", "PA": "42"}
DEFAULT_SHP = {
    "OH": "./data/tracts2019/OH/tl_2019_39_tract_proj.shp",
    "AZ": "./data/tracts2019/AZ/tl_2019_04_tract_wi.shp",
    "CA": "./data/tracts2019/CA/tl_2019_06_tract_wi.shp",
    "GA": "./data/tracts2019/GA/tl_2019_13_tract_wi.shp",
    "PA": "./data/tracts2019/PA/tl_2019_42_tract_wi.shp",
}


def download_tiles(tiles, year, output_folder: Path):
    version = {2020: "v100", 2021: "v200"}[year]
    output_folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for tile in tqdm(tiles.ll_tile, desc="Downloading WorldCover tiles"):
        url = f"{S3_URL_PREFIX}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
        out_fn = output_folder / Path(url).name
        if out_fn.exists():
            paths.append(out_fn)
            continue
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(out_fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        paths.append(out_fn)
    return paths


def tract_class_counts(tract_geom, tile_gdf, tile_paths_by_ll):
    """Sum per-class pixel counts for one tract across every WorldCover
    tile it intersects (a tract can straddle a tile boundary)."""
    hits = tile_gdf[tile_gdf.intersects(tract_geom)]
    total = Counter()
    for ll_tile in hits.ll_tile:
        path = tile_paths_by_ll.get(ll_tile)
        if path is None:
            continue
        with rxr.open_rasterio(path, masked=True, chunks=True) as src:
            try:
                clipped = src.rio.clip([tract_geom], from_disk=True)
            except Exception:
                continue
            unique, counts = np.unique(clipped, return_counts=True, equal_nan=True)
            for u, c in zip(unique, counts):
                if not np.isnan(u):
                    total[int(u)] += int(c)
    return total


def compute_landcover(state: str, tract_shp: Path, year: int, out_dir: Path, out_csv: Path):
    state = state.upper()
    print(f"[{state}] Loading tracts: {tract_shp}")
    tracts = gpd.read_file(tract_shp)
    geom_all = tracts.dissolve().geometry.iloc[0]

    print("Loading WorldCover grid index...")
    grid = gpd.read_file(f"{S3_URL_PREFIX}/esa_worldcover_grid.geojson")
    tiles = grid[grid.intersects(geom_all)]
    print(f"[{state}] {len(tiles)} WorldCover tile(s) intersect this state's AOI")

    tile_paths = download_tiles(tiles, year, out_dir / "wc_tiles")
    tile_paths_by_ll = {p.stem.split("_")[-2]: p for p in tile_paths}

    val_dict = {}
    for _, row in tqdm(tracts.iterrows(), total=len(tracts), desc=f"[{state}] Clipping tracts"):
        try:
            counts = tract_class_counts(row.geometry, tiles, tile_paths_by_ll)
            if counts:
                val_dict[str(int(row.GEOID))] = dict(counts)
        except Exception as e:
            print(f"  {row.GEOID}: failed ({e})")

    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / f"raw_counts_{state.lower()}.json"
    with open(raw_path, "w") as f:
        json.dump(val_dict, f)
    print(f"[{state}] Wrote raw per-class pixel counts: {raw_path}")

    classes_present = sorted({int(k) for counts in val_dict.values() for k in counts})
    unknown = [c for c in classes_present if c not in ESA_CLASS_NAMES]
    if unknown:
        print(f"  Note: unrecognized WorldCover class codes present, "
              f"skipping from named columns: {unknown}")

    rows = {}
    for geoid, counts in val_dict.items():
        total = sum(v for k, v in counts.items() if int(k) in ESA_CLASS_NAMES)
        if total == 0:
            continue
        rows[geoid] = {
            ESA_CLASS_NAMES[c]: counts.get(c, 0) / total
            for c in classes_present if c in ESA_CLASS_NAMES
        }

    import pandas as pd
    lc = pd.DataFrame.from_dict(rows, orient="index").fillna(0.0)
    lc.index.name = "GEOID"
    lc.to_csv(out_csv)
    print(f"[{state}] Wrote {out_csv} ({len(lc):,} tracts x {lc.shape[1]} land-cover classes)")
    return lc


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, choices=sorted(STATE_FIPS) + [s.lower() for s in STATE_FIPS])
    p.add_argument("--tract-shp", type=Path, default=None,
                    help="Tract shapefile to use (default: looked up from --state, see DEFAULT_SHP)")
    p.add_argument("--year", type=int, default=2021, choices=[2020, 2021],
                    help="WorldCover product year (2021 = v200, 2020 = v100)")
    p.add_argument("--out-dir", type=Path, default=Path("./wc"),
                    help="Where to cache downloaded tiles + raw counts JSON")
    p.add_argument("--out-csv", type=Path, default=None,
                    help="Output lc.csv path (default: ./lc_<state>.csv)")
    args = p.parse_args()

    state = args.state.upper()
    tract_shp = args.tract_shp or Path(DEFAULT_SHP[state])
    out_csv = args.out_csv or Path(f"./lc_{state.lower()}.csv")
    if not tract_shp.exists():
        raise SystemExit(
            f"Tract shapefile not found: {tract_shp}\n"
            f"Run download_acs_tracts.py (and, for az/ca/ga/pa, "
            f"clean_wealth_index.py) first, or pass --tract-shp explicitly."
        )
    compute_landcover(state, tract_shp, args.year, args.out_dir, out_csv)


if __name__ == "__main__":
    main()
