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
  - Reads each WorldCover tile in bounded pixel blocks (BLOCK_PX) rather
    than one array covering a tract's whole bounding box. Statewide tract
    shapefiles mix small urban tracts with rare, enormous rural ones (AZ's
    Coconino County tracts are the canonical example -- geoetl hit and
    documented the identical problem in geoetl/io/mpc.py), and even a
    "windowed"/from_disk read materializes the whole bounding box for
    those outliers, which OOM-kills the process. Peak memory here is
    bounded per-block regardless of tract size; a hard MAX_TRACT_PIXELS
    cap (skipped + logged to skipped_oversized_tracts_<state>.txt) is a
    belt-and-suspenders safety valve for a truly degenerate geometry.

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
import rasterio
import requests
from rasterio.features import geometry_mask
from rasterio.windows import Window
from tqdm.auto import tqdm

S3_URL_PREFIX = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# Read each tile in bounded pixel blocks instead of one array covering the
# tract's whole bounding box. Mirrors geoetl's chunk_px fix in
# geoetl/io/mpc.py::_build_composite for the identical problem: a
# statewide tract shapefile mixes small urban tracts with rare, enormous
# rural ones (geoetl's own comment there names a Coconino County, AZ tract
# as ~100x wider than a Phoenix one), so an AOI-sized bounding-box read
# that's fine for 99% of tracts can OOM on the outliers regardless of
# "windowed"/from_disk reads, because the window itself is still huge for
# those tracts. Block reads bound peak memory per tract to one block.
BLOCK_PX = 2048

# Hard safety cap, mirroring geoetl's AOITooLargeError / max_composite_mb:
# even block-by-block, a truly degenerate geometry (e.g. a bad multipolygon
# spanning the whole state) shouldn't be allowed to grind forever. ~800M
# uint8 pixels is a ~280km x 280km AOI at 10m resolution -- far bigger than
# any real census tract, so this should never actually trigger on good data.
MAX_TRACT_PIXELS = 800_000_000

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


def _iter_blocks(window: Window, block_px: int):
    """Tile a rasterio Window into block_px x block_px sub-windows (absolute
    pixel offsets into the source raster), last row/col clipped to fit."""
    row0, col0 = int(window.row_off), int(window.col_off)
    height, width = int(window.height), int(window.width)
    for r in range(0, height, block_px):
        for c in range(0, width, block_px):
            h = min(block_px, height - r)
            w = min(block_px, width - c)
            yield Window(col0 + c, row0 + r, w, h)


def tract_class_counts(tract_geom, tile_gdf, tile_paths_by_ll, on_skip=None):
    """Sum per-class pixel counts for one tract across every WorldCover
    tile it intersects (a tract can straddle a tile boundary), reading each
    tile in bounded BLOCK_PX x BLOCK_PX blocks so peak memory doesn't scale
    with tract size (see BLOCK_PX comment above)."""
    hits = tile_gdf[tile_gdf.intersects(tract_geom)]
    total = Counter()
    for ll_tile in hits.ll_tile:
        path = tile_paths_by_ll.get(ll_tile)
        if path is None:
            continue
        with rasterio.open(path) as src:
            try:
                window = rasterio.windows.from_bounds(*tract_geom.bounds, transform=src.transform)
            except Exception:
                continue
            window = window.intersection(Window(0, 0, src.width, src.height))
            if window.width <= 0 or window.height <= 0:
                continue
            n_px = window.width * window.height
            if n_px > MAX_TRACT_PIXELS:
                if on_skip:
                    on_skip(f"tile window ~{window.width:.0f}x{window.height:.0f} px "
                            f"(~{n_px / 1e6:.0f}M) exceeds MAX_TRACT_PIXELS")
                continue

            for block in _iter_blocks(window, BLOCK_PX):
                data = src.read(1, window=block)
                block_transform = src.window_transform(block)
                inside = geometry_mask([tract_geom], out_shape=data.shape,
                                        transform=block_transform, invert=True)
                vals = data[inside]
                if vals.size:
                    u, c = np.unique(vals, return_counts=True)
                    for uu, cc in zip(u, c):
                        total[int(uu)] += int(cc)
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

    out_dir.mkdir(parents=True, exist_ok=True)
    skipped_path = out_dir / f"skipped_oversized_tracts_{state.lower()}.txt"
    skipped_geoids = set()

    val_dict = {}
    for _, row in tqdm(tracts.iterrows(), total=len(tracts), desc=f"[{state}] Clipping tracts"):
        geoid = str(int(row.GEOID))
        try:
            def _log_skip(reason, geoid=geoid):
                skipped_geoids.add(geoid)
                with open(skipped_path, "a") as f:
                    f.write(f"{geoid}\t{reason}\n")

            counts = tract_class_counts(row.geometry, tiles, tile_paths_by_ll, on_skip=_log_skip)
            if counts:
                val_dict[geoid] = dict(counts)
        except Exception as e:
            print(f"  {geoid}: failed ({e})")

    if skipped_geoids:
        print(f"[{state}] Skipped {len(skipped_geoids)} oversized tract(s) "
              f"(logged to {skipped_path}) -- these tracts are missing from "
              f"lc.csv. Cross-check against geoetl's skipped_oversized_aois.txt "
              f"for this state; if they overlap, those tracts have no imagery "
              f"either and are already excluded by the analysis scripts' joins.")

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
    if tract_shp.is_dir():
        raise SystemExit(
            f"--tract-shp {tract_shp} is a directory, not a .shp file. If it "
            f"contains multiple tract layers (e.g. both "
            f"tl_2019_{STATE_FIPS[state]}_tract.shp and "
            f"tl_2019_{STATE_FIPS[state]}_tract_wi.shp), geopandas will "
            f"silently pick one of them rather than the one you meant. Point "
            f"this at the .shp file directly, e.g.:\n"
            f"  --tract-shp {tract_shp / f'tl_2019_{STATE_FIPS[state]}_tract_wi.shp'}"
        )
    if not tract_shp.exists():
        raise SystemExit(
            f"Tract shapefile not found: {tract_shp}\n"
            f"Run download_acs_tracts.py (and, for az/ca/ga/pa, "
            f"clean_wealth_index.py) first, or pass --tract-shp explicitly."
        )
    compute_landcover(state, tract_shp, args.year, args.out_dir, out_csv)


if __name__ == "__main__":
    main()
