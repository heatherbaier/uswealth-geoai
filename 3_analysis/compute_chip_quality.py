"""
Per-chip composite-quality proxy, computed directly from already-
downloaded raw chips -- no new imagery download. Fills the "composite
quality" covariate (valid observation count, cloud fraction, snow
fraction) amplitude_covariates.py's regression needs, which nothing else
in this project currently tracks: geoetl's download configs set
mask_clouds: False and only fetch the 12 analysis bands (see
generate_download_config.py) -- no Sentinel-2 SCL/QA60 scene-
classification layer is ever downloaded, so there's no real per-pixel
cloud/snow QA data to read. These are deliberately cheap, approximate
proxies computed from the 12 bands already on disk, not real QA data --
good enough to use as a REGRESSION CONTROL (soak up "this composite
looked unusually hazy/snowy" variance) but not to trust as a precise
cloud-cover percentage. If these controls end up mattering a lot in
amplitude_covariates.py's results, that itself is a signal worth getting
real Sentinel-2 SCL data for, not something to over-read from this proxy
alone.

geoetl writes chips as uint16 "reflectance x 10000", nodata=0 (same
convention check_null_imagery.py/compute_pixel_diversity.py document) --
resolve_chip_dir is imported directly from check_null_imagery.py rather
than reimplemented. Band order is the same 12-band "_allbands" list
band_importance.py uses (B01..B12, generate_download_config.py's
authoritative order) -- NOT compute_pixel_diversity.py's SENSOR_BANDS,
which documents an older/different RGB-only "sentinel2" download that
predates this project's "_allbands" convention and isn't what these
trained models' chips actually are.

Per chip:
  valid_obs_frac   fraction of pixels that are nonzero in EVERY band
                    pooled together (same metric check_null_imagery.py
                    calls nonzero_frac) -- 1.0 - this is the "missing/
                    nodata" share of the chip.
  snow_frac         fraction of valid pixels with NDSI = (B03-B11)/
                    (B03+B11) above --snow-ndsi-threshold (0.4 is the
                    standard remote-sensing snow threshold).
  cloud_frac        fraction of valid pixels whose mean visible
                    reflectance (B02,B03,B04) exceeds --cloud-
                    reflectance-threshold (clouds/thick haze are much
                    brighter in the visible than almost any clear-sky
                    ground cover).

Output: one row per (state, year, quarter, GEOID) --
    <out-dir>/<state>_chip_quality.csv

Usage:
    python 3_analysis/compute_chip_quality.py --state oh --years 2017 2018 2019
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from check_null_imagery import resolve_chip_dir  # noqa: E402

# Same authoritative 12-band order as band_importance.py -- NOT
# reimported from there to avoid pulling in sail/torch for a script that
# only needs rasterio/numpy.
BAND_NAMES = ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
              "B08", "B8A", "B09", "B11", "B12"]
B02_IDX, B03_IDX, B04_IDX, B11_IDX = (BAND_NAMES.index(b) for b in ("B02", "B03", "B04", "B11"))


def chip_quality(path: Path, snow_ndsi_threshold: float, cloud_reflectance_threshold: float):
    with rasterio.open(path) as src:
        data = src.read()  # (12, H, W), uint16, nodata=0

    if data.shape[0] != len(BAND_NAMES):
        raise ValueError(f"{path}: expected {len(BAND_NAMES)} bands, got {data.shape[0]}")

    valid = np.all(data != 0, axis=0)  # (H, W) -- nonzero in EVERY band
    n_total = valid.size
    n_valid = int(valid.sum())
    valid_obs_frac = n_valid / n_total if n_total else float("nan")

    if n_valid == 0:
        return {"valid_obs_frac": 0.0, "snow_frac": float("nan"), "cloud_frac": float("nan")}

    reflectance = data.astype("float32") / 10000.0
    green, swir1 = reflectance[B03_IDX][valid], reflectance[B11_IDX][valid]
    ndsi_denom = green + swir1
    ndsi = np.where(ndsi_denom > 0, (green - swir1) / np.where(ndsi_denom > 0, ndsi_denom, 1.0), 0.0)
    is_snow = ndsi > snow_ndsi_threshold
    snow_frac = float(np.mean(is_snow))

    # Snow is also bright in the visible bands, so it would otherwise
    # trip the cloud heuristic too (every snow pixel double-counted as
    # "cloudy") -- classify snow first and only ask "is this NON-snow
    # pixel unusually bright" for cloud, the same snow-before-cloud
    # priority real QA layers (e.g. Sentinel-2's SCL) use. cloud_frac is
    # a share of ALL valid pixels, not just non-snow ones, so the two
    # fractions stay comparable/summable as "fraction of this chip
    # affected by X".
    visible_mean = np.mean([reflectance[B02_IDX][valid], reflectance[B03_IDX][valid],
                             reflectance[B04_IDX][valid]], axis=0)
    is_cloud = ~is_snow & (visible_mean > cloud_reflectance_threshold)
    cloud_frac = float(np.mean(is_cloud))

    return {"valid_obs_frac": valid_obs_frac, "snow_frac": snow_frac, "cloud_frac": cloud_frac}


def process_quarter(state: str, year: int, quarter: int, snow_ndsi_threshold: float,
                     cloud_reflectance_threshold: float):
    try:
        chip_dir = resolve_chip_dir(state, year, quarter)
    except SystemExit as e:
        print(f"[{state} Q{quarter} {year}] {e}, skipping")
        return []
    paths = sorted(chip_dir.glob("*.tif"))
    if not paths:
        print(f"[{state} Q{quarter} {year}] no chips under {chip_dir}, skipping")
        return []

    rows = []
    for path in tqdm(paths, desc=f"{state} Q{quarter} {year}"):
        try:
            q = chip_quality(path, snow_ndsi_threshold, cloud_reflectance_threshold)
        except Exception as e:
            print(f"  {path.name}: couldn't read, skipping ({e})")
            continue
        rows.append({"state": state, "year": year, "quarter": quarter, "GEOID": path.stem, **q})
    print(f"[{state} Q{quarter} {year}] {len(rows)}/{len(paths)} chips processed")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--snow-ndsi-threshold", type=float, default=0.4,
                    help="Standard remote-sensing NDSI snow threshold (default 0.4)")
    p.add_argument("--cloud-reflectance-threshold", type=float, default=0.3,
                    help="Mean visible-band reflectance above which a pixel is flagged bright/"
                         "hazy (default 0.3 -- clouds/thick haze run well above typical clear-"
                         "sky ground cover; tune if this looks off for your states)")
    p.add_argument("--out-dir", default="3_analysis/outputs/amplitude")
    args = p.parse_args()

    state = args.state.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for year in args.years:
        for quarter in args.quarters:
            rows += process_quarter(state, year, quarter, args.snow_ndsi_threshold,
                                     args.cloud_reflectance_threshold)

    if not rows:
        raise SystemExit("No chips processed -- nothing to write.")
    df = pd.DataFrame(rows)
    out_path = out_dir / f"{state}_chip_quality.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df):,} rows)")
    print(df[["valid_obs_frac", "snow_frac", "cloud_frac"]].describe())


if __name__ == "__main__":
    main()
