"""
One row per tract (GEOID), one column per (quarter, metric), summarizing
each quarter's composite imagery -- a wide-format companion to compute_
chip_quality.py's long-format per-chip output, for a quick per-tract look
at how a state's remote-sensing signal moves across seasons without
needing to load a model at all.

Reuses compute_chip_quality.py's quality_from_array() for valid_obs_frac/
cloud_frac/snow_frac (same NDSI/visible-brightness proxies, same snow-
before-cloud priority -- see that script's docstring for why these are
approximate, not real Sentinel-2 QA data) rather than reimplementing
them, and reads each chip exactly ONCE (band array + valid mask shared
between the quality computation and the indices below), not once per
metric.

Metrics per (GEOID, year, quarter), each the MEAN over valid pixels:
  ndvi      (B08-B04)/(B08+B04)   vegetation greenness -- the standard
                                   growing-season signal this whole
                                   project's seasonality question is
                                   ultimately about.
  ndwi      (B03-B08)/(B03+B08)   surface water/moisture (McFeeters).
  ndmi      (B08-B11)/(B08+B11)   vegetation/canopy water content --
                                   NOT also reporting NDBI (built-up
                                   index) alongside this: NDBI is
                                   (B11-B08)/(B11+B08), the exact
                                   NEGATION of NDMI by construction
                                   (same two bands, flipped sign) --
                                   reporting both would be reporting the
                                   same number twice. Low ndmi reads as
                                   "more built-up/bare" just as directly
                                   as a separately-labeled NDBI would.
  b01_mean, b02_mean, b09_mean     raw mean reflectance (not an index) --
                                   included specifically because band_
                                   importance.py's real runs on this
                                   project's models flagged B02 (visible/
                                   blue) and the atmospheric group (B01,
                                   B09) as unexpectedly large contributors
                                   to wealth predictions, including one
                                   quarter where B02 alone nearly matched
                                   the model's full baseline R^2 -- having
                                   these bands' own seasonal trajectory
                                   alongside NDVI/snow/cloud makes it
                                   possible to check whether a quarter's
                                   outsized B02/atmospheric reliance
                                   lines up with an actual swing in that
                                   band's raw signal, or looks more like
                                   a modeling artifact.
  valid_obs_frac, cloud_frac,      from compute_chip_quality.py --
  snow_frac                       composite completeness/haze/snow proxy.

Column naming: q<quarter>_<year>_<metric>, e.g. q1_2018_ndvi -- year is
ALWAYS included (not just "q1_ndvi") even for a single --years value, so
a later run adding more years never collides with or shadows an earlier
one's columns.

Incremental save: writes the full output CSV again after EVERY (year,
quarter) finishes, not just at the end -- open it anytime mid-run to see
whichever quarters are done so far; columns for quarters not yet reached
simply aren't there yet.

Usage:
    python 3_analysis/compute_quarterly_rs_summary.py --state oh --years 2018
    python 3_analysis/compute_quarterly_rs_summary.py --state oh --years 2017 2018 2019
"""

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from check_null_imagery import resolve_chip_dir  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_chip_quality import BAND_NAMES, quality_from_array  # noqa: E402

B04_IDX, B08_IDX = BAND_NAMES.index("B04"), BAND_NAMES.index("B08")
B03_IDX, B11_IDX = BAND_NAMES.index("B03"), BAND_NAMES.index("B11")
B01_IDX, B02_IDX, B09_IDX = BAND_NAMES.index("B01"), BAND_NAMES.index("B02"), BAND_NAMES.index("B09")

METRICS = ["ndvi", "ndwi", "ndmi", "b01_mean", "b02_mean", "b09_mean",
           "valid_obs_frac", "cloud_frac", "snow_frac"]


def _safe_ratio(numerator, denominator):
    """(a-b)/(a+b)-style index, valid-pixels-only, guarding the
    extremely rare all-zero-in-both-bands case (denominator 0)."""
    denom = numerator + denominator
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(denom != 0, (numerator - denominator) / denom, 0.0)
    return ratio


def chip_rs_summary(path: Path, snow_ndsi_threshold: float, cloud_reflectance_threshold: float):
    with rasterio.open(path) as src:
        data = src.read()  # (12, H, W), uint16, nodata=0
    if data.shape[0] != len(BAND_NAMES):
        raise ValueError(f"expected {len(BAND_NAMES)} bands, got {data.shape[0]}")

    valid = np.all(data != 0, axis=0)
    quality = quality_from_array(data, snow_ndsi_threshold, cloud_reflectance_threshold, valid=valid)
    if int(valid.sum()) == 0:
        return {**quality, "ndvi": np.nan, "ndwi": np.nan, "ndmi": np.nan,
                "b01_mean": np.nan, "b02_mean": np.nan, "b09_mean": np.nan}

    reflectance = data.astype("float32") / 10000.0
    red, nir, green, swir1 = (reflectance[B04_IDX][valid], reflectance[B08_IDX][valid],
                               reflectance[B03_IDX][valid], reflectance[B11_IDX][valid])

    ndvi = float(np.mean(_safe_ratio(nir, red)))
    ndwi = float(np.mean(_safe_ratio(green, nir)))
    ndmi = float(np.mean(_safe_ratio(nir, swir1)))
    b01_mean = float(np.mean(reflectance[B01_IDX][valid]))
    b02_mean = float(np.mean(reflectance[B02_IDX][valid]))
    b09_mean = float(np.mean(reflectance[B09_IDX][valid]))

    return {**quality, "ndvi": ndvi, "ndwi": ndwi, "ndmi": ndmi,
            "b01_mean": b01_mean, "b02_mean": b02_mean, "b09_mean": b09_mean}


def process_quarter(state: str, year: int, quarter: int, snow_ndsi_threshold: float,
                     cloud_reflectance_threshold: float) -> pd.DataFrame:
    try:
        chip_dir = resolve_chip_dir(state, year, quarter)
    except SystemExit as e:
        print(f"[{state} Q{quarter} {year}] {e}, skipping")
        return pd.DataFrame(columns=["GEOID"] + METRICS)
    paths = sorted(chip_dir.glob("*.tif"))
    if not paths:
        print(f"[{state} Q{quarter} {year}] no chips under {chip_dir}, skipping")
        return pd.DataFrame(columns=["GEOID"] + METRICS)

    rows = []
    for path in tqdm(paths, desc=f"{state} Q{quarter} {year}"):
        try:
            summary = chip_rs_summary(path, snow_ndsi_threshold, cloud_reflectance_threshold)
        except Exception as e:
            print(f"  {path.name}: couldn't read, skipping ({e})")
            continue
        rows.append({"GEOID": path.stem, **summary})
    print(f"[{state} Q{quarter} {year}] {len(rows)}/{len(paths)} chips processed")
    return pd.DataFrame(rows, columns=["GEOID"] + METRICS)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--years", nargs="+", type=int, required=True)
    p.add_argument("--quarters", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    p.add_argument("--snow-ndsi-threshold", type=float, default=0.4)
    p.add_argument("--cloud-reflectance-threshold", type=float, default=0.3)
    p.add_argument("--out-dir", default="3_analysis/outputs/rs_summary")
    args = p.parse_args()

    state = args.state.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{state}_quarterly_rs_summary.csv"

    wide = pd.DataFrame(columns=["GEOID"]).set_index("GEOID")

    for year, quarter in itertools.product(args.years, args.quarters):
        long_df = process_quarter(state, year, quarter, args.snow_ndsi_threshold,
                                   args.cloud_reflectance_threshold)
        if long_df.empty:
            continue
        long_df = long_df.set_index("GEOID")
        prefix = f"q{quarter}_{year}_"
        long_df = long_df.rename(columns={m: f"{prefix}{m}" for m in METRICS})
        wide = wide.combine_first(long_df) if not wide.empty else long_df

        wide_out = wide.reset_index().rename(columns={"index": "GEOID"})
        wide_out.insert(0, "state", state)
        wide_out.to_csv(out_path, index=False)
        print(f"  -> saved {out_path} ({len(wide_out):,} tracts, {wide_out.shape[1]} columns so far)\n")

    if wide.empty:
        raise SystemExit("No quarters produced any rows -- nothing was written.")
    print(f"Done. Final: {out_path} ({len(wide):,} tracts, {wide.shape[1] + 2} columns)")


if __name__ == "__main__":
    main()
