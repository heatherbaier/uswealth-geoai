"""
Compute per-chip pixel-value diversity ("flatness") metrics from raw
Sentinel-2 (or Landsat) GeoTIFF chips.

Why this has to run on the raw .tif, not the training PNGs: convert_tifs_to_pngs.py
stretches each chip independently to its own 2nd-98th percentile before mapping
to uint8. That per-image normalization would make a spectrally flat winter chip
and a spectrally rich summer chip both use the full 0-255 range -- exactly the
signal this analysis needs to preserve. Point --chip-dir at the original uint16
GeoTIFFs used to produce the PNGs, not the PNGs themselves.

Output is one row per chip with:
    name                          full path to the chip (join key downstream)
    mean_band_std                 mean, across bands, of the per-band std dev
    mean_band_range               mean, across bands, of the (p2, p98) range
    band{i}_std / band{i}_range_2_98   per-band detail
    ndvi_std / ndvi_mean / ndvi_range  only if this sensor's band map defines
                                        both "red" and "nir"

Usage:
    python compute_pixel_diversity.py \
        --chip-dir /data/hbaier/new_data/tlag/imagery/q1_2017_s2/chips \
        --sensor sentinel2 \
        --out ./diversity/q1_2017.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm


# Band name -> index within each chip's band stack.
# sentinel2 here only maps RGB, same as convert_tifs_to_pngs.py's BAND_ORDER.
# <<< If your Sentinel-2 chips also carry B08 (NIR), add "nir": <index> below
# so NDVI diversity gets computed -- that's the more direct test of the
# "vegetation signal" theory than raw RGB spread.
SENSOR_BANDS = {
    "sentinel2": {"red": 0, "green": 1, "blue": 2},
    "landsat8":  {"swir22": 0, "swir16": 1, "nir": 2, "red": 3, "green": 4, "blue": 5},
    "landsat5":  {"nir": 2, "red": 3, "green": 4, "blue": 5},  # <<< verify vs mpc.py
}

LOW_PCT, HIGH_PCT = 2, 98  # match the stretch percentiles used for the PNGs


def chip_diversity(path, band_map):
    with rasterio.open(path) as src:
        data = src.read().astype("float64")  # (bands, H, W)

    band_stds, band_ranges, per_band = [], [], {}
    for b in range(data.shape[0]):
        band = data[b]
        valid = band[band > 0]  # 0 = nodata, same convention as convert_tifs_to_pngs.py
        if valid.size < 10:
            continue
        std = float(np.std(valid))
        lo, hi = np.percentile(valid, [LOW_PCT, HIGH_PCT])
        rng = float(hi - lo)
        band_stds.append(std)
        band_ranges.append(rng)
        per_band[f"band{b}_std"] = std
        per_band[f"band{b}_range_{LOW_PCT}_{HIGH_PCT}"] = rng

    if not band_stds:
        return None

    out = {
        "name": str(path),
        "mean_band_std": float(np.mean(band_stds)),
        "mean_band_range": float(np.mean(band_ranges)),
        **per_band,
    }

    if "red" in band_map and "nir" in band_map:
        red = data[band_map["red"]]
        nir = data[band_map["nir"]]
        denom = nir + red
        valid = denom > 0
        ndvi = np.full_like(denom, np.nan)
        ndvi[valid] = (nir[valid] - red[valid]) / denom[valid]
        ndvi_valid = ndvi[~np.isnan(ndvi)]
        if ndvi_valid.size >= 10:
            lo, hi = np.percentile(ndvi_valid, [LOW_PCT, HIGH_PCT])
            out["ndvi_std"] = float(np.std(ndvi_valid))
            out["ndvi_mean"] = float(np.mean(ndvi_valid))
            out["ndvi_range"] = float(hi - lo)

    return out


def compute_directory(chip_dir, sensor, out_csv, pattern="*.tif"):
    chip_dir = Path(chip_dir)
    band_map = SENSOR_BANDS[sensor]
    paths = sorted(chip_dir.glob(pattern))
    print(f"Found {len(paths)} chips in {chip_dir}")
    if "nir" not in band_map:
        print(f"  Note: no NIR band configured for sensor={sensor}; "
              f"NDVI diversity metrics will be skipped.")

    rows = []
    n_fail = 0
    for p in tqdm(paths):
        try:
            row = chip_diversity(p, band_map)
            if row is not None:
                rows.append(row)
        except Exception as e:
            print(f"  Failed {p.name}: {e}")
            n_fail += 1

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Wrote {len(df):,} rows to {out_csv} ({n_fail} failed)")
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chip-dir", required=True, help="Directory of raw .tif chips for one quarter")
    p.add_argument("--sensor", required=True, choices=list(SENSOR_BANDS))
    p.add_argument("--out", required=True, help="Output CSV path")
    p.add_argument("--pattern", default="*.tif", help="Glob pattern for chip files")
    args = p.parse_args()
    compute_directory(args.chip_dir, args.sensor, args.out, args.pattern)


if __name__ == "__main__":
    main()
