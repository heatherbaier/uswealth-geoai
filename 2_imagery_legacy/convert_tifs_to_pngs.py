"""
Convert 6-band uint16 GeoTIFFs from the geoetl MPC pipeline into 3-band uint8
RGB PNGs so they can be read by PIL/torchvision without any dataloader changes.

Uses percentile-based contrast stretching (2nd-98th percentile) so the output
PNGs look like reasonable natural-color images regardless of the input's
reflectance distribution.

Usage:
    python convert_tifs_to_pngs.py \\
        --src-dir /data/.../chips \\
        --dst-dir /data/.../chips_png \\
        --sensor sentinel2

Sensor argument determines which of the 6 bands are the RGB bands (band
order comes from mpc.py SENSOR_CONFIGS):
    landsat8, landsat5: bands are [swir22, swir16, nir08, red, green, blue]
                        → RGB is band indices 3, 4, 5
    sentinel2:          bands are [B04, B03, B02] = [red, green, blue]
                        → RGB is band indices 0, 1, 2
"""

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image


BAND_ORDER = {
    "sentinel2": (0, 1, 2),   # B04=red, B03=green, B02=blue
    "landsat8":  (3, 4, 5),   # red, green, blue at indices 3, 4, 5
    "landsat5":  (2, 3, 4),   # L5 band order in mpc.py is different — see cfg
}


def stretch_to_uint8(band, low_pct=2, high_pct=98):
    """Percentile-based contrast stretch to uint8."""
    valid = band[band > 0]
    if valid.size == 0:
        return np.zeros_like(band, dtype=np.uint8)
    lo, hi = np.percentile(valid, [low_pct, high_pct])
    if hi <= lo:
        return np.zeros_like(band, dtype=np.uint8)
    out = np.clip((band - lo) / (hi - lo) * 255, 0, 255)
    return out.astype(np.uint8)


def convert_one(src_path, dst_path, sensor):
    with rasterio.open(src_path) as src:
        data = src.read()   # (bands, H, W)

    r_idx, g_idx, b_idx = BAND_ORDER[sensor]
    r = stretch_to_uint8(data[r_idx])
    g = stretch_to_uint8(data[g_idx])
    b = stretch_to_uint8(data[b_idx])

    rgb = np.stack([r, g, b], axis=-1)   # (H, W, 3)
    Image.fromarray(rgb, mode="RGB").save(dst_path, optimize=True)


def convert_directory(src_dir, dst_dir, sensor, skip_existing=True):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    tifs = sorted(src_dir.glob("*.tif"))
    print(f"Found {len(tifs)} .tif files in {src_dir}")
    print(f"Writing PNGs to {dst_dir}")
    print(f"Sensor: {sensor}, RGB band indices: {BAND_ORDER[sensor]}")

    n_ok = n_skip = n_fail = 0
    for i, src_path in enumerate(tifs, start=1):
        dst_path = dst_dir / (src_path.stem + ".png")
        if skip_existing and dst_path.exists():
            n_skip += 1
            continue
        try:
            convert_one(src_path, dst_path, sensor)
            n_ok += 1
        except Exception as e:
            print(f"  Failed {src_path.name}: {e}")
            n_fail += 1
        if i % 100 == 0:
            print(f"  {i}/{len(tifs)}  ok={n_ok} skipped={n_skip} failed={n_fail}")

    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src-dir", required=True, help="Directory of input .tif files")
    p.add_argument("--dst-dir", required=True, help="Directory to write .png files")
    p.add_argument("--sensor", required=True, choices=list(BAND_ORDER),
                   help="Sensor type — determines which bands are RGB")
    p.add_argument("--overwrite", action="store_true",
                   help="Re-convert files that already exist in dst_dir")
    args = p.parse_args()

    convert_directory(
        args.src_dir,
        args.dst_dir,
        args.sensor,
        skip_existing=not args.overwrite,
    )


if __name__ == "__main__":
    main()