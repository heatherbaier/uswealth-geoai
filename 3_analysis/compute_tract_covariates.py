"""
Step 3 of the seasonal-amplitude pipeline: one row of tract-level
covariates per GEOID, joined from what already exists in the repo (per
the earlier covariate-availability review -- see that conversation for
the full gap list) plus the two small additions made to support it:

  wealth_index            --wealth-csv's own wealth_index column (the
                            ACS ground truth, NOT a model prediction --
                            deliberately not reusing amplitude_
                            predictions.py's own `label` column here,
                            since using a model's own target as a
                            covariate to explain variance IN that same
                            model's predictions would be circular)
  renter_pct               1 - --wealth-csv's owner_pct
  population_density       --wealth-csv's race_total_pop (added to
                            download_acs.py's B03002 pull specifically
                            for this) / tract area from --tract-shp's
                            ALAND -- a density-based urbanicity proxy,
                            the agreed substitute for a RUCA-code join
  white_pct/black_pct/     --wealth-csv's white_nonhispanic/
  asian_pct/hispanic_pct   black_nonhispanic/asian_nonhispanic/
                            hispanic_latino (same B03002 pull) /
                            race_total_pop
  tree_canopy_pct          --lc-csv's "Tree cover" fraction (ESA
  cropland_pct             WorldCover, via compute_landcover.py -- the
                            agreed substitute for NLCD; "Cropland")
  tract_area_km2           --tract-shp's ALAND (square meters) / 1e6
  valid_obs_frac_mean       --chip-quality-csv (compute_chip_quality.py),
  cloud_frac_mean           MEAN across every (year, quarter) row present
  snow_frac_mean            for that GEOID -- these are genuinely PER-
                            QUARTER quantities, but the amplitude
                            regression (step 4) is one row per TRACT, so
                            they're collapsed to a single tract-level
                            quality summary here. Mean, not e.g. worst-
                            quarter, is the assumption -- flagged
                            explicitly since it's a real modeling choice,
                            not a neutral default.

GEOID normalization: --lc-csv's GEOID went through compute_landcover.py's
`str(int(row.GEOID))`, which drops a leading zero for AZ/CA's state FIPS
(04/06) -- the same bug class flagged throughout this project (see
update_ys_labels.py's normalize_geoid docstring). Every join key here
(wealth CSV's geoid, lc.csv's GEOID, the shapefile's GEOID, chip-quality
CSV's GEOID) is run through that SAME normalize_geoid function rather
than trusted as-is, specifically because of that known corruption source.

Assumptions (stated per the original request):
  - --wealth-csv is the same per-state ACS-derived file update_ys_
    labels.py/plot_wealth_variables.py already use, now including the
    race_total_pop/white_nonhispanic/black_nonhispanic/asian_nonhispanic/
    hispanic_latino columns added to wealth/2_code/download_acs.py (run
    that + wealth/2_code/run_wealth_index.py again to regenerate it with
    those columns if you haven't already).
  - Tract geometry/area comes from state_registry.yml's shp_path (the
    same _wi.shp clean_wealth_index.py produces) via --use-state-registry,
    or an explicit --tract-shp -- either way it's assumed to carry
    ALAND/AWATER (standard TIGER/Line fields, not dropped by any step in
    this pipeline).
  - --lc-csv is optional (ESA WorldCover, not NLCD -- the agreed
    substitute); --chip-quality-csv is optional (the band-derived proxy,
    not real Sentinel-2 QA data). Either missing just drops those
    columns from the output with a warning, not a crash -- step 4's
    regression can still run without them, just without those controls.

Usage:
    python 3_analysis/compute_tract_covariates.py --state oh \
        --wealth-csv ./wealth_indices_2019_OH.csv --use-state-registry \
        --lc-csv ./lc_oh.csv --chip-quality-csv ./oh_chip_quality.csv
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry, resolve_state_settings  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "1_data_prep"))
from update_ys_labels import normalize_geoid  # noqa: E402

RACE_COLUMNS = {
    "white_pct": "white_nonhispanic", "black_pct": "black_nonhispanic",
    "asian_pct": "asian_nonhispanic", "hispanic_pct": "hispanic_latino",
}


def load_wealth_covariates(wealth_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(wealth_csv, dtype=str)
    if "geoid" not in df.columns:
        raise SystemExit(f"--wealth-csv {wealth_csv} has no 'geoid' column. Available: {list(df.columns)}")
    df["GEOID"] = df["geoid"].apply(normalize_geoid)

    numeric_needed = ["wealth_index", "owner_pct", "race_total_pop"] + list(RACE_COLUMNS.values())
    missing = [c for c in numeric_needed if c not in df.columns]
    if missing:
        raise SystemExit(
            f"--wealth-csv {wealth_csv} is missing: {missing}.\n"
            f"race_total_pop/white_nonhispanic/black_nonhispanic/asian_nonhispanic/"
            f"hispanic_latino need wealth/2_code/download_acs.py's B03002 addition --"
            f"re-run download_acs.py + run_wealth_index.py if this file predates that change.\n"
            f"Available columns: {list(df.columns)}"
        )
    for c in numeric_needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    out = pd.DataFrame({
        "GEOID": df["GEOID"],
        "wealth_index": df["wealth_index"],
        "renter_pct": 1.0 - df["owner_pct"],
        "race_total_pop": df["race_total_pop"],
    })
    for out_col, src_col in RACE_COLUMNS.items():
        out[out_col] = df[src_col] / df["race_total_pop"]
    return out.drop_duplicates(subset="GEOID")


def load_landcover_covariates(lc_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lc_csv)
    geoid_col = "GEOID" if "GEOID" in df.columns else df.columns[0]
    df["GEOID"] = df[geoid_col].apply(normalize_geoid)
    out = pd.DataFrame({"GEOID": df["GEOID"]})
    out["tree_canopy_pct"] = df["Tree cover"] if "Tree cover" in df.columns else np.nan
    out["cropland_pct"] = df["Cropland"] if "Cropland" in df.columns else np.nan
    if "Tree cover" not in df.columns or "Cropland" not in df.columns:
        print(f"  WARNING: {lc_csv} is missing 'Tree cover' and/or 'Cropland' (no pixels of "
              f"that class observed anywhere in this state) -- left as NaN, columns present: "
              f"{[c for c in df.columns if c != geoid_col]}")
    return out.drop_duplicates(subset="GEOID")


def load_tract_area(shp_path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(shp_path)
    if "GEOID" not in gdf.columns:
        raise SystemExit(f"{shp_path} has no GEOID column -- not a standard TIGER/Line tract shapefile?")
    if "ALAND" not in gdf.columns:
        raise SystemExit(f"{shp_path} has no ALAND column -- can't compute tract_area_km2/"
                          f"population_density without it.")
    # TIGER/Line tracts are distributed in NAD83 geographic coords
    # (EPSG:4269) -- centroid .x/.y are already lon/lat in degrees, close
    # enough to WGS84 at state scale for amplitude_covariates.py's
    # Moran's I (same "simple planar approximation, fine at this scale"
    # tradeoff analyze_band_importance_spatial.py's own _build_knn_
    # weights already makes, not reimplemented more precisely here).
    centroid = gdf.geometry.centroid
    out = pd.DataFrame({
        "GEOID": gdf["GEOID"].apply(normalize_geoid),
        "tract_area_km2": pd.to_numeric(gdf["ALAND"], errors="coerce") / 1e6,
        "lon": centroid.x, "lat": centroid.y,
    })
    return out.drop_duplicates(subset="GEOID")


def load_chip_quality(chip_quality_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(chip_quality_csv, dtype={"GEOID": str})
    df["GEOID"] = df["GEOID"].apply(normalize_geoid)
    agg = df.groupby("GEOID", as_index=False)[["valid_obs_frac", "cloud_frac", "snow_frac"]].mean()
    agg = agg.rename(columns={"valid_obs_frac": "valid_obs_frac_mean", "cloud_frac": "cloud_frac_mean",
                               "snow_frac": "snow_frac_mean"})
    return agg


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--wealth-csv", required=True, type=Path,
                    help="Per-state ACS-derived wealth CSV (same file update_ys_labels.py uses)")
    p.add_argument("--lc-csv", type=Path, default=None,
                    help="compute_landcover.py output for this state (ESA WorldCover fractions). "
                         "Optional -- tree_canopy_pct/cropland_pct are left out if omitted.")
    p.add_argument("--chip-quality-csv", type=Path, default=None,
                    help="compute_chip_quality.py output for this state. Optional -- quality "
                         "controls are left out if omitted.")
    p.add_argument("--use-state-registry", action="store_true",
                    help="Get the tract shapefile path from state_registry.yml's shp_path "
                         "instead of --tract-shp")
    p.add_argument("--tract-shp", type=Path, default=None,
                    help="Explicit tract shapefile path (must have GEOID + ALAND), instead of "
                         "--use-state-registry")
    p.add_argument("--out-dir", default="3_analysis/outputs/amplitude")
    args = p.parse_args()

    if not args.use_state_registry and args.tract_shp is None:
        raise SystemExit("Pass either --use-state-registry or --tract-shp")

    state = args.state.lower()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.use_state_registry:
        registry = load_registry()
        st = resolve_state_settings(state, registry)
        shp_path = Path(st["shp_path"])
    else:
        shp_path = args.tract_shp

    print(f"[{state.upper()}] wealth covariates from {args.wealth_csv}")
    covariates = load_wealth_covariates(args.wealth_csv)
    print(f"  {len(covariates):,} tracts")

    print(f"[{state.upper()}] tract area from {shp_path}")
    area = load_tract_area(shp_path)
    covariates = covariates.merge(area, on="GEOID", how="left")
    covariates["population_density"] = covariates["race_total_pop"] / covariates["tract_area_km2"]

    if args.lc_csv is not None:
        print(f"[{state.upper()}] land cover from {args.lc_csv}")
        lc = load_landcover_covariates(args.lc_csv)
        covariates = covariates.merge(lc, on="GEOID", how="left")
    else:
        print(f"[{state.upper()}] no --lc-csv given -- tree_canopy_pct/cropland_pct left out")
        covariates["tree_canopy_pct"] = np.nan
        covariates["cropland_pct"] = np.nan

    if args.chip_quality_csv is not None:
        print(f"[{state.upper()}] chip quality from {args.chip_quality_csv}")
        quality = load_chip_quality(args.chip_quality_csv)
        covariates = covariates.merge(quality, on="GEOID", how="left")
    else:
        print(f"[{state.upper()}] no --chip-quality-csv given -- quality controls left out")
        covariates["valid_obs_frac_mean"] = np.nan
        covariates["cloud_frac_mean"] = np.nan
        covariates["snow_frac_mean"] = np.nan

    covariates.insert(0, "state", state)
    covariates.insert(2, "county_fips", covariates["GEOID"].str[:5])
    out_path = out_dir / f"{state}_tract_covariates.csv"
    covariates.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(covariates):,} tracts, {covariates.shape[1]} columns)")
    n_missing = covariates[["tree_canopy_pct", "cropland_pct", "valid_obs_frac_mean"]].isna().all().to_dict()
    print(f"Columns entirely NaN (no source given): "
          f"{[c for c, v in n_missing.items() if v] or 'none'}")


if __name__ == "__main__":
    main()
