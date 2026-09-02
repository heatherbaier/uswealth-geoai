"""
Download US census tract shapefiles for a given ACS 5-year estimate period.

The Census Bureau's convention is that ACS 5-year estimates use the TIGER/Line
shapefile vintage corresponding to the END YEAR of the 5-year period. For
example, the 2015-2019 ACS 5-year estimate pairs with the 2019 TIGER/Line
tract shapefiles.

This script downloads tract shapefiles for all 50 states + DC (and optionally
Puerto Rico) from the Census TIGER/Line FTP archive, then optionally merges
them into a single national GeoDataFrame and writes it to disk.

Usage examples:
    # Download tracts for the 2015-2019 ACS (uses TIGER 2019)
    python download_acs_tracts.py --acs-period 2015-2019 --output-dir ./tracts_2019

    # Download tracts for 2019-2023 ACS, write a merged national file
    python download_acs_tracts.py --acs-period 2019-2023 \\
        --output-dir ./tracts_2023 --merged-output ./tracts_2023/us_tracts.gpkg

    # Skip Puerto Rico
    python download_acs_tracts.py --acs-period 2015-2019 \\
        --output-dir ./tracts --no-puerto-rico
"""

import argparse
import io
import os
import sys
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

# Census state FIPS codes. Includes the 50 states, DC, and PR.
STATE_FIPS = {
    'AL': '01', 'AK': '02', 'AZ': '04', 'AR': '05', 'CA': '06',
    'CO': '08', 'CT': '09', 'DE': '10', 'DC': '11', 'FL': '12',
    'GA': '13', 'HI': '15', 'ID': '16', 'IL': '17', 'IN': '18',
    'IA': '19', 'KS': '20', 'KY': '21', 'LA': '22', 'ME': '23',
    'MD': '24', 'MA': '25', 'MI': '26', 'MN': '27', 'MS': '28',
    'MO': '29', 'MT': '30', 'NE': '31', 'NV': '32', 'NH': '33',
    'NJ': '34', 'NM': '35', 'NY': '36', 'NC': '37', 'ND': '38',
    'OH': '39', 'OK': '40', 'OR': '41', 'PA': '42', 'RI': '44',
    'SC': '45', 'SD': '46', 'TN': '47', 'TX': '48', 'UT': '49',
    'VT': '50', 'VA': '51', 'WA': '53', 'WV': '54', 'WI': '55',
    'WY': '56', 'PR': '72',
}

TIGER_BASE_URL = "https://www2.census.gov/geo/tiger/TIGER{year}/TRACT"


def parse_acs_period(period: str) -> tuple[int, int]:
    """Parse an ACS period string like '2015-2019' into (start, end)."""
    try:
        start_str, end_str = period.split('-')
        start, end = int(start_str), int(end_str)
    except (ValueError, AttributeError):
        raise ValueError(
            f"ACS period '{period}' is not in expected format 'YYYY-YYYY'"
        )
    if end - start != 4:
        raise ValueError(
            f"ACS 5-year period must span exactly 5 years; got "
            f"{period} (span = {end - start + 1} years). Did you mean a "
            f"3-year period or include the wrong endpoints?"
        )
    if start < 2005:
        raise ValueError(
            f"ACS 5-year estimates start with 2005-2009; period {period} "
            f"is before that."
        )
    return start, end


def tiger_year_for_acs(acs_period: str) -> int:
    """
    Return the TIGER/Line vintage year that pairs with the given ACS 5-year
    period. Census convention: use the end year of the ACS period.
    """
    _, end_year = parse_acs_period(acs_period)
    return end_year


def tract_filename(tiger_year: int, state_fips: str) -> str:
    """Build the standard TIGER/Line tract filename for a state and year."""
    return f"tl_{tiger_year}_{state_fips}_tract.zip"


def tract_url(tiger_year: int, state_fips: str) -> str:
    """Build the full URL for a state's tract shapefile."""
    base = TIGER_BASE_URL.format(year=tiger_year)
    return f"{base}/{tract_filename(tiger_year, state_fips)}"


def download_one_state(
    state_abbrev: str,
    state_fips: str,
    tiger_year: int,
    output_dir: Path,
    overwrite: bool = False,
    timeout: int = 120,
    max_retries: int = 3,
) -> Optional[Path]:
    """
    Download and extract one state's tract shapefile.

    Returns the path to the extracted .shp file, or None if the download
    failed after retries.
    """
    url = tract_url(tiger_year, state_fips)
    zip_name = tract_filename(tiger_year, state_fips)
    state_dir = output_dir / state_abbrev
    shp_path = state_dir / zip_name.replace('.zip', '.shp')

    if shp_path.exists() and not overwrite:
        print(f"  [{state_abbrev}] Already exists, skipping: {shp_path.name}")
        return shp_path

    state_dir.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, max_retries + 1):
        try:
            print(f"  [{state_abbrev}] Downloading (attempt {attempt})...")
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            content = r.content
            # Extract in-memory to the state directory
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                z.extractall(state_dir)
            if not shp_path.exists():
                # The shapefile component may have a different inner name
                shp_candidates = list(state_dir.glob('*.shp'))
                if not shp_candidates:
                    raise RuntimeError(
                        f"No .shp file found after extracting {zip_name}"
                    )
                shp_path = shp_candidates[0]
            print(f"  [{state_abbrev}] OK: {shp_path.name}")
            return shp_path
        except (requests.RequestException, zipfile.BadZipFile, RuntimeError) as e:
            print(f"  [{state_abbrev}] Attempt {attempt} failed: {e}")
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [{state_abbrev}] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [{state_abbrev}] FAILED after {max_retries} attempts.")
                return None


def download_all(
    acs_period: str,
    output_dir: Path,
    include_puerto_rico: bool = True,
    overwrite: bool = False,
    merged_output: Optional[Path] = None,
) -> dict:
    """
    Download tract shapefiles for all states (and optionally PR) for the
    given ACS 5-year period. Returns a dict mapping state abbrev -> shp path
    (or None if failed).
    """
    tiger_year = tiger_year_for_acs(acs_period)
    print(f"\nACS period: {acs_period}")
    print(f"Using TIGER/Line vintage: {tiger_year}")
    print(f"Output directory: {output_dir}")
    print(f"Include Puerto Rico: {include_puerto_rico}")
    print(f"Overwrite existing: {overwrite}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    states_to_download = {
        abbrev: fips for abbrev, fips in STATE_FIPS.items()
        if include_puerto_rico or abbrev != 'PR'
    }

    results = {}
    for abbrev, fips in sorted(states_to_download.items()):
        shp = download_one_state(
            abbrev, fips, tiger_year, output_dir, overwrite=overwrite
        )
        results[abbrev] = shp

    n_ok = sum(1 for v in results.values() if v is not None)
    n_fail = sum(1 for v in results.values() if v is None)
    print(f"\nDownload summary: {n_ok} succeeded, {n_fail} failed")
    if n_fail > 0:
        failed = [k for k, v in results.items() if v is None]
        print(f"Failed states: {', '.join(failed)}")

    if merged_output is not None:
        merge_to_single_file(results, merged_output)

    return results


def merge_to_single_file(
    results: dict,
    output_path: Path,
) -> None:
    """Merge per-state shapefiles into a single national file."""
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError:
        print(
            "geopandas/pandas not installed; cannot merge. "
            "Install with: pip install geopandas",
            file=sys.stderr,
        )
        return

    print(f"\nMerging state shapefiles into {output_path}...")
    parts = []
    for state, shp in results.items():
        if shp is None:
            continue
        try:
            gdf = gpd.read_file(shp)
            parts.append(gdf)
        except Exception as e:
            print(f"  Failed to read {state}: {e}", file=sys.stderr)

    if not parts:
        print("  No state files to merge.", file=sys.stderr)
        return

    merged = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Pick a driver from the file extension
    suffix = output_path.suffix.lower()
    if suffix == '.gpkg':
        merged.to_file(output_path, driver='GPKG')
    elif suffix == '.shp':
        merged.to_file(output_path, driver='ESRI Shapefile')
    elif suffix in ('.parquet', '.pq'):
        merged.to_parquet(output_path)
    elif suffix == '.geojson':
        merged.to_file(output_path, driver='GeoJSON')
    else:
        # Default to GeoPackage
        print(f"  Unknown suffix {suffix}; writing as GeoPackage")
        output_path = output_path.with_suffix('.gpkg')
        merged.to_file(output_path, driver='GPKG')
    print(f"  Wrote {len(merged):,} tracts to {output_path}")


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--acs-period',
        required=True,
        help='ACS 5-year period as YYYY-YYYY (e.g., 2015-2019)',
    )
    p.add_argument(
        '--output-dir',
        required=True,
        type=Path,
        help='Directory to write per-state shapefiles into',
    )
    p.add_argument(
        '--merged-output',
        type=Path,
        default=None,
        help=(
            'Optional path to write a merged national file '
            '(.gpkg, .parquet, .geojson, or .shp)'
        ),
    )
    p.add_argument(
        '--no-puerto-rico',
        action='store_true',
        help='Exclude Puerto Rico from the download',
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Re-download files even if they already exist',
    )
    args = p.parse_args()

    download_all(
        acs_period=args.acs_period,
        output_dir=args.output_dir,
        include_puerto_rico=not args.no_puerto_rico,
        overwrite=args.overwrite,
        merged_output=args.merged_output,
    )


if __name__ == '__main__':
    main()