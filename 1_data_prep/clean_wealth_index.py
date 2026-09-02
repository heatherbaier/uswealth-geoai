"""
Join a state's tract shapefile to the ACS-derived wealth index and write:
    <state>_wi2019.csv       GEOID + wealth_index_overall_core
    tl_2019_<fips>_tract_wi.shp   tract shapefile with wealth index attached

Replaces the four near-identical notebooks this project used to have
(clean_wealth_index_{az,ca,ga,pa}.ipynb) with one script driven by --state,
so adding a new state (or re-running an existing one) doesn't mean
copy-pasting a notebook and hand-editing every path in it.

Fixes a real bug found while merging the notebooks: the PA notebook wrote
its output to "tl_2019_42_tract.shp" (no "_wi" suffix), silently
overwriting the plain source shapefile instead of writing the
wealth-index-joined file the AZ/CA/GA notebooks wrote to a separate
"_wi.shp" path. This script always writes the "_wi" suffix.

There is no OH equivalent -- OH's label join predates this repo's history
and OH's geoetl config uses a different, unexplained shapefile/label
convention (see REPLICATION.md). --state oh is rejected on purpose rather
than guessed at.

Usage:
    python clean_wealth_index.py --state az
    python clean_wealth_index.py --state ca --tracts-dir ./data/tracts2019 \
        --wealth-csv ./final_wealth_indices_2019.csv
"""

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

STATE_FIPS = {"AZ": "04", "CA": "06", "GA": "13", "PA": "42"}


def clean_wealth_index(state: str, tracts_dir: Path, wealth_csv: Path):
    state = state.upper()
    if state not in STATE_FIPS:
        raise SystemExit(
            f"--state {state} not supported here (no wealth-index join "
            f"defined). Supported: {sorted(STATE_FIPS)}. If this is OH: "
            f"there is no equivalent for OH in this repo -- see the note "
            f"under this script's docstring / REPLICATION.md."
        )
    fips = STATE_FIPS[state]
    state_dir = tracts_dir / state
    shp_path = state_dir / f"tl_2019_{fips}_tract.shp"
    if not shp_path.exists():
        raise SystemExit(
            f"Tract shapefile not found: {shp_path}\n"
            f"Run download_acs_tracts.py first (or point --tracts-dir at "
            f"wherever it wrote {state}'s shapefile)."
        )

    print(f"[{state}] Loading tracts: {shp_path}")
    gdf = gpd.read_file(shp_path)
    gdf["GEOID"] = gdf["GEOID"].astype(int)

    print(f"[{state}] Loading wealth index: {wealth_csv}")
    df = pd.read_csv(wealth_csv)
    sub = df[df["geoid"].isin(gdf["GEOID"])][["geoid", "wealth_index_overall_core"]]
    print(f"[{state}] Matched {len(sub):,} / {len(gdf):,} tracts to a wealth index row")

    wi_csv_path = state_dir / f"{state.lower()}_wi2019.csv"
    sub.to_csv(wi_csv_path, index=False)
    print(f"[{state}] Wrote {wi_csv_path}")

    sub = sub.rename(columns={"geoid": "GEOID"})
    merged = pd.merge(gdf, sub, on="GEOID")

    wi_shp_path = state_dir / f"tl_2019_{fips}_tract_wi.shp"
    merged.to_file(wi_shp_path)
    print(f"[{state}] Wrote {wi_shp_path} ({len(merged):,} tracts with wealth index)")

    return wi_csv_path, wi_shp_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, choices=sorted(STATE_FIPS) + ["OH", "oh"],
                    help="State abbreviation (az/ca/ga/pa; oh is rejected -- see docstring)")
    p.add_argument("--tracts-dir", type=Path, default=Path("./data/tracts2019"),
                    help="Root dir containing <STATE>/tl_2019_<fips>_tract.shp (default: ./data/tracts2019)")
    p.add_argument("--wealth-csv", type=Path, default=Path("./final_wealth_indices_2019.csv"),
                    help="Path to the ACS-derived wealth index CSV (geoid, wealth_index_overall_core, ...)")
    args = p.parse_args()
    clean_wealth_index(args.state, args.tracts_dir, args.wealth_csv)


if __name__ == "__main__":
    main()
