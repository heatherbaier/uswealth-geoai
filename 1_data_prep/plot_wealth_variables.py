"""
Plot a state's census tracts, one subplot per wealth_index_* target
variable, colored by that variable's value -- a visual sanity check on the
new 6-variable ACS wealth CSV (see update_ys_labels.py's docstring) before
training a model on any of them: does each variable's spatial pattern look
plausible, and how much of the state is actually covered (tracts with no
matching CSV row are drawn gray, not silently dropped from the map).

Joins <tracts-dir>/<STATE>/tl_2019_<fips>_tract.shp -- the RAW shapefile,
not clean_wealth_index.py's tl_2019_<fips>_tract_wi.shp -- to --wealth-csv
directly in pandas/geopandas, rather than via a shapefile write-and-rejoin.
That matters specifically here: clean_wealth_index.py's output shapefile
carries its wealth column through a DBF file, whose field names silently
truncate to 10 characters (that's why "wealth_index_overall_core" comes
back out as "wealth_ind" in geoetl's label_column) -- and two of the six
new variables collide on those first 10 characters
("wealth_index_housing_core" and "wealth_index_financial" both start
"wealth_ind"), so writing all six through a shapefile would silently merge
two variables into one column. Reading the CSV directly in pandas has no
such limit, so that's what this script does.

GEOID matching reuses update_ys_labels.py's normalize_geoid() (digits
only, zero-padded to 11) instead of clean_wealth_index.py's
gdf["GEOID"].astype(int) -- the same dropped-leading-zero risk on
single-digit state FIPS (AZ=04, CA=06, GA=13) that update_ys_labels.py's
docstring already flags for the CSV side; the shapefile's GEOID needs the
same normalization, not a second, possibly-inconsistent implementation of
it.

Usage:
    python plot_wealth_variables.py --state az --wealth-csv ./partner_acs_wealth.csv
    python plot_wealth_variables.py --state ca --tracts-dir ./data/tracts2019 \
        --wealth-csv ./partner_acs_wealth.csv --out ./ca_wealth_variables.png
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from update_ys_labels import normalize_geoid  # noqa: E402

# Matches clean_wealth_index.py -- OH is deliberately excluded there (no
# reproducible wealth-index join for OH in this repo) and for the same
# reason is excluded here.
STATE_FIPS = {"AZ": "04", "CA": "06", "GA": "13", "PA": "42"}

# Must match update_ys_labels.py's DEFAULT_TARGETS -- the 6 wealth_index_*
# columns models are actually trained on.
WEALTH_VARIABLES = [
    "wealth_index",
    "wealth_index_sat",
    "wealth_index_housing_core",
    "wealth_index_transport",
    "wealth_index_financial",
    "wealth_index_utilities",
]


def load_state_tracts(state: str, tracts_dir: Path) -> gpd.GeoDataFrame:
    state = state.upper()
    if state not in STATE_FIPS:
        raise SystemExit(
            f"--state {state} not supported (no tract shapefile convention "
            f"defined). Supported: {sorted(STATE_FIPS)}."
        )
    fips = STATE_FIPS[state]
    shp_path = tracts_dir / state / f"tl_2019_{fips}_tract.shp"
    if not shp_path.exists():
        raise SystemExit(
            f"Tract shapefile not found: {shp_path}\n"
            f"Run download_acs_tracts.py first (or point --tracts-dir at "
            f"wherever it wrote {state}'s shapefile)."
        )
    gdf = gpd.read_file(shp_path)
    gdf["_geoid_norm"] = gdf["GEOID"].apply(normalize_geoid)
    return gdf


def load_wealth_csv(path: Path, targets: list) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "geoid" not in df.columns:
        raise SystemExit(f"--wealth-csv has no 'geoid' column. Available: {list(df.columns)}")
    missing = [t for t in targets if t not in df.columns]
    if missing:
        raise SystemExit(f"--wealth-csv is missing columns: {missing}\n"
                          f"Available: {list(df.columns)}")
    df["_geoid_norm"] = df["geoid"].apply(normalize_geoid)
    for t in targets:
        df[t] = pd.to_numeric(df[t], errors="coerce")
    return df


def plot_state_wealth_variables(state: str, tracts_dir: Path, wealth_csv: Path,
                                 targets: list, out_path: Path, ncols: int = 3):
    gdf = load_state_tracts(state, tracts_dir)
    df = load_wealth_csv(wealth_csv, targets)

    merged = gdf.merge(df[["_geoid_norm"] + targets], on="_geoid_norm", how="left")
    n_matched = int(merged[targets[0]].notna().sum())
    print(f"[{state.upper()}] {len(merged):,} tracts, {n_matched:,} matched to --wealth-csv "
          f"({len(merged) - n_matched:,} will be drawn gray/no data)")

    n = len(targets)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = [axes] if n == 1 else axes.flatten()

    for ax, var in zip(axes, targets):
        merged.plot(
            column=var,
            ax=ax,
            cmap="viridis",
            linewidth=0.1,
            edgecolor="white",
            legend=True,
            missing_kwds={"color": "lightgray", "label": "no data"},
        )
        ax.set_title(var, fontsize=11)
        ax.set_axis_off()

    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle(f"{state.upper()} — wealth index variables by tract", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True, help="State abbreviation (az/ca/ga/pa)")
    p.add_argument("--tracts-dir", type=Path, default=Path("./data/tracts2019"),
                    help="Root dir containing <STATE>/tl_2019_<fips>_tract.shp (default: ./data/tracts2019)")
    p.add_argument("--wealth-csv", type=Path, required=True,
                    help="Partner's ACS-derived wealth CSV (geoid + wealth_index_* columns)")
    p.add_argument("--targets", nargs="+", default=WEALTH_VARIABLES,
                    help=f"Column(s) to plot, one subplot each (default: all 6: {WEALTH_VARIABLES})")
    p.add_argument("--out", type=Path, default=None,
                    help="Default: ./wealth_variables_<state>.png")
    p.add_argument("--ncols", type=int, default=3, help="Subplot grid columns (default: 3)")
    args = p.parse_args()

    out_path = args.out or Path(f"./wealth_variables_{args.state.lower()}.png")
    plot_state_wealth_variables(args.state, args.tracts_dir, args.wealth_csv,
                                 args.targets, out_path, args.ncols)


if __name__ == "__main__":
    main()
