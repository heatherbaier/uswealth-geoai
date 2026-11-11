"""
One figure, one subplot per state, each a tract-level choropleth of that
state's wealth index -- the geographic counterpart to plot_wealth_
variables.py (one state, one subplot per VARIABLE); this is one VARIABLE,
one subplot per STATE, for comparing wealth's spatial pattern across
states side by side.

Reads each state's --shp-path/--label-column straight from
pipeline_configs/state_registry.yml (the same shp_path/label_column
fields generate_download_config.py already uses) rather than a raw
wealth CSV: that shapefile is clean_wealth_index.py's OUTPUT (tl_2019_
<fips>_tract_wi.shp), already joined to the wealth index by GEOID -- no
second join needed here, just read label_column straight off it. Every
state's label_column is currently "wealth_ind" (the ACS wealth index's
DBF-truncated field name -- see plot_wealth_variables.py's docstring for
why), but this reads it from the registry per state rather than
hardcoding that string, in case that ever differs.

Every subplot shares ONE color scale (computed across every loaded
state's values) and ONE colorbar, not one per subplot -- unlike plot_
wealth_variables.py, where each subplot is a DIFFERENT variable (not
comparable on a shared scale), every subplot here is the SAME variable
for a different state, so a shared scale is what makes "is state A
wealthier than state B" readable at a glance instead of each subplot
secretly being auto-scaled to its own state's min/max.

A state missing its shapefile on disk (or missing a usable shp_path/
label_column in the registry -- e.g. OH's join predates this repo and
uses a different, unverified convention, see clean_wealth_index.py's
docstring) is skipped with a clear warning instead of crashing the whole
combined figure; the other states still get plotted.

Usage:
    python 1_data_prep/plot_wealth_index_map.py
    python 1_data_prep/plot_wealth_index_map.py --states az ga pa \
        --out ./wealth_index_by_state.png --ncols 2
"""

import argparse
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline_configs"))
from generate_train_config import load_registry  # noqa: E402


def load_state_wealth(state: str, registry: dict):
    """Returns a GeoDataFrame with a numeric "_wealth" column, or None
    (with a printed reason) if this state can't be mapped right now."""
    state = state.lower()
    entry = registry.get(state, {})
    shp_path = entry.get("shp_path")
    label_column = entry.get("label_column")
    if not shp_path or not label_column:
        print(f"[{state.upper()}] skipping -- no shp_path/label_column in "
              f"state_registry.yml for this state yet")
        return None
    if not Path(shp_path).exists():
        print(f"[{state.upper()}] skipping -- shp_path not found on disk: {shp_path}")
        return None

    gdf = gpd.read_file(shp_path)
    if label_column not in gdf.columns:
        print(f"[{state.upper()}] skipping -- {label_column!r} not in {shp_path} "
              f"(columns: {list(gdf.columns)})")
        return None

    gdf["_wealth"] = pd.to_numeric(gdf[label_column], errors="coerce")
    n_matched = int(gdf["_wealth"].notna().sum())
    print(f"[{state.upper()}] {len(gdf):,} tracts, {n_matched:,} with a wealth index value "
          f"({len(gdf) - n_matched:,} will be drawn gray/no data)")
    return gdf


def plot_wealth_index_by_state(states: list, registry: dict, out_path: Path,
                                ncols: int, cmap: str):
    loaded = {}
    for state in states:
        gdf = load_state_wealth(state, registry)
        if gdf is not None:
            loaded[state] = gdf

    if not loaded:
        raise SystemExit("No state could be loaded -- nothing to plot.")

    # One shared color scale across every state, so relative color
    # intensity is actually comparable state-to-state (see docstring).
    all_values = np.concatenate([gdf["_wealth"].dropna().to_numpy() for gdf in loaded.values()])
    vmin, vmax = float(all_values.min()), float(all_values.max())

    n = len(loaded)
    ncols = min(ncols, n)
    nrows = -(-n // ncols)  # ceil
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows))
    axes = [axes] if n == 1 else np.atleast_1d(axes).flatten()

    for ax, (state, gdf) in zip(axes, loaded.items()):
        gdf.plot(
            column="_wealth",
            ax=ax,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidth=0.1,
            edgecolor="white",
            missing_kwds={"color": "lightgray", "label": "no data"},
        )
        ax.set_title(state.upper(), fontsize=13)
        ax.set_axis_off()

    for ax in axes[len(loaded):]:
        ax.set_visible(False)

    sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=list(axes[:len(loaded)]), label="Wealth index", shrink=0.8)

    fig.suptitle("Wealth index by state (tract-level)", fontsize=15, y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    registry = load_registry()
    p.add_argument("--states", nargs="+", default=sorted(registry),
                    help=f"Default: every state in state_registry.yml ({sorted(registry)})")
    p.add_argument("--ncols", type=int, default=3, help="Subplot grid columns (default: 3)")
    p.add_argument("--cmap", default="viridis",
                    help="Matplotlib colormap (default: viridis, matching plot_wealth_variables.py)")
    p.add_argument("--out", type=Path, default=Path("./wealth_index_by_state.png"))
    args = p.parse_args()

    plot_wealth_index_by_state([s.lower() for s in args.states], registry,
                                args.out, args.ncols, args.cmap)


if __name__ == "__main__":
    main()
