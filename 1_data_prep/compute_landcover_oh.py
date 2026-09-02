import matplotlib.pyplot as plt
from tqdm.auto import tqdm  # provides a progressbar
from pathlib import Path
import rioxarray as rxr
import geopandas as gpd
import rasterio as rio
import pandas as pd
import xarray as xr
import numpy as np
import requests
import json
import os

s3_url_prefix = "https://esa-worldcover.s3.eu-central-1.amazonaws.com"

# load natural earth low res shapefile
ne = gpd.read_file("./data/tracts2019/OH/tl_2019_39_tract_proj.shp")

# get AOI geometry (dissolved to a single geometry)
geom = ne.dissolve().geometry.iloc[0]

# load worldcover grid
url = f'{s3_url_prefix}/esa_worldcover_grid.geojson'
grid = gpd.read_file(url)

# get grid tiles intersecting AOI (fixed: intersects against a single geometry, not a GeoDataFrame)
tiles = grid[grid.intersects(geom)]

year = 2021  # setting this to 2020 will download the v100 product instead
# select version tag, based on the year
version = {2020: 'v100',
           2021: 'v200'}[year]

output_folder = "./wc/"  # use current directory or set a different one to store downloaded files

for tile in tqdm(tiles.ll_tile):
    url = f"{s3_url_prefix}/{version}/{year}/map/ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    out_fn = Path(output_folder) / Path(url).name

    # stream to disk instead of buffering the whole response in memory
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_fn, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

# open lazily (dask-backed) instead of eagerly loading the full tile into RAM
b8 = rxr.open_rasterio(
    "./wc/ESA_WorldCover_10m_2021_v200_N39W084_Map.tif",
    masked=True,
    chunks=True,
)

val_dict = {}
for col, row in ne.iterrows():
    
    try:
        
        print(row.GEOID)
        # from_disk=True does a windowed read instead of materializing the full array
        clipped = b8.rio.clip([row.geometry], from_disk=True)
        unique, counts = np.unique(clipped, return_counts=True, equal_nan=True)
        total_non_nan = counts[~np.isnan(unique)].sum()
        val_dict[str(int(row.GEOID))] = dict(zip(unique.tolist(), counts.tolist()))
    
        with open("./test.json", "w") as f:
            json.dump(val_dict, f)

    except:

        pass



