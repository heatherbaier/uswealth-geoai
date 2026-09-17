# PA, from scratch. Run from the repo root.

# 1. Get PA's real tract shapefile only -- not all 50 states + DC + PR again.
python3 1_data_prep/download_acs_tracts.py --acs-period 2015-2019 \
    --output-dir ./data/tracts2019 --states PA

# 2. Join it to the wealth-index CSV -> ./data/tracts2019/PA/tl_2019_42_tract_wi.shp
#    (this is what geoetl's state_registry.yml pa.shp_path assumes exists --
#    see geoetl/scripts/state_registry.yml's note on this).
cd 1_data_prep
python3 clean_wealth_index.py --state PA --tracts-dir ../data/tracts2019/ \
    --wealth-csv ../data/final_wealth_indices_2019.csv

# 3. Visual sanity check before spending download/train time on it.
python3 plot_wealth_variables.py --state pa --tracts-dir ../data/tracts2019/ \
    --wealth-csv ../data/final_wealth_indices_2019.csv --out ../pa_wealth_variables.png
cd ..

# 4. Land cover (can run any time after step 1; doesn't depend on the wealth join).
python3 1_data_prep/compute_landcover.py --state pa \
    --tract-shp ./data/tracts2019/PA/tl_2019_42_tract_wi.shp

# --- from here on, this repeats per quarter/year/variable, same as AZ/GA ---

# 5. In geoetl: generate a download config + SLURM job, then sbatch it.
#    (cd /home/hbaier/packages/geoetl first)
#    python scripts/generate_download_config.py --state pa --year 2016 --quarter 1 --launch

# 6. Once imagery is downloaded for a few quarters: in sail, run
#    find_spatial_block_deg.py and compute_shared_band_stats.py for PA and
#    fill in its scripts/state_registry.yml entry (still null right now --
#    can't be computed without real downloaded imagery).

# 7. In sail: generate a train config + SLURM job, then sbatch it.
#    (cd /home/hbaier/packages/sail first)
#    python scripts/generate_train_config.py --state pa --year 2016 --quarter 1 \
#        --variable wealth_index_sat --launch

# 8. Once trained: generate + sbatch the matching validate config.
#    python scripts/generate_validate_config.py --state pa --year 2016 --quarter 1 \
#        --variable wealth_index_sat --launch
