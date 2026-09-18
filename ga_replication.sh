# GA, from scratch. Run from the repo root.

# 1. Get GA's real tract shapefile only -- not all 50 states + DC + PR again.
python3 1_data_prep/download_acs_tracts.py --acs-period 2015-2019 \
    --output-dir ./data/tracts2019 --states GA

# 2. Join it to the wealth-index CSV -> ./data/tracts2019/GA/tl_2019_42_tract_wi.shp
#    (this is what geoetl's state_registry.yml pa.shp_path assumes exists --
#    see geoetl/scripts/state_registry.yml's note on this).
cd 1_data_prep
python3 clean_wealth_index.py --state GA --tracts-dir ../data/tracts2019/ \
    --wealth-csv ../data/wealth_indices_2019_USA.csv

# 3. Visual sanity check before spending download/train time on it.
python3 plot_wealth_variables.py --state ga --tracts-dir ../data/tracts2019/ \
    --wealth-csv ../data/wealth_indices_2019_USA.csv --out ../ga_wealth_variables.png
cd ..

# 4. Land cover (can run any time after step 1; doesn't depend on the wealth join).
python3 1_data_prep/compute_landcover.py --state ga \
    --tract-shp ./data/tracts2019/GA/tl_2019_13_tract_wi.shp

# --- from here on, this repeats per quarter/year/variable, same as AZ/GA.
# Everything below runs from THIS repo's root -- no cd into geoetl/sail,
# they only need to be pip-installed (see REPLICATION.md's Setup section
# and ./requirements.txt). ---

# 5. Generate a geoetl download config + SLURM job, then sbatch it.
python3 pipeline_configs/generate_download_config.py --state ga --year 2019 --quarter 1 --launch

# 6. Once imagery is downloaded for a few quarters: run sail's
#    find_sgatial_block_deg.py and compute_shared_band_stats.py
#    (in ~/packages/sail/scripts/ -- these two are analysis/one-off tools,
#    not part of the per-run config generation, so they're not duplicated
#    here) for GA, and fill in pipeline_configs/state_registry.yml's pa
#    entry (spatial_block_deg/band_mean/band_std -- still null right now,
#    can't be computed without real downloaded imagery).

# 7. Generate a sail train config + SLURM job, then sbatch it.
python3 pipeline_configs/generate_train_config.py --state ga --year 2016 --quarter 1 --variable wealth_index_sat --launch

# 8. Once trained: generate + sbatch the matching validate config.
python3 pipeline_configs/generate_validate_config.py --state ga --year 2016 --quarter 1 --variable wealth_index 
