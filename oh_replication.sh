python3 clean_wealth_index.py --state OH --tracts-dir ../data/tracts2019/ --wealth-csv ../data/final_wealth_indices_2019.csv

python3 compute_landcover.py --state OH --tract-shp ../data/tracts2019/OH/tl_2019_39_tract_wi.shp --year 2020 --out-dir ./test_lc/

python3 3_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/imagery/

python3 4_analysis/assemble_preds_wide.py --state OH --imagery-root /data/hbaier/new_data/tlag/imagery/ --out ./oh_wide.csv

python3 4_analysis/analyze_seasonality_lag.py --state OH --preds-csv ./oh_wide.csv --lc-csv ./1_data_prep/lc_oh.csv
