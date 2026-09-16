python3 clean_wealth_index.py --state AZ --tracts-dir ../data/tracts2019/ --wealth-csv ../data/final_wealth_indices_2019.csv

python3 compute_landcover.py --state AZ --tract-shp ../data/tracts2019/AZ/tl_2019_04_tract_wi.shp --year 2020 --out-dir ./test_lc/

python3 3_validation/validate.py --state AZ --imagery-root /data/hbaier/new_data/tlag/az_imagery/
