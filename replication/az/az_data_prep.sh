python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q3_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q4_2016_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv

python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q2_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q3_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q4_2017_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv


python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q2_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q3_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q4_2018_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv


python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q2_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q3_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/az_imagery/q4_2019_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv




python3 clean_wealth_index.py --state AZ --tracts-dir ../data/tracts2019/ --wealth-csv ../data/final_wealth_indices_2019.csv

python3 compute_landcover.py --state AZ --tract-shp ../data/tracts2019/AZ/tl_2019_04_tract_wi.shp --year 2020 --out-dir ./test_lc/

python3 3_validation/validate.py --state AZ --imagery-root /data/hbaier/new_data/tlag/az_imagery/