# Create wealth index shapefile
python3 clean_wealth_index.py --state OH --tracts-dir ../data/tracts2019/ --wealth-csv ../data/wealth_indices_2019_USA.csv

# Create the JSON's for the auxiliary wealth indices
python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q2_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q3_2016_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q4_2016_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv

python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q2_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q3_2017_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q4_2017_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv


python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q2_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q3_2018_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q4_2018_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv

 python 1_data_prep/update_ys_labels.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q2_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q3_2019_s2_allbands/ \
/data/hbaier/new_data/tlag/oh_imagery/q4_2019_s2_allbands/ \
 --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv



# Calculate band mean and std statistics for pipeline_configs/state_registry.yml

python 1_data_prep/compute_shared_band_stats.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/oh_imagery/q2_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/oh_imagery/q3_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/oh_imagery/q4_2016_s2_allbands/ \
    --prefixes oh_2016_q1_s2_allbands oh_2016_q2_s2_allbands \
             oh_2016_q3_s2_allbands oh_2016_q4_s2_allbands \
    --out ./oh_2016_shared_band_stats.json


python 1_data_prep/find_spatial_block_deg.py \
    --data-roots /data/hbaier/new_data/tlag/oh_imagery/q1_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/oh_imagery/q2_2016_s2_allbands/ \
    --prefixes oh_2016_q1_s2_allbands oh_2016_q2_s2_allbands \
    --block-degs 0.02 0.05 0.1 0.15 0.2 0.3 0.5


python3 compute_landcover.py --state OH --tract-shp ../data/tracts2019/OH/tl_2019_39_tract_wi.shp --year 2020 --out-dir ./test_lc/

python3 3_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/oh_imagery/