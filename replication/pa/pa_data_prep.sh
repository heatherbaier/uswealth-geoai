
python 1_data_prep/update_ys_labels.py \
    --data-roots  /data/hbaier/new_data/tlag/pa_imagery/q1_2017_s2_allbands/ \
		 /data/hbaier/new_data/tlag/pa_imagery/q2_2017_s2_allbands/ \
                 /data/hbaier/new_data/tlag/pa_imagery/q3_2017_s2_allbands/ \
                 /data/hbaier/new_data/tlag/pa_imagery/q4_2017_s2_allbands/ \
    --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv

python 1_data_prep/update_ys_labels.py \
    --data-roots  /data/hbaier/new_data/tlag/pa_imagery/q1_2016_s2_allbands/ \
		 /data/hbaier/new_data/tlag/pa_imagery/q2_2016_s2_allbands/ \
                 /data/hbaier/new_data/tlag/pa_imagery/q3_2016_s2_allbands/ \
                 /data/hbaier/new_data/tlag/pa_imagery/q4_2016_s2_allbands/ \
    --wealth-csv /home/hbaier/projects/tlags_v2/data/wealth_indices_2019_USA.csv





python 1_data_prep/compute_shared_band_stats.py \
  --data-roots /data/hbaier/new_data/tlag/pa_imagery/q1_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/pa_imagery/q2_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/pa_imagery/q3_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/pa_imagery/q4_2016_s2_allbands/ \
  --prefixes pa_2016_q1_s2_allbands pa_2016_q2_s2_allbands \
             pa_2016_q3_s2_allbands pa_2016_q4_s2_allbands \
  --out ./pa_2016_shared_band_stats.json


python 1_data_prep/find_spatial_block_deg.py \
  --data-roots /data/hbaier/new_data/tlag/pa_imagery/q1_2016_s2_allbands/ \
               /data/hbaier/new_data/tlag/pa_imagery/q2_2016_s2_allbands/ \
  --prefixes pa_2016_q1_s2_allbands pa_2016_q2_s2_allbands \
  --block-degs 0.02 0.05 0.1 0.15 0.2 0.3 0.5