# Cross-validation 2016
python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2016 --model-quarter 1     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2016_wealth_index_sat_v1_on_2016q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2016_wealth_index_sat_v1_on_2016q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2016_wealth_index_sat_v1_on_2016q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2016_wealth_index_sat_v1_on_2016q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2016 --model-quarter 2     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2016_wealth_index_sat_v1_on_2016q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2016_wealth_index_sat_v1_on_2016q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2016_wealth_index_sat_v1_on_2016q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2016_wealth_index_sat_v1_on_2016q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2016 --model-quarter 3     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2016_wealth_index_sat_v1_on_2016q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2016_wealth_index_sat_v1_on_2016q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2016_wealth_index_sat_v1_on_2016q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2016_wealth_index_sat_v1_on_2016q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2016 --model-quarter 4     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2016_wealth_index_sat_v1_on_2016q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2016_wealth_index_sat_v1_on_2016q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2016_wealth_index_sat_v1_on_2016q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2016_wealth_index_sat_v1_on_2016q4_validate.yml

python 3_analysis/build_cross_quarter_r2_matrix.py     --state oh --variable wealth_index_sat     --quarters 2016Q1 2016Q2 2016Q3 2016Q4

python 3_analysis/analyze_cross_quarter_r2_structure.py     --r2-matrix ./out_cross_quarter/oh_wealth_index_sat/r2_matrix_2016.csv


# Cross-validation 2017
# Cross-validation 2017
python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2017 --model-quarter 1     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2017_wealth_index_sat_v1_on_2017q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2017_wealth_index_sat_v1_on_2017q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2017_wealth_index_sat_v1_on_2017q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2017_wealth_index_sat_v1_on_2017q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2017 --model-quarter 2     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2017_wealth_index_sat_v1_on_2017q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2017_wealth_index_sat_v1_on_2017q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2017_wealth_index_sat_v1_on_2017q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2017_wealth_index_sat_v1_on_2017q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2017 --model-quarter 3     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2017_wealth_index_sat_v1_on_2017q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2017_wealth_index_sat_v1_on_2017q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2017_wealth_index_sat_v1_on_2017q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2017_wealth_index_sat_v1_on_2017q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2017 --model-quarter 4     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2017_wealth_index_sat_v1_on_2017q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2017_wealth_index_sat_v1_on_2017q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2017_wealth_index_sat_v1_on_2017q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2017_wealth_index_sat_v1_on_2017q4_validate.yml

python 3_analysis/build_cross_quarter_r2_matrix.py     --state oh --variable wealth_index_sat     --quarters 2016Q1 2016Q2 2016Q3 2016Q4 2017Q1 2017Q2 2017Q3 2017Q4


# Cross-validation 2018
python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2018 --model-quarter 1     --imagery-quarters 2018Q1 2018Q2 2018Q3 2018Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2018_wealth_index_sat_v1_on_2018q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2018_wealth_index_sat_v1_on_2018q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2018_wealth_index_sat_v1_on_2018q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2018_wealth_index_sat_v1_on_2018q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2018 --model-quarter 2     --imagery-quarters 2018Q1 2018Q2 2018Q3 2018Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2018_wealth_index_sat_v1_on_2018q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2018_wealth_index_sat_v1_on_2018q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2018_wealth_index_sat_v1_on_2018q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2018_wealth_index_sat_v1_on_2018q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2018 --model-quarter 3     --imagery-quarters 2018Q1 2018Q2 2018Q3 2018Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2018_wealth_index_sat_v1_on_2018q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2018_wealth_index_sat_v1_on_2018q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2018_wealth_index_sat_v1_on_2018q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2018_wealth_index_sat_v1_on_2018q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2018 --model-quarter 4     --imagery-quarters 2018Q1 2018Q2 2018Q3 2018Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2018_wealth_index_sat_v1_on_2018q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2018_wealth_index_sat_v1_on_2018q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2018_wealth_index_sat_v1_on_2018q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2018_wealth_index_sat_v1_on_2018q4_validate.yml

python 3_analysis/build_cross_quarter_r2_matrix.py     --state oh --variable wealth_index_sat     --quarters 2016Q1 2016Q2 2016Q3 2016Q4 2017Q1 2017Q2 2017Q3 2017Q4 2018Q1 2018Q2 2018Q3 2018Q4

# Cross-validation 2019
python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2019 --model-quarter 1     --imagery-quarters 2019Q1 2019Q2 2019Q3 2019Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2019_wealth_index_sat_v1_on_2019q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2019_wealth_index_sat_v1_on_2019q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2019_wealth_index_sat_v1_on_2019q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q1_2019_wealth_index_sat_v1_on_2019q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2019 --model-quarter 2     --imagery-quarters 2019Q1 2019Q2 2019Q3 2019Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2019_wealth_index_sat_v1_on_2019q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2019_wealth_index_sat_v1_on_2019q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2019_wealth_index_sat_v1_on_2019q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q2_2019_wealth_index_sat_v1_on_2019q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2019 --model-quarter 3     --imagery-quarters 2019Q1 2019Q2 2019Q3 2019Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2019_wealth_index_sat_v1_on_2019q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2019_wealth_index_sat_v1_on_2019q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2019_wealth_index_sat_v1_on_2019q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q3_2019_wealth_index_sat_v1_on_2019q4_validate.yml


python pipeline_configs/generate_cross_validate_config.py     --state oh --variable wealth_index_sat     --model-year 2019 --model-quarter 4     --imagery-quarters 2019Q1 2019Q2 2019Q3 2019Q4

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2019_wealth_index_sat_v1_on_2019q1_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2019_wealth_index_sat_v1_on_2019q2_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2019_wealth_index_sat_v1_on_2019q3_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/cross/oh_q4_2019_wealth_index_sat_v1_on_2019q4_validate.yml

python 3_analysis/build_cross_quarter_r2_matrix.py     --state oh --variable wealth_index_sat     --quarters 2016Q1 2016Q2 2016Q3 2016Q4 2017Q1 2017Q2 2017Q3 2017Q4 2018Q1 2018Q2 2018Q3 2018Q4 2019Q1 2019Q2 2019Q3 2019Q4
