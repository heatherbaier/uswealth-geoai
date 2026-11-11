# Generate Validation 

# 2016
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 1 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 2 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 3 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 4 --variable wealth_index_sat

# 2017
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 1 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 2 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 3 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 4 --variable wealth_index_sat

# 2018
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 1 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 2 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 3 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 4 --variable wealth_index_sat

# 2019
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 1 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 2 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 3 --variable wealth_index_sat
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 4 --variable wealth_index_sat

# epoch130_valset_oh_2019_q3_s2_allbands_wealth_index_preds

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_wealth_index_sat_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_wealth_index_sat_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_wealth_index_sat_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_wealth_index_sat_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_wealth_index_sat_validate.yml

python3 2_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/oh_imagery/ --variable wealth_index_sat






python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 1 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 2 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 3 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 4 --variable gas_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_gas_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 1 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 2 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 3 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 4 --variable gas_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_gas_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 1 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 2 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 3 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 4 --variable gas_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_gas_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 1 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 2 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 3 --variable gas_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 4 --variable gas_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_gas_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_gas_heat_pct_validate.yml






python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 4 --variable rent_burdened_30plus_pct

# 2017
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 4 --variable rent_burdened_30plus_pct

# 2018
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 4 --variable rent_burdened_30plus_pct

# 2019
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 4 --variable rent_burdened_30plus_pct

# epoch130_valset_oh_2019_q3_s2_allbands_wealth_index_preds

simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_rent_burdened_30plus_pct_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_rent_burdened_30plus_pct_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_rent_burdened_30plus_pct_validate.yml


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_rent_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_rent_burdened_30plus_pct_validate.yml

python3 2_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/oh_imagery/ --variable rent_burdened_30plus_pct




python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_solar_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_solar_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_solar_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_solar_heat_pct_validate.yml



python3 2_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/oh_imagery/ --variable solar_heat_pct



# owner_burdened
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 1 --variable owner_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 2 --variable owner_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 3 --variable owner_burdened_30plus_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2016 --quarter 4 --variable owner_burdened_30plus_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_owner_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_owner_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_owner_burdened_30plus_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_owner_burdened_30plus_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2017 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_solar_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2018 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_solar_heat_pct_validate.yml

python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 1 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 2 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 3 --variable solar_heat_pct
python3 ../pipeline_configs/generate_validate_config.py --state oh --year 2019 --quarter 4 --variable solar_heat_pct


simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_solar_heat_pct_validate.yml
simba /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_solar_heat_pct_validate.yml



python3 2_validation/validate.py --state OH --imagery-root /data/hbaier/new_data/tlag/oh_imagery/ --variable owner_burdened_30plus_pct

