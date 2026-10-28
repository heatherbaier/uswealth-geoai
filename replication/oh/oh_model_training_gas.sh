# Generate Training 

# 2016
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2016 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2016 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2016 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2016 --quarter 4 --variable rent_burdened_30plus_pct

# 2017
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2017 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2017 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2017 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2017 --quarter 4 --variable rent_burdened_30plus_pct

# 2018
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2018 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2018 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2018 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2018 --quarter 4 --variable rent_burdened_30plus_pct

# 2019
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2019 --quarter 1 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2019 --quarter 2 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2019 --quarter 3 --variable rent_burdened_30plus_pct
python3 ../pipeline_configs/generate_train_config.py --state oh --year 2019 --quarter 4 --variable rent_burdened_30plus_pct



sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q1_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q2_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q3_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2016_q4_rent_burdened_30plus_pct_train.sh

sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q1_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q2_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q3_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2017_q4_rent_burdened_30plus_pct_train.sh

sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q1_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q2_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q3_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2018_q4_rent_burdened_30plus_pct_train.sh

sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q1_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q2_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q3_rent_burdened_30plus_pct_train.sh
sbatch /home/hbaier/projects/tlags_v2/configs/sail/tlags/oh/oh_2019_q4_rent_burdened_30plus_pct_train.sh