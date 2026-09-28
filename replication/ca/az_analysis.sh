# Cross-validation 2016
python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2016 --model-quarter 1     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2016 --model-quarter 2     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2016 --model-quarter 3     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2016 --model-quarter 4     --imagery-quarters 2016Q1 2016Q2 2016Q3 2016Q4

python 4_analysis/build_cross_quarter_r2_matrix.py     --state az --variable wealth_index     --quarters 2016Q1 2016Q2 2016Q3 2016Q4
python 4_analysis/analyze_cross_quarter_r2_structure.py     --r2-matrix ./out_cross_quarter/az_wealth_index/r2_matrix_2016.csv


# Cross-validation 2017
python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2017 --model-quarter 1     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2017 --model-quarter 2     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2017 --model-quarter 3     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

python pipeline_configs/generate_cross_validate_config.py     --state az --variable wealth_index     --model-year 2017 --model-quarter 4     --imagery-quarters 2017Q1 2017Q2 2017Q3 2017Q4

python 4_analysis/build_cross_quarter_r2_matrix.py     --state az --variable wealth_index     --quarters 2017Q1 2017Q2 2017Q3 2017Q4
python 4_analysis/analyze_cross_quarter_r2_structure.py     --r2-matrix ./out_cross_quarter/az_wealth_index/r2_matrix_2017.csv