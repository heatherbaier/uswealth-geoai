#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -G 1
#SBATCH -c 8            # number of cores
#SBATCH -t 48:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_hbaier       # QOS
#SBATCH -J weight_swap
#SBATCH -o slurm.ca_2016_q1_s2_allbands.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.ca_2016_q1_s2_allbands.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our environment
source activate geomain

#Change to the directory of our project
cd /home/hbaier/projects/tlags_v2/

python 3_analysis/compute_calibration_vs_ranking_matrices.py --state oh --variable wealth_index

python 3_analysis/weight_swap_intervention_spread.py --state oh --variable wealth_index \
    --years 2016 2017 2018 2019 \
    --pairs 2018:1:2018:3 2017:2:2017:4 2016:4:2016:2