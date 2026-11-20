#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -G 1
#SBATCH -c 8            # number of cores
#SBATCH -t 48:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_hbaier       # QOS
#SBATCH -J pa_weight_swap
#SBATCH -o slurm.pa_weight_swap.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.pa_weight_swap.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our environment
source activate geomain

#Change to the directory of our project
cd /home/hbaier/projects/tlags_v2/

python 4_analysis/weight_swap_intervention_batch.py --state pa --variable wealth_index \
    --cross-quarter-r2-csv out_cross_quarter/pa_wealth_index/r2_matrix.csv \
    --all-cells
