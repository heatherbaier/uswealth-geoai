#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -G 1
#SBATCH -c 8            # number of cores
#SBATCH -t 48:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q grp_hbaier       # QOS
#SBATCH -J az_crossq
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

bash /home/hbaier/projects/tlags_v2/run_cross_quarter_full.sh
