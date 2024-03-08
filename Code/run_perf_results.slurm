#!/bin/bash -l

#SBATCH --time=4:00:00
#SBATCH --partition=small
#SBATCH --ntasks=2
#SBATCH --mem=10g

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e perf_results.err
#SBATCH -o perf_results.out

cd ~/masters-thesis/Code

module load python3
source activate tfgpu

python3 collect_perf_results.py

conda deactivate
