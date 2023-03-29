#!/bin/bash -l

#SBATCH --time=06:00:00
#SBATCH --partition=k40
#SBATCH --gres=gpu:k40:2
#SBATCH --ntasks=24
#SBATCH --mem=6g

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e %x.err
#SBATCH -o %x.out

cd ~/masters-thesis/Code

module load python3
source activate tfgpu

python run_models.py $directory $project $data_name $model $binary

conda deactivate
