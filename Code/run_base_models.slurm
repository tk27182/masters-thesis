#!/bin/bash -l

#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --ntasks=24
#SBATCH --mem=16g
#SBATCH --array=1-459 #55-57,64-66,73-75,82-84,91-93,100-102,109-111,163-165,172-174,181-183,244-246,253-255,262-265,379-381,388-390,397-399

#SBATCH --mail-type=END,FAIL,REQUEUE,TIME_LIMIT
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e RFLR_models_%A_%a.err
#SBATCH -o RFLR_models_%A_%a.out

cd ~/masters-thesis/Code

module load python3
source activate tfgpu

config=final_base1000_model_configurations.txt

modeltype=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $2}' $config)
subject=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $3}' $config)
sensor=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $4}' $config)
hour=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $5}' $config)
event=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $6}' $config)
smote=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $7}' $config)

model=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $8}' $config)
binary=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $9}' $config)
epochs=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $10}' $config)
callbacks=$(awk -v ArrayTaskID=$SLURM_ARRAY_TASK_ID '$1==ArrayTaskID {print $11}' $config)

directory=${subject}_hypertuning_results
data_name=${modeltype}_${subject}_${sensor}_${hour}_${event}${smote}_Epochs${epochs}_${callbacks}
project=hypermodel_${model}_${data_name}

timeout 47.9h python run_base_models.py $directory $project $data_name $model $binary
if [[ $? == 124 ]]; then
    scontrol requeue ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
fi

conda deactivate

sacct -n -X -j $SLURM_JOB_ID -o state%20,jobid%20 | grep FAILED  >> failed_jobs_base1000.txt
