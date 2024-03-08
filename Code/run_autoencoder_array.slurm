#!/bin/bash -l

#SBATCH --open-mode=append
#SBATCH --time=24:00:00
#SBATCH -p k40 #a100-4
#SBATCH --gres=gpu:k40:2#--gres=gpu:a100:1
###SBATCH --requeue

#SBATCH --ntasks=24
#SBATCH --mem=30g #60g
#SBATCH --array=11-108 #216

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e model_job_%A_%a.err
#SBATCH -o model_job_%A_%a.out

cd ~/masters-thesis/Code
module load python3
source activate tfgpu

config=final_autoencoder_model_configurations.txt

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

python run_models.py $directory $project $data_name $model $binary
# timeout 23.5h python run_models.py $directory $project $data_name $model $binary
# if [[ $? == 124 ]]; then
#     scontrol requeue ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}
# fi

conda deactivate

sacct -n -X -j $SLURM_JOB_ID -o state%20,jobid%20 | grep FAILED  >> failed_autoencoder_jobs.txt