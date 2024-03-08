#!/bin/bash -l

#SBATCH --time=24:00:00
#SBATCH --partition=small
#SBATCH --ntasks=24
#SBATCH --mem=10g # 30
#SBATCH --array=11-51 #2,6,8,15,22,26,28,31,34,51

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kirsh012@umn.edu

#SBATCH -e oneclassSVM_%A_%a.err
#SBATCH -o oneclassSVM_%A_%a.out

cd ~/masters-thesis/Code

module load python3
source activate tfgpu

config=final_oneclassSVM_model_configurations.txt

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

# Print to a file a message that includes the current $SLURM_ARRAY_TASK_ID, the data name, model, and feature selection method
echo "This is array task ${SLURM_ARRAY_TASK_ID}, the model is saved in the directory ${directory}, the project folder is ${project}, the data name is ${data_name}, the model name is ${model}, and the fact that the model is doing classification is ${binary}." >> oneclassSVM_run_model_job_tracking.txt

python run_oneclassSVM_models.py $directory $project $data_name $model $binary

conda deactivate
