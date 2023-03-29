#!/bin/bash 

model_type_set=(indv) # general)
subjects=("3-jk") # "31-ns" "43-cm")

hours=("dlh0") # "dlh1" "dlh2") # "dlh3_" "dlh4_" "dlh5_")

sensor_set=(both) # left right)
events=(classification) #regression
cl_set=(simplernn ann)


for mtype in ${model_type_set[@]}; do
    for subj in ${subjects[@]}; do
        for hour in ${hours[@]}; do
            for sensor in ${sensor_set[@]}; do
                for event in ${events[@]}; do
               
                    name=${mtype}_${subj}_${sensor}_${hour}_${event}

                    directory=${subj}_hypertuning_results
                    
                    for cl in ${cl_set[@]}; do
           		
               		jobname=${name}_${cl}
                        echo $jobname                        
                        project=hypermodel_${cl}    
			sbatch -J $jobname --export=directory=$directory,project=$project,data_name=$name,model=$cl,binary=True run_models.slurm 

       		    done
		done
            done
	done
    done
done
