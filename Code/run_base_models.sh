#!/bin/bash

model_type_set=(indv) # general)
subjects=("3-jk") # "1-sf" "10-rc" "12-mb" "17-sb" "19-me" "2-bd" "22-ap" "26-tc" "3-jk" "31-ns" "32-rf" "36-af" "38-cs" "39-dg" "4-rs" "41-pk" "43-cm" "7-sb"

hours=("dlh2") #"dlh3" "dlh4" "dlh5" "dlh0" "dlh1" "dlh2") #

sensor_set=(both)
events=(classification) #regression
cl_set=(lr)
smote_set=("_None") # "_gauss" "")

for mtype in ${model_type_set[@]}; do
    for subj in ${subjects[@]}; do
        for hour in ${hours[@]}; do
            for sensor in ${sensor_set[@]}; do
                for event in ${events[@]}; do
                    for smote in "${smote_set[@]}"; do

                        name=${mtype}_${subj}_${sensor}_${hour}_${event}${smote}_Epochs100_None

                        directory=test_logistic_${subj}_hypertuning_results

                        for cl in ${cl_set[@]}; do

                            project=test_logistic_hypermodel_${cl}_${name}
                            echo $name $cl
                            python run_base_models.py $directory $project $name $cl True

                        done
                    done
                done
            done
        done
    done
done