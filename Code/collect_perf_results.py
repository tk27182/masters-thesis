import perfmetrics as pm
import pandas as pd

model_type_set = ['indv']
subjects = ['3-jk', '31-ns', '43-cm']

hours=["dlh0", "dlh1", "dlh2"] # "dlh3_" "dlh4_" "dlh5_")

sensor_set=['left', 'right', 'both']
events=['classification'] #regression
smote_set=["_smote", "_gauss"] # "")

cl_set=['lstm', 'ann', 'simplernn']

data_name_set = [f"{mdl}_{subj}_{sensor}_{hour}_{event}{smote}"
                for mdl in model_type_set
                for subj in subjects
                for sensor in sensor_set
                for hour in hours
                for event in events
                for smote in smote_set
                ]

cr = pm.CollectResults()

for data_name in data_name_set:
    for cl in cl_set:
        cr.compute(data_name=data_name, model_name=cl)

df = cr.make_dataframe()

# Save results to dataframe
df.to_csv("../Results/preliminary_results.csv")