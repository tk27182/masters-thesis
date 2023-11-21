import perfmetrics as pm

model_type_set = ['indv', 'general']
subjects = ['3-jk', "1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd",
            "22-ap", "26-tc", "31-ns", "32-rf", "36-af", "38-cs", "39-dg",
            "4-rs", "41-pk", "43-cm", "7-sb"]

hours=["dlh0", "dlh1", 'dlh2'] #"dlh0", "dlh1", "dlh2", "dlh3_", "dlh4_", "dlh5_"]

sensor_set=['left']
events=['classification'] #regression
cl_set=['lstm', 'ann', 'simplernn']
base_cl_set=['randomforest', 'lr']
smote_set=["_None", "_downsample", "_smote"] #"_smote", "_gauss", "", "_downsample", "_original"]
epochs = ['Epochs100', 'Epochs200', 'Epochs500', 'Epochs1000']
callbacks = ['None']

# Neural Network folders
data_name_set = [f"{mdl}_{subj}_{sensor}_{hour}_{event}{smote}_{epoch}_{cb}"
                for mdl in model_type_set
                for subj in subjects
                for sensor in sensor_set
                for hour in hours
                for event in events
                for smote in smote_set
                for epoch in epochs
                for cb in callbacks
                ]

# Base Model folders
base_data_name_set = []
for mdl in model_type_set:
    for subj in subjects:
        for sensor in sensor_set:
            for hour in hours:
                for event in events:
                    for smote in smote_set:
                        # if smote == '_None':
                        #     smote = ''

                        bdata_name = f"{mdl}_{subj}_{sensor}_{hour}_{event}{smote}"
                        base_data_name_set.append(bdata_name)

# Collect the results
cr = pm.CollectResults()

# Get the Neural Network Results
for data_name in data_name_set:
    for cl in cl_set:
        print(data_name, cl)
        cr.compute(data_name=data_name, model_name=cl)

# Get the Base model results
for base_data_name in base_data_name_set:
    for bcl in base_cl_set:
        cr.compute(data_name=base_data_name, model_name=bcl)

df = cr.make_dataframe()

# Save results to dataframe
df.to_csv("../Results/penultimate_results_nn_base.csv", index=False)
