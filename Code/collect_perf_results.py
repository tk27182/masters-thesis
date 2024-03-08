import perfmetrics as pm

model_type_set = ['indv'] #, 'general']
subjects = ['3-jk', "1-sf", "10-rc", "12-mb", "17-sb", "19-me", "2-bd",
            "22-ap",  "31-ns", "32-rf", "36-af", "38-cs", "39-dg",
            "4-rs", "41-pk", "43-cm", "7-sb"] # "26-tc",

hours=["dlh0", "dlh1", 'dlh2'] #"dlh0", "dlh1", "dlh2", "dlh3_", "dlh4_", "dlh5_"]

sensor_set=['left']
events=['classification'] #regression
cl_set=['randomforest', 'lr', 'sgd', 'oneclassSVM-default', 'oneclassSVM', 'lstm', 'ann', 'simplernn']
base_cl_set=['randomforest', 'lr']
ae_set = ['autoencoder', 'lstm-autoencoder']
smote_set=["_None", "_downsample", "_smote"] #"_smote", "_gauss", "", "_downsample", "_original"]
epochs = ['Epochs1000'] #, 'Epochs200', 'Epochs500', 'Epochs1000']
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

data_name5000_set =  [
            f"{mdl}_{subj}_{sensor}_{hour}_{event}_None_Epochs5000_{cb}"
            for mdl in model_type_set
            for subj in subjects
            for sensor in sensor_set
            for hour in hours
            for event in events
            for cb in callbacks
        ]

# Base Model folders
# base_data_name_set = []
# for mdl in model_type_set:
#     for subj in subjects:
#         for sensor in sensor_set:
#             for hour in hours:
#                 for event in events:
#                     for smote in smote_set:
#                         # if smote == '_None':
#                         #     smote = ''

#                         bdata_name = f"{mdl}_{subj}_{sensor}_{hour}_{event}{smote}"
#                         base_data_name_set.append(bdata_name)

# Collect the results
cr = pm.CollectResults()

# Get the Neural Network Results
for data_name in data_name_set:
    for cl in cl_set:
        print(data_name, cl)
        cr.compute(data_name=data_name, model_name=cl)

# Get the Base model results
# for base_data_name in base_data_name_set:
#     for bcl in base_cl_set:
#         print(base_data_name, bcl)
#         cr.compute(data_name=base_data_name, model_name=bcl)

# Add the Autoencoder models
for data_name in data_name5000_set:
    for acl in ae_set:
        print(data_name, acl)
        cr.compute(data_name=data_name, model_name=acl)

df = cr.make_dataframe()

# Save results to dataframe
df.to_csv("../Results/ultimate_results_nn2.csv", index=False)
