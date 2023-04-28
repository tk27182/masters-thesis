import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from collections import defaultdict

from sklearn.metrics import roc_curve, auc, precision_recall_curve

import visualization as viz

class CollectResults:
    def __init__(self):
        self.results_dict = defaultdict(list)

    def _auroc(self, preds, target, pos_label=1, return_all=False):

        fpr, tpr, thresh = roc_curve(target, preds, pos_label=pos_label)
        auroc = auc(fpr, tpr)
        return auroc

    def _auprc(self, preds, target, pos_label=1, return_all=False):

        ppr, rec, thresh = precision_recall_curve(target, preds, pos_label=pos_label)
        auprc = auc(rec, ppr)
        return auprc

    def load_results(self, data_name, model_name):

        subj = data_name.split('_')[1]
        filename = Path(f"../Results/{subj}_hypertuning_results/{data_name}/{model_name}_results/")

        try:
            pred_results = np.load(filename / "predictions.npz")
            targ_results = np.load(filename / "targets.npz")
            resz_results = np.load(filename / "resource_metrics.npz")
        except FileNotFoundError:
            pred_results = {'train_preds': np.nan, 'val_preds': np.nan, 'test_preds': np.nan}
            targ_results = {'train_target': np.nan, 'val_target': np.nan, 'test_target': np.nan}
            resz_results = {'rez': np.nan, 'time': np.nan}

        return pred_results, targ_results, resz_results

    def compute(self, data_name, model_name):

        preds, targets, resz = self.load_results(data_name=data_name, model_name=model_name)

        train_preds, val_preds, test_preds = preds['train_preds'], preds['val_preds'], preds['test_preds']
        train_target, val_target, test_target = targets['train_target'], targets['val_target'], targets['test_target']
        resources, total_time = resz['rez'], resz['time']
        
        if isinstance(train_preds, np.ndarray):
            if train_preds.shape[1] == 2:
                train_preds = train_preds[:,1]
                test_preds  = test_preds[:,1]
                val_preds   = val_preds[:,1]

        # Compute the AU-ROC and AU-PRC
        if isinstance(train_preds, float) or isinstance(train_target, float):
            train_auroc = np.nan
            train_auprc = np.nan
        else:
            train_auroc = self._auroc(train_preds, train_target)
            train_auprc = self._auprc(train_preds, train_target)

        if isinstance(val_preds, float) or isinstance(val_target, float):
            val_auroc = np.nan
            val_auprc = np.nan
        else:
            val_auroc   = self._auroc(val_preds, val_target)
            val_auprc   = self._auprc(val_preds, val_target)

        if isinstance(test_preds, float) or isinstance(test_target, float):
            test_auroc = np.nan
            test_auprc = np.nan
        else:
            test_auroc  = self._auroc(test_preds, test_target)
            test_auprc  = self._auprc(test_preds, test_target)


        # Check if resources and time were recorded
        if isinstance(resources, float):
            resources = np.nan
        else:
            resources = resources.tolist()

        if isinstance(total_time, float):
            total_time = np.nan
        else:
            total_time = total_time.tolist()

        # Store the results in the default dict
        self.results_dict['Data Name'].append(data_name)
        self.results_dict['Model'].append(model_name)

        self.results_dict['Train AU-ROC'].append(train_auroc)
        self.results_dict['Val AU-ROC'].append(val_auroc)
        self.results_dict['Test AU-ROC'].append(test_auroc)

        self.results_dict['Train AU-PRC'].append(train_auprc)
        self.results_dict['Val AU-PRC'].append(val_auprc)
        self.results_dict['Test AU-PRC'].append(test_auprc)

        self.results_dict['RAM'].append(resources)
        self.results_dict['Time'].append(total_time)

        return


    def make_dataframe(self):

        #df = pd.DataFrame.from_dict(self.results_dict)
        df = pd.DataFrame(self.results_dict)

        # Split the data_name columnn into separate ones
        new_cols = ['Model Type', 'ID', 'Sensor', 'Hours in Advance', 'Event', 'Augmentation']
        temp_df = (df['Data Name'].str.split('_', expand = True)
               .rename(columns={i: col for i, col in enumerate(new_cols)})
               )

        # Add temp_df to df
        df.drop('Data Name', axis = 1, inplace = True)
        df = pd.concat([temp_df, df], axis = 1)

        return df

    def plot_auc_results(self, df, xcol, ycol, hue):

        fig, ax = plt.subplots(figsize=(12,10))

        #g = sns.catplot(data=df, x)
        return ax
