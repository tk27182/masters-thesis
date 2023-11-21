import numpy as np
from scipy.stats import norm
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
        filename = f"../Results/{subj}_hypertuning_results/{data_name}"
        results_folder = f"{model_name}_results"

        try:
            results = np.load(f"{filename}/{results_folder}/results.npz")

            # pred_results = np.load(filename / "predictions.npz")
            # targ_results = np.load(filename / "targets.npz")
            # resz_results = np.load(filename / "resource_metrics.npz")
        except FileNotFoundError:
            try:
                results = np.load(filename + "\r/" + results_folder + "/results.npz")

            except FileNotFoundError:
                results = {'train_preds': np.nan, 'val_preds': np.nan, 'test_preds': np.nan,
                        'train_preds_100': np.nan, 'val_preds_100': np.nan, 'test_preds_100': np.nan,
                        'train_preds_200': np.nan, 'val_preds_200': np.nan, 'test_preds_200': np.nan,
                        'train_preds_500': np.nan, 'val_preds_500': np.nan, 'test_preds_500': np.nan,
                        'train_preds_max': np.nan, 'val_preds_max': np.nan, 'test_preds_max': np.nan,

                            'train_target': np.nan, 'val_target': np.nan, 'test_target': np.nan,
                            'rez': np.nan, 'time': np.nan
                        }

            # pred_results = {'train_preds': np.nan, 'val_preds': np.nan, 'test_preds': np.nan}
            # targ_results = {'train_target': np.nan, 'val_target': np.nan, 'test_target': np.nan}
            # resz_results = {'rez': np.nan, 'time': np.nan}

        return results # pred_results, targ_results, resz_results

    def compute(self, data_name, model_name):

        results = self.load_results(data_name=data_name, model_name=model_name)
        # preds, targets, resz = self.load_results(data_name=data_name, model_name=model_name)

        train_preds, val_preds, test_preds = results['train_preds'], results['val_preds'], results['test_preds']
        # train_preds_100, val_preds_100, test_preds_100 = results['train_preds_100'], results['val_preds_100'], results['test_preds_100']
        # train_preds_200, val_preds_200, test_preds_200 = results['train_preds_200'], results['val_preds_200'], results['test_preds_200']
        # train_preds_500, val_preds_500, test_preds_500 = results['train_preds_500'], results['val_preds_500'], results['test_preds_500']
        # train_preds_max, val_preds_max, test_preds_max = results['train_preds_max'], results['val_preds_max'], results['test_preds_max']

        train_target, val_target, test_target = results['train_target'], results['val_target'], results['test_target']
        resources, total_time = results['rez'], results['time']

        # train_preds, val_preds, test_preds = preds['train_preds'], preds['val_preds'], preds['test_preds']
        # train_target, val_target, test_target = targets['train_target'], targets['val_target'], targets['test_target']
        # resources, total_time = resz['rez'], resz['time']

        # This might be an issue for the Autoencoder models
        if isinstance(train_preds, np.ndarray):
            if train_preds.shape[1] == 2:
                train_preds = train_preds[:,0]
                test_preds  = test_preds[:,0]
                val_preds   = val_preds[:,0]

        # Compute the AU-ROC and AU-PRC
        if isinstance(train_preds, float) or isinstance(train_target, float):
            train_auroc = np.nan
            train_auprc = np.nan

            train_auroc_100 = np.nan
            train_auroc_200 = np.nan
            train_auroc_500 = np.nan
            train_auroc_max = np.nan

            train_auprc_100 = np.nan
            train_auprc_200 = np.nan
            train_auprc_500 = np.nan
            train_auprc_max = np.nan

        else:
            train_auroc = self._auroc(train_preds, train_target)
            train_auprc = self._auprc(train_preds, train_target)

            # train_auroc_100 = self._auroc(train_preds_100, train_target)
            # train_auroc_200 = self._auroc(train_preds_200, train_target)
            # train_auroc_500 = self._auroc(train_preds_500, train_target)
            # train_auroc_max = self._auroc(train_preds_max, train_target)

            # train_auprc_100 = self._auprc(train_preds_100, train_target)
            # train_auprc_200 = self._auprc(train_preds_200, train_target)
            # train_auprc_500 = self._auprc(train_preds_500, train_target)
            # train_auprc_max = self._auprc(train_preds_max, train_target)

        if isinstance(val_preds, float) or isinstance(val_target, float):
            val_auroc = np.nan
            val_auprc = np.nan

            val_auroc_100 = np.nan
            val_auroc_200 = np.nan
            val_auroc_500 = np.nan
            val_auroc_max = np.nan

            val_auprc_100 = np.nan
            val_auprc_200 = np.nan
            val_auprc_500 = np.nan
            val_auprc_max = np.nan
        else:
            val_auroc   = self._auroc(val_preds, val_target)
            val_auprc   = self._auprc(val_preds, val_target)

            # val_auroc_100 = self._auroc(val_preds_100, val_target)
            # val_auroc_200 = self._auroc(val_preds_200, val_target)
            # val_auroc_500 = self._auroc(val_preds_500, val_target)
            # val_auroc_max = self._auroc(val_preds_max, val_target)

            # val_auprc_100 = self._auprc(val_preds_100, val_target)
            # val_auprc_200 = self._auprc(val_preds_200, val_target)
            # val_auprc_500 = self._auprc(val_preds_500, val_target)
            # val_auprc_max = self._auprc(val_preds_max, val_target)

        if isinstance(test_preds, float) or isinstance(test_target, float):
            test_auroc = np.nan
            test_auprc = np.nan

            test_auroc_100 = np.nan
            test_auroc_200 = np.nan
            test_auroc_500 = np.nan
            test_auroc_max = np.nan

            test_auprc_100 = np.nan
            test_auprc_200 = np.nan
            test_auprc_500 = np.nan
            test_auprc_max = np.nan
        else:
            test_auroc  = self._auroc(test_preds, test_target)
            test_auprc  = self._auprc(test_preds, test_target)

            # test_auroc_100 = self._auroc(test_preds_100, test_target)
            # test_auroc_200 = self._auroc(test_preds_200, test_target)
            # test_auroc_500 = self._auroc(test_preds_500, test_target)
            # test_auroc_max = self._auroc(test_preds_max, test_target)

            # test_auprc_100 = self._auprc(test_preds_100, test_target)
            # test_auprc_200 = self._auprc(test_preds_200, test_target)
            # test_auprc_500 = self._auprc(test_preds_500, test_target)
            # test_auprc_max = self._auprc(test_preds_max, test_target)


        # Check if resources and time were recorded
        if isinstance(resources, float):
            resources = np.nan
        else:
            resources = resources.tolist()

        if isinstance(total_time, float):
            total_time = np.nan
        else:
            total_time = total_time.tolist()

        # Calculate the AUC confidence intervals
        train_ci_l, train_ci_u, train_ci_p, _, _             = self.estimate_auc_interval(train_target, train_preds)
        # train_ci_l_100, train_ci_u_100, train_ci_p_100, _, _ = self.estimate_auc_interval(train_target, train_preds_100)
        # train_ci_l_200, train_ci_u_200, train_ci_p_200, _, _ = self.estimate_auc_interval(train_target, train_preds_200)
        # train_ci_l_500, train_ci_u_500, train_ci_p_500, _, _ = self.estimate_auc_interval(train_target, train_preds_500)
        # train_ci_l_max, train_ci_u_max, train_ci_p_max, _, _ = self.estimate_auc_interval(train_target, train_preds_max)

        val_ci_l, val_ci_u, val_ci_p, _, _             = self.estimate_auc_interval(val_target, val_preds)
        # val_ci_l_100, val_ci_u_100, val_ci_p_100, _, _ = self.estimate_auc_interval(val_target, val_preds_100)
        # val_ci_l_200, val_ci_u_200, val_ci_p_200, _, _ = self.estimate_auc_interval(val_target, val_preds_200)
        # val_ci_l_500, val_ci_u_500, val_ci_p_500, _, _ = self.estimate_auc_interval(val_target, val_preds_500)
        # val_ci_l_max, val_ci_u_max, val_ci_p_max, _, _ = self.estimate_auc_interval(val_target, val_preds_max)

        test_ci_l, test_ci_u, test_ci_p, _, _             = self.estimate_auc_interval(test_target, test_preds)
        # test_ci_l_100, test_ci_u_100, test_ci_p_100, _, _ = self.estimate_auc_interval(test_target, test_preds_100)
        # test_ci_l_200, test_ci_u_200, test_ci_p_200, _, _ = self.estimate_auc_interval(test_target, test_preds_200)
        # test_ci_l_500, test_ci_u_500, test_ci_p_500, _, _ = self.estimate_auc_interval(test_target, test_preds_500)
        # test_ci_l_max, test_ci_u_max, test_ci_p_max, _, _ = self.estimate_auc_interval(test_target, test_preds_max)


        # Store the results in the default dict
        self.results_dict['Data Name'].append(data_name)
        self.results_dict['Model'].append(model_name)

        ### Best Epoch ###
        self.results_dict['Train AU-ROC Best'].append(train_auroc)
        self.results_dict['Val AU-ROC Best'].append(val_auroc)
        self.results_dict['Test AU-ROC Best'].append(test_auroc)

        self.results_dict['Train AU-PRC Best'].append(train_auprc)
        self.results_dict['Val AU-PRC Best'].append(val_auprc)
        self.results_dict['Test AU-PRC Best'].append(test_auprc)

        self.results_dict['Train AUC CI Best'].append(f"[{train_ci_l:.3f}, {train_ci_u:.3f}]")
        self.results_dict['Val AUC CI Best'].append(f"[{val_ci_l:.3f}, {val_ci_u:.3f}]")
        self.results_dict['Test AUC CI Best'].append(f"[{test_ci_l:.3f}, {test_ci_u:.3f}]")

        self.results_dict['Train AUC CI p-value Best'].append(train_ci_p)
        self.results_dict['Val AUC CI p-value Best'].append(val_ci_p)
        self.results_dict['Test AUC CI p-value Best'].append(test_ci_p)

        #self.results_dict['Train AU-ROC Best'].append(train_auroc)
        #self.results_dict['Val AU-ROC Best'].append(val_auroc)
        #self.results_dict['Test AU-ROC Best'].append(test_auroc)

        #self.results_dict['Train AU-PRC Best'].append(train_auprc)
        #self.results_dict['Val AU-PRC Best'].append(val_auprc)
        #self.results_dict['Test AU-PRC Best'].append(test_auprc)

        #self.results_dict['Train AUC CI Best'].append(f"[{train_ci_l:.3f}, {train_ci_u:.3f}]")
        #self.results_dict['Val AUC CI Best'].append(f"[{val_ci_l:.3f}, {val_ci_u:.3f}]")
        #self.results_dict['Test AUC CI Best'].append(f"[{test_ci_l:.3f}, {test_ci_u:.3f}]")

        #self.results_dict['Train AUC CI p-value Best'].append(train_ci_p)
        #self.results_dict['Val AUC CI p-value Best'].append(val_ci_p)
        #self.results_dict['Test AUC CI p-value Best'].append(test_ci_p)
        ###########

        ### Epoch 100 ###
        # self.results_dict['Train AU-ROC 100'].append(train_auroc_100)
        # self.results_dict['Val AU-ROC 100'].append(val_auroc_100)
        # self.results_dict['Test AU-ROC 100'].append(test_auroc_100)

        # self.results_dict['Train AU-PRC 100'].append(train_auprc_100)
        # self.results_dict['Val AU-PRC 100'].append(val_auprc_100)
        # self.results_dict['Test AU-PRC 100'].append(test_auprc_100)

        # self.results_dict['Train AUC CI 100'].append(f"[{train_ci_l_100:.3f}, {train_ci_u_100:.3f}]")
        # self.results_dict['Val AUC CI 100'].append(f"[{val_ci_l_100:.3f}, {val_ci_u_100:.3f}]")
        # self.results_dict['Test AUC CI 100'].append(f"[{test_ci_l_100:.3f}, {test_ci_u_100:.3f}]")

        # self.results_dict['Train AUC CI p-value 100'].append(train_ci_p_100)
        # self.results_dict['Val AUC CI p-value 100'].append(val_ci_p_100)
        # self.results_dict['Test AUC CI p-value 100'].append(test_ci_p_100)

        # self.results_dict['Train AU-ROC 100'].append(train_auroc_100)
        # self.results_dict['Val AU-ROC 100'].append(val_auroc_100)
        # self.results_dict['Test AU-ROC 100'].append(test_auroc_100)

        # self.results_dict['Train AU-PRC 100'].append(train_auprc_100)
        # self.results_dict['Val AU-PRC 100'].append(val_auprc_100)
        # self.results_dict['Test AU-PRC 100'].append(test_auprc_100)

        # self.results_dict['Train AUC CI 100'].append(f"[{train_ci_l_100:.3f}, {train_ci_u_100:.3f}]")
        # self.results_dict['Val AUC CI 100'].append(f"[{val_ci_l_100:.3f}, {val_ci_u_100:.3f}]")
        # self.results_dict['Test AUC CI 100'].append(f"[{test_ci_l_100:.3f}, {test_ci_u_100:.3f}]")

        # self.results_dict['Train AUC CI p-value 100'].append(train_ci_p_100)
        # self.results_dict['Val AUC CI p-value 100'].append(val_ci_p_100)
        # self.results_dict['Test AUC CI p-value 100'].append(test_ci_p_100)
        # ###########

        # ### Epoch 200 ###
        # self.results_dict['Train AU-ROC 200'].append(train_auroc_200)
        # self.results_dict['Val AU-ROC 200'].append(val_auroc_200)
        # self.results_dict['Test AU-ROC 200'].append(test_auroc_200)

        # self.results_dict['Train AU-PRC 200'].append(train_auprc_200)
        # self.results_dict['Val AU-PRC 200'].append(val_auprc_200)
        # self.results_dict['Test AU-PRC 200'].append(test_auprc_200)

        # self.results_dict['Train AUC CI 200'].append(f"[{train_ci_l_200:.3f}, {train_ci_u_200:.3f}]")
        # self.results_dict['Val AUC CI 200'].append(f"[{val_ci_l_200:.3f}, {val_ci_u_200:.3f}]")
        # self.results_dict['Test AUC CI 200'].append(f"[{test_ci_l_200:.3f}, {test_ci_u_200:.3f}]")

        # self.results_dict['Train AUC CI p-value 200'].append(train_ci_p_200)
        # self.results_dict['Val AUC CI p-value 200'].append(val_ci_p_200)
        # self.results_dict['Test AUC CI p-value 200'].append(test_ci_p_200)

        # self.results_dict['Train AU-ROC 200'].append(train_auroc_200)
        # self.results_dict['Val AU-ROC 200'].append(val_auroc_200)
        # self.results_dict['Test AU-ROC 200'].append(test_auroc_200)

        # self.results_dict['Train AU-PRC 200'].append(train_auprc_200)
        # self.results_dict['Val AU-PRC 200'].append(val_auprc_200)
        # self.results_dict['Test AU-PRC 200'].append(test_auprc_200)

        # self.results_dict['Train AUC CI 200'].append(f"[{train_ci_l_200:.3f}, {train_ci_u_200:.3f}]")
        # self.results_dict['Val AUC CI 200'].append(f"[{val_ci_l_200:.3f}, {val_ci_u_200:.3f}]")
        # self.results_dict['Test AUC CI 200'].append(f"[{test_ci_l_200:.3f}, {test_ci_u_200:.3f}]")

        # self.results_dict['Train AUC CI p-value 200'].append(train_ci_p_200)
        # self.results_dict['Val AUC CI p-value 200'].append(val_ci_p_200)
        # self.results_dict['Test AUC CI p-value 200'].append(test_ci_p_200)
        # ###########

        # ### Epoch 500 ###
        # self.results_dict['Train AU-ROC 500'].append(train_auroc_500)
        # self.results_dict['Val AU-ROC 500'].append(val_auroc_500)
        # self.results_dict['Test AU-ROC 500'].append(test_auroc_500)

        # self.results_dict['Train AU-PRC 500'].append(train_auprc_500)
        # self.results_dict['Val AU-PRC 500'].append(val_auprc_500)
        # self.results_dict['Test AU-PRC 500'].append(test_auprc_500)

        # self.results_dict['Train AUC CI 500'].append(f"[{train_ci_l_500:.3f}, {train_ci_u_500:.3f}]")
        # self.results_dict['Val AUC CI 500'].append(f"[{val_ci_l_500:.3f}, {val_ci_u_500:.3f}]")
        # self.results_dict['Test AUC CI 500'].append(f"[{test_ci_l_500:.3f}, {test_ci_u_500:.3f}]")

        # self.results_dict['Train AUC CI p-value 500'].append(train_ci_p_500)
        # self.results_dict['Val AUC CI p-value 500'].append(val_ci_p_500)
        # self.results_dict['Test AUC CI p-value 500'].append(test_ci_p_500)

        # self.results_dict['Train AU-ROC 500'].append(train_auroc_500)
        # self.results_dict['Val AU-ROC 500'].append(val_auroc_500)
        # self.results_dict['Test AU-ROC 500'].append(test_auroc_500)

        # self.results_dict['Train AU-PRC 500'].append(train_auprc_500)
        # self.results_dict['Val AU-PRC 500'].append(val_auprc_500)
        # self.results_dict['Test AU-PRC 500'].append(test_auprc_500)

        # self.results_dict['Train AUC CI 500'].append(f"[{train_ci_l_500:.3f}, {train_ci_u_500:.3f}]")
        # self.results_dict['Val AUC CI 500'].append(f"[{val_ci_l_500:.3f}, {val_ci_u_500:.3f}]")
        # self.results_dict['Test AUC CI 500'].append(f"[{test_ci_l_500:.3f}, {test_ci_u_500:.3f}]")

        # self.results_dict['Train AUC CI p-value 500'].append(train_ci_p_500)
        # self.results_dict['Val AUC CI p-value 500'].append(val_ci_p_500)
        # self.results_dict['Test AUC CI p-value 500'].append(test_ci_p_500)
        # ###########

        # ### Max Epoch ###
        # self.results_dict['Train AU-ROC Max'].append(train_auroc_max)
        # self.results_dict['Val AU-ROC Max'].append(val_auroc_max)
        # self.results_dict['Test AU-ROC Max'].append(test_auroc_max)

        # self.results_dict['Train AU-PRC Max'].append(train_auprc_max)
        # self.results_dict['Val AU-PRC Max'].append(val_auprc_max)
        # self.results_dict['Test AU-PRC Max'].append(test_auprc_max)

        # self.results_dict['Train AUC CI Max'].append(f"[{train_ci_l_max:.3f}, {train_ci_u_max:.3f}]")
        # self.results_dict['Val AUC CI Max'].append(f"[{val_ci_l_max:.3f}, {val_ci_u_max:.3f}]")
        # self.results_dict['Test AUC CI Max'].append(f"[{test_ci_l_max:.3f}, {test_ci_u_max:.3f}]")

        # self.results_dict['Train AUC CI p-value Max'].append(train_ci_p_max)
        # self.results_dict['Val AUC CI p-value Max'].append(val_ci_p_max)
        # self.results_dict['Test AUC CI p-value Max'].append(test_ci_p_max)

        # self.results_dict['Train AU-ROC Max'].append(train_auroc_max)
        # self.results_dict['Val AU-ROC Max'].append(val_auroc_max)
        # self.results_dict['Test AU-ROC Max'].append(test_auroc_max)

        # self.results_dict['Train AU-PRC Max'].append(train_auprc_max)
        # self.results_dict['Val AU-PRC Max'].append(val_auprc_max)
        # self.results_dict['Test AU-PRC Max'].append(test_auprc_max)

        # self.results_dict['Train AUC CI Max'].append(f"[{train_ci_l_max:.3f}, {train_ci_u_max:.3f}]")
        # self.results_dict['Val AUC CI Max'].append(f"[{val_ci_l_max:.3f}, {val_ci_u_max:.3f}]")
        # self.results_dict['Test AUC CI Max'].append(f"[{test_ci_l_max:.3f}, {test_ci_u_max:.3f}]")

        # self.results_dict['Train AUC CI p-value Max'].append(train_ci_p_max)
        # self.results_dict['Val AUC CI p-value Max'].append(val_ci_p_max)
        # self.results_dict['Test AUC CI p-value Max'].append(test_ci_p_max)
        ###########

        self.results_dict['RAM'].append(resources)
        self.results_dict['Time'].append(total_time)

        return

    def estimate_auc_interval(self, target, predictions):

        # Reshape the predictions
        if isinstance(predictions, np.ndarray):
            predictions = predictions.reshape(-1, 1)

            N = sum(target != 0)
            M = len(target) - N

            if (N==1) & (M==1):
                raise ValueError("Cannot estimate confidence interval with <2 instances.")

            X = predictions[target == 0]
            Y = predictions[target != 0]

            # P = np.tile(X, (1, len(Y))) - np.tile(Y.T, (len(X), 1))
            P1 = np.tile(X, (1, Y.shape[0]))
            P2 = np.tile(Y.T, (X.shape[0], 1))
            P = P1 - P2

            idx0 = np.argwhere(P == 0)

            P = np.where(P > 0, 0, P)
            P[idx0.T] = 0.5
            P = np.where(P < 0, 1, P)

            # Compute AUC and variance
            if len(X) == 1:
                V_x = np.mean(P, axis = 1)
                V_y = P
            elif len(Y) == 1:
                V_x = P
                V_y = np.mean(P, axis = 0)
            else:
                V_x = np.mean(P, axis = 1)
                V_y = np.mean(P, axis = 0)

            a = np.mean(V_x)
            var_x = np.var(V_x, ddof=1)
            var_y = np.var(V_y, ddof=1)
            var_a = var_x/N + var_y/M
            se_a = np.sqrt(var_a)
            
            if se_a < 1e-1:
                test_stat = np.inf
            else:
                test_stat = (a - 0.5)/se_a
            p = 1 - norm.cdf(np.abs(test_stat))
            ci_l = np.max([a - 1.96*se_a, 0])
            ci_u = np.min([a + 1.96*se_a, 1])

            return ci_l, ci_u, p, a, se_a

        else:
            return np.nan, np.nan, np.nan, np.nan, np.nan


    def make_dataframe(self):

        #df = pd.DataFrame.from_dict(self.results_dict)
        df = pd.DataFrame(self.results_dict)

        # Split the data_name columnn into separate ones
        new_cols = ['Model Type', 'ID', 'Sensor', 'Hours in Advance', 'Event', 'Augmentation', 'Epochs', 'Callbacks']
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
