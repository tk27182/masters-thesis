#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tues Feb 7 23 21:03:00

@author: kirsh012
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path

#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF as SquaredExponential, RationalQuadratic, ExpSineSquared
import george
from george import kernels
import emcee
import corner


import dataprocessing as dp

### Load the data as times series
left_df, right_df, target = dp.load_data_left_right('1-sf', sensor='both', dlh=0, keep_SH=True, keep_event=False)

left_target = target['SH_Event_l'].values
right_target = target['SH_Event_r'].values

left_data = dp.get_original_time_series(left_df.values)
right_data = dp.get_original_time_series(right_df.values)

n = left_data.shape[0]

train_idx = np.arange(0, int(.6*n))
val_idx = np.arange(int(.6*n), int(.8*n), 1)
test_idx = np.arange(int(.8*n), n, 1)

left_train_data = left_data[train_idx]
left_val_data = left_data[val_idx]
left_test_data = left_data[test_idx]

#left_train_target = left_target[train_idx]
#left_val_target   = left_target[val_idx]
#left_test_target  = left_target[test_idx]

right_train_data = right_data[train_idx]
right_val_data = right_data[val_idx]
right_test_data = right_data[test_idx]

#right_train_target = right_target[train_idx]
#right_val_target   = right_target[val_idx]
#right_test_target  = right_target[test_idx]

N = left_train_data.shape[0]

def make_kernel(length_scale = 1000.0, periodicity=1.0):

    k1 = SquaredExponential(length_scale)
    k2 = ExpSineSquared(length_scale, periodicity)

    return k1*k2

#kernel = make_kernel()
def get_kernel(A, gamma, logP, lamda):
    k1 = kernels.ExpSine2Kernel(gamma=gamma, log_period = logP)
    k2 = kernels.ExpSquaredKernel(metric = lamda)

    k = A * (k1*k2)
    return k


A = np.std(left_train_data)
gamma = 1
logP = np.log(15)
lamda = 100
print("A = {:.2f}\ngamma = {:.2f}\nlogP = {:.2f}\nlamda = {:.2f}".format(A, gamma, logP, lamda))

yerr = left_train_data - right_train_data

#gpl = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 9)
#gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 9)
#pl.fit(left_train_data, left_train_target)
#gpr.fit(right_train_data, right_train_target)

#left_val_preds = gpl.predict(left_val_data)
#right_val_preds = gpl.predict(right_val_data)

kernel = get_kernel(A, gamma, logP, lamda)

# Create the full model
gp = george.GP(kernel)
x = np.arange(0., N/4., 1./4.)

print("Computing Gaussian Process...")
gp.compute(x, yerr)

left_train_pred, left_train_pred_var = gp.predict(left_train_data, x, return_var = True)
right_train_pred, right_train_pred_var = gp.predict(right_train_data, x, return_var = True)

### Plot the Val predictions and values
def plot_predictions(left_real, left_pred, right_real, right_pred, savename):
    fig, axes = plt.subplots(figsize= (8,6), nrows = 2, ncols = 1)#

    axes[0].plot(x, left_train_data, label = 'Original Values - Left', color = 'black')
    axes[0].plot(x, left_train_pred, label = 'Predictions - Left', color = 'r', linestyle = '--')

    axes[1].plot(x, right_train_data, label = 'Original Values - Right', color = 'black')
    axes[1].plot(x, right_train_pred, label = 'Predictions - Right', color = 'r', linestyle = '--')

    fig.legend()

    fig.savefig(savename, dpi = 300, format='png')

plot_predictions(left_train_data, left_train_pred, right_train_data, right_train_pred, "../Results/figures/testtrain_og_gp.png")
#plt.show()
### Optimize the parameters
def lnprob(p):
    gp.set_parameter_vector(p)
    return gp.log_likelihood(left_train_data, quiet=True)

initial = gp.get_parameter_vector()
ndim, nwalkers = len(initial), 16
p0 = initial + 1e-6 * np.random.randn(nwalkers, ndim)
params = gp.get_parameter_names()

def plot_chains(sampler):
    fig, ax = plt.subplots(ndim, figsize=(8,10), sharex=True)
    for i in range(ndim):
        ax[i].plot(sampler.chain[:,:,i].T, '-k', alpha=0.2)
        ax[i].set_ylabel(params[i])

    fig.savefig("../Results/figures/testchains_1sf.png", dpi = 300)

# Set up the backend
# Don't forget to clear it in case the file already exists
filename = Path("testfile_1-sf.h5")
if filename.exists():
    sampler  = emcee.backends.HDFBackend(filename)
    
    print("The mean acceptance rate is: ", np.mean(sampler.accepted / sampler.iteration))
    #tau = reader.get_autocorr_time()
    
else:
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, backend=backend)

    print("Running production...")
    sampler.run_mcmc(p0, nsteps = 500, progress = True)
    plot_chains(sampler)
    #tau = sampler.get_autocorr_time()
    print("The mean acceptance rate is:", np.mean(sampler.acceptance_fraction))


def plot_chains_from_samples(samples):
    fig, ax = plt.subplots(ndim, figsize=(8,10), sharex=True)
    for i in range(ndim):
        ax[i].plot(samples[:,i], '-k', alpha=0.2)
        ax[i].set_ylabel(params[i])

    fig.savefig("../Results/figures/testchains_1sf.png", dpi = 300)



### Remove burnin
burnin = 200#int(2 * np.max(tau))
#thin = int(0.5 * np.min(tau))
samples = sampler.get_chain(discard=burnin, flat=True) #, thin=thin)

plot_chains_from_samples(samples)
### Get the best parameters
best_A     = np.median(samples[:, 0]) #np.median(sampler.flatchain[:, 0])
best_gamma = np.median(samples[:, 1]) #np.median(sampler.flatchain[:, 1])
best_logP  = np.median(samples[:, 2]) #np.median(sampler.flatchain[:, 2])
best_lamda = np.median(samples[:, 3]) #np.median(sampler.flatchain[:, 3])

print("The best parameters are: \nA = {:.2f}\ngamma = {:.2f}\nlogP = {:.2f}\nlamda = {:.2f}".format(best_A, best_gamma, best_logP, best_lamda))

### Show corner plot
figure = corner.corner(samples, labels = params, quantiles = [0.5], show_titles = True, title_kwargs={'fontsize': 12})
figure.savefig("testcornerplot_1sf.png", dpi=300)

### Create and run best model
best_kernel = get_kernel(best_A, best_gamma, best_logP, best_lamda)
best_gp = george.GP(best_kernel)

print("Computing Gaussian Process...")
best_gp.compute(x, yerr)

# Get best predictions to check fit
best_left_train_pred, best_left_train_pred_var   = best_gp.predict(left_train_data, x, return_var = True)
best_right_train_pred, best_right_train_pred_var = best_gp.predict(right_train_data, x, return_var = True)

test_x = np.arange(0., left_test_data.shape[0]/4., 1./4.)
print(test_x.shape)
print(left_test_data.shape)
best_left_test_pred, best_left_test_pred_var     = best_gp.predict(left_test_data, test_x, return_var = True)
best_right_test_pred, best_right_test_pred_var   = best_gp.predict(right_test_data, test_x, return_var = True)
val_x = np.arange(0., left_val_data.shape[0]/4., 1./4.)
best_left_val_pred, best_val_train_pred_var      = best_gp.predict(left_val_data, val_x, return_var = True)
best_right_val_pred, best_val_train_pred_var     = best_gp.predict(right_val_data, val_x, return_var = True)

plot_predictions(left_train_data, best_left_test_pred, right_train_data, best_right_train_pred, "../Results/figures/best_testtrain_predictions.png")
plot_predictions(left_val_data, best_left_val_pred, right_val_data, best_right_val_pred, "../Results/figures/best_testval_predictions.png")
plot_predictions(left_test_data, best_left_test_pred, right_test_data, best_right_test_pred, "../Results/figures/best_testtest_predictions.png")

plt.show()

