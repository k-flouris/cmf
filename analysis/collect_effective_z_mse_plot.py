#!/usr/bin/env python3
#  run python collect_test_metric.py -d ../runs/images/train_dimensions_2nd

import glob
import os
import json
import shutil
import argparse
import numpy as np
import itertools


from scipy.stats import norm
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (4,4)
import os
# import sys
 
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '../')

# from cif.experiment import test_and_visualize


_DEFAULT_RUNS_DIRECTORY = "../runs/metric_test/"
_OUTPUT_FILE = "images_test_metric_table.csv"
_LAMBDAS=[0, 0.01]
_DATASETS = ["fashion-mnist"]#, "fashion-mnist"]
_METHODS = _LAMBDAS #["RNF", f"CML-l-{_l_SMALL}",f"CML-l-{_l_MED}", f"CML-l-{_l_LARGE}"]
_DIMS = [20] #(30 fmnist 3rd)
DATASET =_DATASETS[0] # only one dataset for now ;p
_FS=10
savedir='results_tables/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
_OUTPUT_FILE=  savedir +_OUTPUT_FILE
    
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default=_DEFAULT_RUNS_DIRECTORY,
    help="Location for runs directory")
parser.add_argument("--overwrite-metrics", action="store_true",
    help="Overwrite existing metrics for each run")
args = parser.parse_args()


all_runs = glob.glob(os.path.join(args.dir, "*"))


results_table = {}
for lam in  _LAMBDAS:
    method=str(lam)
    results_table[method] = {}
            

# Test all relevant runs and move to new directories
for run in all_runs:
    try:
        with open(os.path.join(run, "config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no config")
        continue

    try:
        with open(os.path.join(run, "test_metric/recon.json"), "r") as f:
            reconfile = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no metrics")
        continue


    # print(reconfile)

    dataset = config["dataset"]
    config_dim=config["latent_dimension"]   
    lamb=config["metric_regularization_param"]   


    if not dataset in _DATASETS:
        print(f"Skipping {run} because {dataset} not of the desired image")
        continue
    elif not config["latent_dimension"] in _DIMS:
        print(f"Skipping {run} because {config_dim} not of the desired dimensions")
        continue
    elif not config["metric_regularization_param"] in _LAMBDAS:
        print(f"Skipping {run} because {lamb} not of the desired metric parameters")
        continue
    # Figure out the method by referencing the config        
    else:
        method =str(lamb)
        dim_=str(config_dim)
            #  it stores only the last one it finds btw
    results_table[method]=reconfile

# print(results_table)

fig, ax = plt.subplots(1, 1)
# # Collect all the results, excluding NaN values. Give warnings for NaNs and fewer entries than expected.
# csv_output = "Method(lam)_ld, " + ", ".join(_DATASETS) + "\n"
for method in results_table.items():
    dim_list=[]
    recon_list=[]
    error_list=[]
    print(method)
    for dims, recon in method[1].items():

        # if not dims==0:
            dim_list.append(dims)
            recon_list.append(recon)
    
                    
    if method[0]=='0':
        labe="RNF"
    elif method[0] in ['0.1',"0.01"] :    
        labe="CMF"
    else:
        labe="lam="+method[0]

    plt.plot(dim_list,recon_list,'-o', ms=10, label=labe)    # plt.errorbar(dim_list, FID_list, yerr=error_list, fmt='.'); # NB if error is not nan you can plot this 



ax.set_xlabel(r'effective d', fontsize=_FS)
ax.set_ylabel(r'$|| x - \hat{x}||^2_2$', fontsize=_FS)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=1, frameon=False, fontsize=_FS)
plt.title
plt.tight_layout()
plt.savefig(savedir+'/mse_vs_dim_effective_z_{}_ld{}.pdf'.format(DATASET,_DIMS[0]), bbox_inches='tight')

# print(csv_output)
# with open(_OUTPUT_FILE, "w") as f:
#     f.write(csv_output)
