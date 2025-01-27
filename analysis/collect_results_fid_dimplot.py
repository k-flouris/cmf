#!/usr/bin/env python3

# run:  python collect_results_fid_dimplot.py -d ../runs/images/train_dimensions_3rd

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


_DEFAULT_RUNS_DIRECTORY = "../runs/images_collected_fortesting"
_OUTPUT_FILE = "images_fid_table_dimplot.csv"
_EXPECTED_ENTRIES = 5
_LAMBDAS=[0, 0.01]
_DATASETS = ["fashion-mnist"]#, "fashion-mnist"]
_METHODS = _LAMBDAS #["RNF", f"CML-l-{_l_SMALL}",f"CML-l-{_l_MED}", f"CML-l-{_l_LARGE}"]
_DIMS = [5,10,15,20,30,40]
DATASET =_DATASETS[0] # only one dataset for now ;p
savedir='results_tables/'
_FS=10
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


# Create full results table, and directory for runs if args.move_runs == True.
# You will get errors if the directories exist.
results_table = {}
for lam in _LAMBDAS:
    method=str(lam)
    results_table[method] = {}
    for dim in _DIMS:
        dim_=str(dim)
        results_table[method][dim_] = {}
        for dataset in _DATASETS:
            results_table[method][dim_][dataset] = []   

            
print(results_table)   

# Test all relevant runs and move to new directories
for run in all_runs:
    try:
        with open(os.path.join(run, "config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no config")
        continue

    try:
        with open(os.path.join(run, "metrics.json"), "r") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no metrics")
        continue

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
        method = str(lamb)
        dim_=str(config_dim)
        
    fid_like_metric = metrics["test_fid"]
    if fid_like_metric > 1000:
        continue
    else:
        results_table[method][dim_][dataset].append(fid_like_metric)
    

fig, ax = plt.subplots(1, 1)
# Collect all the results, excluding NaN values. Give warnings for NaNs and fewer entries than expected.
csv_output = "Method(lam)_ld, " + ", ".join(_DATASETS) + "\n"
for method, dims in results_table.items():
    dim_list=[]
    FID_list=[]
    error_list=[]
    for dim, datasets in dims.items():
        csv_output += f"{method}"+f"_{dim}"
        dim_list.append(int(dim))

        for dataset, results in datasets.items():
            if len(results) < _EXPECTED_ENTRIES:
                print(f"Warning: Method {method} on dataset {dataset} only has {len(results)} entries")
            
            num_non_nan = np.sum(~np.isnan(results))
            if num_non_nan < len(results):
                print(f"Warning: Method {method} on dataset {dataset} has only {num_non_nan} non-NaN entries")
            mean = np.nanmean(results)
            # minimum = np.nanmin(results)
            stderr = np.nanstd(results, ddof=1)/np.sqrt(num_non_nan)
            print(dataset, results)
            
            FID_list.append(mean)
            error_list.append(stderr)
        
            
            csv_output += f", {mean:.3f} +/- {stderr:.3f}"
        csv_output += "\n"
    if method=='0':
        labe="RNF"
    elif method=='0.01':    
        labe="CMF"
    else:
        labe="lam="+method

    plt.plot(dim_list,FID_list,'-o', ms=10, label=labe)
    # plt.errorbar(dim_list, FID_list, yerr=error_list, fmt='.'); # NB if error is not nan you can plot this 



ax.set_xlabel(r'd', fontsize=_FS)
ax.set_ylabel(r'FID score', fontsize=_FS)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1], loc=1, frameon=False, fontsize=_FS)
plt.title
plt.tight_layout()
plt.savefig(savedir+'/fid_vs_dim_{}.pdf'.format(DATASET), bbox_inches='tight')

print(csv_output)
with open(_OUTPUT_FILE, "w") as f:
    f.write(csv_output)
