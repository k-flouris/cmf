#!/usr/bin/env python3

import glob
import os
import json
import shutil
import argparse
import numpy as np
import itertools
# import sys
 
# adding Folder_2/subfolder to the system path
# sys.path.insert(0, '../')

# from cif.experiment import test_and_visualize
_OOD_TYPES=['mnist_train','mnist_no_train', 'fmnist_train','fmnist_no_train']

_DEFAULT_RUNS_DIRECTORY = "../runs/images/train_dimensions"
_EXPECTED_ENTRIES = 5
_LAMBDAS=[0,0.001, 0.01,0.1,1]
_DATASETS = ["mnist", "fashion-mnist"]
_METHODS = _LAMBDAS #["RNF", f"CML-l-{_l_SMALL}",f"CML-l-{_l_MED}", f"CML-l-{_l_LARGE}"]
_DIMS = [5,10,15,20,25,30,40]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default=_DEFAULT_RUNS_DIRECTORY,
    help="Location for runs directory")
parser.add_argument("--overwrite-metrics", action="store_true",
    help="Overwrite existing metrics for each run")
parser.add_argument("--move-runs", action="store_true",
    help="Rearrange runs into hierarchical structure")
args = parser.parse_args()


all_runs = glob.glob(os.path.join(args.dir, "*"))
  
savedir='results_tables/'
if not os.path.exists(savedir):
    os.makedirs(savedir)
  
for oodtype in  _OOD_TYPES: 
    _OUTPUT_FILE = savedir+"images_ood_table_{}.csv".format(oodtype)

    # Create full results table, and directory for runs if args.move_runs == True.
    # You will get errors if the directories exist.
    results_table = {}
    for dim,lam in  itertools.product(_DIMS,_LAMBDAS):
        method="lam_"+str(lam)+"_dim_"+str(dim)
        results_table[method] = {}
        for dataset in _DATASETS:
            results_table[method][dataset] = []   
            if args.move_runs:
                os.makedirs(os.path.join(args.dir, method, dataset))
    
    # Test all relevant runs and move to new directories
    for run in all_runs:
        try:
            with open(os.path.join(run, "config.json"), "r") as f:
                config = json.load(f)
        except FileNotFoundError:
            print(f"Skipping {run} because no config")
            continue
    
    
        try:
            with open(os.path.join(run, "ood_metrics_mnist_train=True.json"), "r") as f:
                ood_metrics_mnist_train_True =json.load(f)
        except FileNotFoundError:
            print(f"Skipping {run} because no ood_metrics_mnist_train=True")
            continue
        try:
            with open(os.path.join(run, "ood_metrics_mnist_train=False.json"), "r") as f:
                ood_metrics_mnist_train_False =json.load(f)
        except FileNotFoundError:
            print(f"Skipping {run} because no ood_metrics_mnist_train=False")
            continue
        try:
            with open(os.path.join(run, "ood_metrics_fashion-mnist_train=False.json"), "r") as f:
                ood_metrics_fashion_mnist_train_True =json.load(f)
        except FileNotFoundError:
            print(f"Skipping {run} because no ood_metrics_fashion-mnist_train=False")
            continue
        try:
            with open(os.path.join(run, "ood_metrics_fashion-mnist_train=False.json"), "r") as f:
                ood_metrics_fashion_mnist_train_False =json.load(f)
        except FileNotFoundError:
            print(f"Skipping {run} because no ood_metrics_fashion-mnist_train=False")
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
            method = "lam_"+str(lamb)+"_dim_"+str(config_dim)
    
        likelihood_accuracy = ood_metrics_mnist_train_True["likelihood"]
        
        if oodtype== "mnist_train": results_table[method][dataset].append(ood_metrics_mnist_train_True["likelihood"])
        elif oodtype== "mnist_no_train":results_table[method][dataset].append(ood_metrics_mnist_train_False["likelihood"])
        elif oodtype== "fmnist_train":results_table[method][dataset].append(ood_metrics_fashion_mnist_train_True["likelihood"])
        elif oodtype== "fmnist_no_train":results_table[method][dataset].append(ood_metrics_fashion_mnist_train_False["likelihood"])
    
        if args.move_runs:
            shutil.move(run, os.path.join(args.dir, method, dataset, os.path.basename(run)))
    
    
    # Collect all the results, excluding NaN values. Give warnings for NaNs and fewer entries than expected.
    csv_output = "Method_ld, " + ", ".join(_DATASETS) + "\n"
    for method, datasets in results_table.items():
        csv_output += f"{method}"
        for dataset, results in datasets.items():
            if len(results) < _EXPECTED_ENTRIES:
                print(f"Warning: Method {method} on dataset {dataset} only has {len(results)} entries")
            num_non_nan = np.sum(~np.isnan(results))
            if num_non_nan < len(results):
                print(f"Warning: Method {method} on dataset {dataset} has only {num_non_nan} non-NaN entries")
    
            mean = np.nanmean(results)
            stderr = np.nanstd(results, ddof=1)/np.sqrt(num_non_nan)
    
            csv_output += f", {mean:.3f} +/- {stderr:.3f}"
    
        csv_output += "\n"
    
    
    print(csv_output)
    with open(_OUTPUT_FILE, "w") as f:
        f.write(csv_output)
