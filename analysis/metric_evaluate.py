#!/usr/bin/env python3

import glob
import os
import json
import shutil
import argparse
import numpy as np

import sys
 
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '../')
sys.path.insert(0, '')

from cif.experiment import test_and_visualize, metric_test_plots


_DEFAULT_RUNS_DIRECTORY = "../runs/metric_test"
_OUTPUT_FILE = "metric_table.csv"

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", type=str, default=_DEFAULT_RUNS_DIRECTORY,
    help="Location for runs directory")
parser.add_argument("--overwrite-metrics", action="store_true",
    help="Overwrite existing metrics for each run")
args = parser.parse_args()


all_runs = glob.glob(os.path.join(args.dir, "*"))




# Test all relevant runs and move to new directories
for run in all_runs:
    try:
        with open(os.path.join(run, "config.json"), "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Skipping {run} because no config")
        continue

    dataset = config["dataset"]



# fid = test_and_visualize(config, run, overwrite=args.overwrite_metrics, test_fid=True)["test_fid"]
metric_test_plots(config, run)

