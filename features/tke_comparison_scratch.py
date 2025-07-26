import numpy as np
import torch
import os
import pandas as pd
import matplotlib.pyplot as plt
from Plotting.plot_pressure import plot_case_prediction
from Preprocessing.load_stack import load_and_stack_les
from Trials import TRIALS

# === Trial and Config ===
trial = 'full_extrap'
train_cases = TRIALS[trial]['train']
input_feature_names = ['k']
output_feature_names = ['uu', 'vv', 'ww']
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")  # Make sure root_dir is defined

#load data
x_train, y_train, grid_train = load_and_stack_les(
    cases=train_cases,
    input_feature_names=input_feature_names,
    output_feature_names=output_feature_names,
    data_path=data_path
)
tke_list = [0.5 * torch.sum(y[:, 0:3], dim=1) for y in y_train]  # Each is (n_points,)

for idx, tke in enumerate(tke_list):
    x = x_train[idx]
    case = train_cases[idx]
    grid = grid_train[case]
    plot_case_prediction(x, tke, 'tke_comparison', case, vmin_shared=True, grid_dict=grid)

    #plot_case_prediction(truth, pred, save_dir, case_name, vmin_shared=True, grid_dict=None):


