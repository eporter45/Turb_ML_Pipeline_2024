## this script calculates the gradients for a specific feature, such that i can pass them into my model

from Preprocessing.load_data_pressure import load_case_data, get_case_features
from features.compute_grads import compute_gradients_for_feature
import os
import numpy as np
import pickle as pkl
from tqdm import tqdm
import pandas as pd
from utils.convert_2D import to_grid, to_vector


# === Config ===
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(root_dir, "Data", "RANS_Dict_ext")

feature_to_grad = 'Uy'  # You can change this to 'Uy', etc.

# === Loop over all cases ===
for fname in tqdm(os.listdir(data_dir), desc="Processing RANS cases"):
    if not fname.endswith('_DICT.pkl') or fname.startswith('all'):
        continue

    case = fname.replace('_DICT.pkl', '')
    print(f"[INFO] Processing case: {case}")

    rans_dict = load_case_data(case, os.path.join(root_dir, "Data"), mesh_yn='no')

    # Check for keys
    if not all(k in rans_dict.columns for k in ['Cx', 'Cy', feature_to_grad]):
        print(f"[SKIP] {case} is missing required keys. Skipping.")
        continue

    grid_dict = {
        'Cx': rans_dict['Cx'].to_numpy(),
        'Cy': rans_dict['Cy'].to_numpy(),
        'case_name': case
    }
    feat = rans_dict[feature_to_grad].to_numpy()

    grads = compute_gradients_for_feature(feat, grid_dict, feature_to_grad)

    for k, grad in grads.items():
        rans_dict[k] = pd.Series(grad)

    with open(os.path.join(data_dir, fname), 'wb') as f:
        pkl.dump(rans_dict, f)

    print(f"[SUCCESS] Updated gradients added to {fname}")

#def compute_gradients_for_feature(feature_1d, grid_dict, feature_name):
