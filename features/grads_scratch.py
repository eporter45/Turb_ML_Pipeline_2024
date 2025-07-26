from Preprocessing.load_data_pressure import load_case_data, get_case_features
from utils.convert_2D import to_grid
from Plotting.plot_pressure import plot_single_figure_log, plot_single_figure

import os
import numpy as np
import matplotlib.pyplot as plt


# === Config ===
case = 'BUMP_h26'
features = ['Ux', 'Uy', 'dp_dx', 'dp_dy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']  # You can change or extend this
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")

# === Load case dictionary ===
print(f"[INFO] Loading case: {case}")
rans_dict = load_case_data(case, data_path, mesh_yn='no')
print(f'[INFO] RANS DICT KEYS: {rans_dict.keys()} ')
# === Build grid_dict ===
grid_dict = {
    'Cx': rans_dict['Cx'],
    'Cy': rans_dict['Cy'],
    'case_name': case
}

# === Loop over and plot each gradient feature ===
for feat in features:
    try:
        grad_1d = get_case_features(rans_dict, feat)
        if np.isnan(grad_1d).any() or np.isinf(grad_1d).any():
            print(f'[INFO] Feature has NaNs or Infs: {feat} : {grad_1d.shape}')
        if feat != 'Ux' and feat != 'Uy':
            fig, ax = plot_single_figure_log(grid_dict, grad_1d, feat, case, base=10)
        elif feat == 'Ux' or feat == 'Uy':
            from Plotting.plot_pressure import plot_single_figure
            fig, ax = plot_single_figure(grid_dict, grad_1d, feat, case)
        else:
            print("oops something went wrong")
        plt.show()
        plt.close()
    except KeyError:
        print(f"[WARNING] {feat} not found in {case}")
