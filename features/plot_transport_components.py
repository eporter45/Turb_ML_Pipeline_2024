import torch
import numpy as np
from Trials import TRIALS
import os
from Preprocessing.Load_data import load_case_data, get_case_features
from Preprocessing.load_stack import load_and_stack_rans
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

from features.nondim_norms import get_local_nondimensional_scalars, apply_local_nondim


# === Trial and Config ===
trial = 'phill_inter'
train_cases = TRIALS[trial]['train']

input_feature_names = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
output_feature_names = ['p']

# === Load data ===
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")  # Make sure root_dir is defined
x_train, y_train, grid_train = load_and_stack_rans(
    cases=train_cases,
    input_feature_names=input_feature_names,
    output_feature_names=output_feature_names,
    data_path=data_path,
)

scalars = get_local_nondimensional_scalars(
    x_tensor_list=x_train,
    input_feature_names=input_feature_names,
    grid_dicts=grid_train,
    case_names=train_cases
)

x_train_nondim, _ = apply_local_nondim(
    tensor_list=x_train,
    input_feature_names=input_feature_names,
    scalars=scalars
)


def compute_transport_components_2D(x_tensor: torch.Tensor, input_feature_names):
    """
    Compute the 4 component terms of the transport expression:
    T = u_i u_j ∂u_i/∂x_j in 2D
    Assuming features are ordered: ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
    """

    # Get indices
    ix = {name: idx for idx, name in enumerate(input_feature_names)}

    Ux = x_tensor[:, ix['Ux']]
    Uy = x_tensor[:, ix['Uy']]
    dUx_dx = x_tensor[:, ix['dUx_dx']]
    dUx_dy = x_tensor[:, ix['dUx_dy']]
    dUy_dx = x_tensor[:, ix['dUy_dx']]
    dUy_dy = x_tensor[:, ix['dUy_dy']]

    # Compute components
    T_xx = Ux * Ux * dUx_dx
    T_xy = Ux * Uy * dUx_dy
    T_yx = Uy * Ux * dUy_dx
    T_yy = Uy * Uy * dUy_dy

    return {
        'T_xx': T_xx,
        'T_xy': T_xy,
        'T_yx': T_yx,
        'T_yy': T_yy
    }

def normalize_transport_components_by_mode(T_dict: Dict[str, torch.Tensor], mode="mean", eps=1e-12) -> Dict[str, torch.Tensor]:
    """
    Apply normalization to transport components using specified strategy.

    Args:
        T_dict: dict of transport components {T_xx, T_xy, ...}
        mode: one of ['log', 'mean', 'max', 'mag', 'local_frob']
        eps: small value to prevent div by zero or log(0)

    Returns:
        A new dict with normalized T_ij components
    """
    if mode == "log":
        return {
            k: torch.sign(v) * torch.log10(torch.clamp(torch.abs(v), min=eps))
            for k, v in T_dict.items()
        }

    elif mode == "mean":
        all_vals = torch.cat([torch.abs(v).flatten() for v in T_dict.values()])
        mean_val = torch.mean(all_vals)
        return {k: v / (mean_val + eps) for k, v in T_dict.items()}

    elif mode == "max":
        all_vals = torch.cat([torch.abs(v).flatten() for v in T_dict.values()])
        max_val = torch.max(all_vals)
        return {k: v / (max_val + eps) for k, v in T_dict.items()}

    elif mode == "mag":
        all_squared = torch.stack([v**2 for v in T_dict.values()])
        mag_val = torch.sqrt(torch.sum(all_squared))  # Frobenius-like norm over domain
        return {k: v / (mag_val + eps) for k, v in T_dict.items()}

    elif mode == "local_frob":
        # Stack (n_points, 4), then compute local Frobenius norm per point
        T_stack = torch.stack([v for v in T_dict.values()], dim=1)  # shape (n_points, 4)
        frob_norm = torch.sqrt(torch.sum(T_stack**2, dim=1))  # shape (n_points,)
        return {
            k: v / (frob_norm + eps)
            for k, v in T_dict.items()
        }

    elif mode == "":
        return T_dict

    else:
        raise ValueError(f"Unsupported mode: {mode}")




transport_components = {}
for idx, tensor in enumerate(x_train_nondim):
    transport_components[train_cases[idx]] = compute_transport_components_2D(tensor, input_feature_names)


norm = 'local_frob'

ind = 2
case = train_cases[ind]
T_comp = transport_components[case]
case_grid = grid_train[case]

T_comp_scaled = normalize_transport_components_by_mode(T_comp, mode=norm)

#def plot_single_figure(grid_dict, feature_1d, feature_name, case_name):
from Plotting.plot_pressure import plot_single_figure_log as plot_feature

title = 'nondim ' + norm
for name, feature in T_comp.items():
    eps = 1e-12
    feature_clean = torch.nan_to_num(feature, nan=eps, posinf=eps, neginf=eps)
    #feature_clean = torch.clamp(feature_clean, min=1e-12)  # Avoid log(0)
    fig, ax = plot_feature(case_grid, feature_clean, name, case, norm_name=title)
    plt.show()