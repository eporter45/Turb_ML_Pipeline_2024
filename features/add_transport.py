import torch
import numpy as np
import os
from typing import List, Dict

from Preprocessing.Load_data import load_case_data, get_case_features
from features.nondim_norms import get_local_nondimensional_scalars, apply_local_nondim

def calculate_dimless_transport_terms(case_names: List[str],
                                      data_path: str,
                                      input_feature_names: List[str],
                                      grid_dicts: Dict[str, Dict[str, np.ndarray]],
                                      feature_norm_key: str = 'nondim_local') -> List[torch.Tensor]:
    """
    Load required features from scratch and compute non-dimensional transport terms.
    """
    transport_inputs = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
    tensor_list = []

    for case in case_names:
        rans_dict, _ = load_case_data(case, data_path, mesh_yn='no')
        tensor = torch.tensor(
            np.vstack([get_case_features(rans_dict, feat) for feat in transport_inputs]).T,
            dtype=torch.float32
        )
        tensor_list.append(tensor)

    if feature_norm_key == 'nondim_local':
        scalars = get_local_nondimensional_scalars(tensor_list, transport_inputs, grid_dicts=grid_dicts, case_names=case_names)
        tensor_list, _ = apply_local_nondim(tensor_list, transport_inputs, scalars)
    elif feature_norm_key == '':
        pass  # Use dimensional form
    else:
        raise NotImplementedError(f"Feature norm '{feature_norm_key}' not supported yet.")

    # Compute transport term T = |ui uj ∂ui/∂xj|
    transport_terms = []
    for tensor in tensor_list:
        u = tensor[:, :2]             # Ux, Uy
        du = tensor[:, 2:].view(-1, 2, 2) # gradients reshaped

        u_vec = u.unsqueeze(-1)       # (N, 2, 1)
        u_T = u.unsqueeze(1)            # (N, 1, 2)

        product = torch.bmm(u_T, torch.bmm(du, u_vec)).squeeze()  # (N,)
        transport_terms.append(torch.abs(product))

    return transport_terms

def apply_scale_to_transport_feature(T_list: List[torch.Tensor], mode: str):
    if mode == '' or mode == 'none':
        return T_list, None

    if mode == 'abs':
        scalars = [torch.max(torch.abs(T)) for T in T_list]
        T_scaled = [T / (scalar + 1e-8) for T, scalar in zip(T_list, scalars)]

    elif mode == 'minmax':
        scalars = [(T.min(), T.max()) for T in T_list]
        T_scaled = [(T - tmin) / (tmax - tmin + 1e-8) for T, (tmin, tmax) in zip(T_list, scalars)]

    else:
        raise ValueError(f"[ERROR] Unknown transport scale norm mode '{mode}'")

    return T_scaled, scalars


def add_transport_feature(x_tensor_list,
                          case_names,
                          config,
                          grid_dicts,
                          data_path,
                          device):

    from Preprocessing.load_stack import load_and_stack_rans
    from features.nondim_norms import get_local_nondimensional_scalars

    transport_features = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
    x_stack, _, _ = load_and_stack_rans(case_names, transport_features, ['p'], data_path)
    U_L_scalars = get_local_nondimensional_scalars(x_stack, transport_features, grid_dicts, case_names)

    # Compute non-dimensional transport term
    T_list = []
    for tensor, scalar_dict in zip(x_stack, U_L_scalars):
        U_ref = float(scalar_dict['U_ref'])
        L_ref = float(scalar_dict['L_ref'])
        u = tensor[:, :2]
        du = tensor[:, 2:].view(-1, 2, 2)
        T_raw = torch.bmm(u.unsqueeze(1), torch.bmm(du, u.unsqueeze(-1))).squeeze()
        scale = pow(float(U_ref), 3) / (float(L_ref) + 1e-8)
        T_nondim = torch.abs(T_raw / scale)
        T_list.append(T_nondim)

    # Apply transport reduction (mean, max, log)
    reduction = config['features'].get('transport_norm', '')
    T_reduced = []
    for T in T_list:
        if reduction == 'mean':
            T_reduced.append(T / (T.mean() + 1e-8))
        elif reduction == 'max':
            T_reduced.append(T / (T.max() + 1e-8))
        elif reduction == 'log':
            T_reduced.append(torch.log1p(T))
        elif reduction == 'none' or reduction == '':
            T_reduced.append(T)
        else:
            raise ValueError(f"[ERROR] Unknown transport_reduction: {reduction}")

    # Apply global scale normalization (same as model inputs)
    scale_mode = config['training'].get('scale_norm', '')
    T_scaled, _ = apply_scale_to_transport_feature(T_reduced, scale_mode)

    # Append to input features
    output = []
    for x_tensor, T_col in zip(x_tensor_list, T_scaled):
        x_aug = torch.cat([x_tensor, T_col.unsqueeze(1).to(device)], dim=1)
        output.append(x_aug)

    return output

