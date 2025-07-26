import torch
import numpy as np
from typing import List, Dict

def get_feature_indices(names: List[str], all_features: List[str]) -> List[int]:
    return [all_features.index(name) for name in names]


from utils.convert_2D import to_grid


def get_local_nondimensional_scalars(x_tensor_list: List[torch.Tensor],
                                      input_feature_names: List[str],
                                      grid_dicts: Dict[str, Dict[str, torch.Tensor]],
                                      case_names: List[str]) -> List[Dict[str, float]]:
    """
    Get per-case non-dimensional scalars. Only stores what is needed based on present features.
    Returns list of dicts with keys like 'U_ref', 'L_ref', 'p_ref'.
    """
    scalar_dicts = []
    print(f'[Debug][get_local_nondim_scalars] Grid dict keys {grid_dicts.keys()}')

    for tensor, case_name in zip(x_tensor_list, case_names):
        scalars = {}
        print(f'[Debug][get_local_nondim_scalars] case_name {case_name}')

        # L_ref from Cx range
        if 'Cx' in grid_dicts[case_name]:
            cx = grid_dicts[case_name]['Cx'].flatten()
            scalars['L_ref'] = float((cx.max() - cx.min()).item())

        # U_ref from mode of Ux
        if 'Ux' in input_feature_names or 'Uy' in input_feature_names:
            ux_idx = input_feature_names.index('Ux')
            ux_vals = tensor[:, ux_idx].cpu().numpy()
            scalars['U_ref'] = float(np.round(float(torch.mode(torch.tensor(ux_vals))[0].item()), 6))

        # p_ref from left column of pressure field
        if 'p' in input_feature_names:
            p_idx = input_feature_names.index('p')
            p_vals = tensor[:, p_idx]
            p_grid = to_grid(p_vals, case_name)
            scalars['p_ref'] = float(p_grid[:, 0].mean().item())

        scalar_dicts.append(scalars)

    return scalar_dicts




def apply_local_nondim(tensor_list: List[torch.Tensor],
                       input_feature_names: List[str],
                       scalar_dicts: List[Dict[str, float]]) -> List[torch.Tensor]:
    """
    Applies nondimensionalization to all features present in each tensor
    based on what's available in the corresponding scalar dict.
    """

    # Feature groups
    vel_feats = ['Ux', 'Uy']
    grad_feats = ['dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
    pres_feats = ['p']
    pres_grad_feats = ['dp_0', 'dp_1']
    wall_feats = ['wallDistance']

    all_features = vel_feats + grad_feats + pres_feats + pres_grad_feats + wall_feats

    normed_list = []

    for tensor, scalars in zip(tensor_list, scalar_dicts):
        t = tensor.clone()

        for feat in all_features:
            if feat in input_feature_names:
                idx = input_feature_names.index(feat)

                # Determine scale factor
                if feat in vel_feats and 'U_ref' in scalars:
                    scale = scalars['U_ref']
                elif feat in grad_feats and 'U_ref' in scalars and 'L_ref' in scalars:
                    scale = scalars['U_ref'] / scalars['L_ref']
                elif feat in pres_feats and 'p_ref' in scalars:
                    scale = scalars['p_ref']
                elif feat in pres_grad_feats and 'p_ref' in scalars and 'L_ref' in scalars:
                    scale = scalars['p_ref'] / scalars['L_ref']
                elif feat in wall_feats and 'L_ref' in scalars:
                    scale = scalars['L_ref']
                else:
                    continue  # Skip if required scalar is missing

                t[:, idx] = t[:, idx] / (scale + 1e-8)

        normed_list.append(t)

    return normed_list, scalar_dicts



def apply_local_nondim_to_transport(T_components_list: List[Dict[str, torch.Tensor]],
                                     scalars: List[tuple]):
    """
    Normalize transport components using U_ref^3 / L_ref for each case.
    Input:
        T_components_list: list of dicts with T_ij components for each case
        scalars: list of (U_ref, L_ref) tuples for each case
    Returns:
        List of normalized T_component dicts
    """
    normed_components_list = []

    for T_dict, scalar_dict in zip(T_components_list, scalars):
        U_ref = scalar_dict.get('U_ref', 1.0)
        L_ref = scalar_dict.get('L_ref', 1.0)
        scale = (U_ref ** 3) / (L_ref + 1e-8)
        normed_dict = {k: v / (scale + 1e-12) for k, v in T_dict.items()}
        normed_components_list.append(normed_dict)

    return normed_components_list


from Preprocessing.load_stack import load_and_stack_rans


def compute_transport_term_scalars(tensor_list: List[torch.Tensor],
                                   input_feature_names: List[str],
                                   reduction: str,
                                   case_names: List[str],
                                   data_path: str) -> List[torch.Tensor]:
    """
    Compute transport scalars T = ui uj ∂ui/∂xj, using raw gradient data if needed.
    """
    scalars = []

    need_gradients = not all(g in input_feature_names for g in ['dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy'])
    transport_features = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']

    if need_gradients:
        x_stack, _, _ = load_and_stack_rans(case_names, transport_features, [], data_path)

    for i, tensor in enumerate(tensor_list):
        if need_gradients:
            t = x_stack[i]
        else:
            vel_idx = get_feature_indices(['Ux', 'Uy'], input_feature_names)
            grad_idx = get_feature_indices(['dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy'], input_feature_names)
            t = torch.cat([tensor[:, vel_idx], tensor[:, grad_idx]], dim=1)

        u = t[:, :2]
        du = t[:, 2:].view(-1, 2, 2)
        u_vec = u.unsqueeze(-1)
        u_T = u.unsqueeze(1)
        triple_product = torch.bmm(u_T, torch.bmm(du, u_vec)).squeeze()

        abs_T = torch.abs(triple_product)
        if reduction == 'local':
            scalar = abs_T
        elif reduction == 'mean':
            scalar = torch.mean(abs_T).unsqueeze(0)
        elif reduction == 'max':
            scalar = torch.max(abs_T).unsqueeze(0)
        else:
            raise ValueError(f"[ERROR] Unknown reduction {reduction} for transport term.")

        scalars.append(scalar)

    return scalars



def apply_transport_term_normalization(tensor_list: List[torch.Tensor],
                                       input_feature_names: List[str],
                                       scalars: List[torch.Tensor],
                                       reduction: str='mean') -> List[torch.Tensor]:

    """
    Further scale the normalized velocity and gradient fields by the transport scalar T.
    """
    vel_idx = get_feature_indices(['Ux', 'Uy'], input_feature_names)
    grad_idx = get_feature_indices(['dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy'], input_feature_names)
    all_indices = vel_idx + grad_idx

    normed_list = []
    if reduction != 'local':
        for tensor, scalar in zip(tensor_list, scalars):
            t = tensor.clone()
            t[:, all_indices] = t[:, all_indices] / (scalar + 1e-8)
            normed_list.append(t)
        return normed_list
    elif reduction == 'local':
        for tensor, point_scalars in zip(tensor_list, scalars):
            t = tensor.clone()
            # point_scalars is shape (N,)
            for idx in all_indices:
                t[:, idx] = t[:, idx] / (point_scalars + 1e-8)
            normed_list.append(t)

        return normed_list
    return None



def apply_feature_norms(tensor_list: List[torch.Tensor],
                        input_feature_names: List[str],
                        feature_norm_key: str,
                        case_names: List[str],
                        grid_dicts: Dict[str, Dict[str, torch.Tensor]],
                        transport_norm: bool = False,
                        transport_reduction: str = '',
                        external_scalars: dict = None):
    """
    Apply feature normalization and optionally transport scaling.

    If external_scalars is provided:
        - 'U_L_scalars' = list of (U_ref, L_ref)
        - 'transport_scalars' = list of transport scalars
    """
    if feature_norm_key == '':
        return tensor_list, {}

    if external_scalars is not None:
        # Use provided (U_ref, L_ref) scalars
        normed_list, nondim_scalars = apply_local_nondim(tensor_list, input_feature_names, external_scalars['U_L_scalars'])

        # Optionally apply transport normalization with passed scalars
        if transport_norm and 'transport_scalars' in external_scalars:
            normed_list = apply_transport_term_normalization(normed_list, input_feature_names, external_scalars['transport_scalars'])

        return normed_list, external_scalars

    # Compute and apply scalars from scratch
    if feature_norm_key == 'nondim_local':
        scalars = get_local_nondimensional_scalars(tensor_list, input_feature_names, grid_dicts, case_names)
        normed_list, used_scalars = apply_local_nondim(tensor_list, input_feature_names, scalars)

        if transport_norm:
            print(f'[Debug] transport_norm: {transport_reduction}')
            transport_scalars = compute_transport_term_scalars(normed_list, input_feature_names, reduction=transport_reduction)
            normed_list = apply_transport_term_normalization(normed_list, input_feature_names, transport_scalars, reduction=transport_reduction)
            return normed_list, {'U_L_scalars': used_scalars, 'transport_scalars': transport_scalars}

        return normed_list, {'nondim_scalars': used_scalars}

    else:
        raise ValueError(f"[ERROR] Unknown feature_norm method '{feature_norm_key}'")
