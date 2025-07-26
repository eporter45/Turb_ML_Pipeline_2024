from features.scale_norms import *
from features.nondim_norms import *

def apply_two_step_normalization(x_tensor_list: List[torch.Tensor],
                                  config,
                                  input_feature_names,
                                  case_names=None,
                                  grid_dicts=None,
                                  device=None):
    """
    Two-step normalization pipeline:
    1. Apply feature norm (e.g., nondimensionalization)
    2. Apply scale norm (e.g., abs_abs, rel_minmax)

    Returns:
        x_tensor_list_normalized
        scalers = {
            'feature': ...,  # nondim scalars
            'scale': ...     # scale normalization scalars
        }
    """
    device = device or x_tensor_list[0].device

    feature_norm_key = config['training'].get('feature_norm', '')
    scale_norm_key = config['training'].get('scale_norm', '')

    # Step 1: Feature Norm
    x_feat_normed, feature_scalers = apply_feature_norms(
        tensor_list=x_tensor_list,
        input_feature_names=input_feature_names,
        feature_norm_key=feature_norm_key,
        case_names=case_names,
        grid_dicts=grid_dicts,
    )

    # Step 2: Scale Norm
    if scale_norm_key == '':
        x_final_normed = x_feat_normed
        scale_scalers_train = None
    else:
        x_final_normed, _, scale_scalers_train = apply_scale_norm(
            x_train_list=x_feat_normed,
            x_test_list=[],
            mode=scale_norm_key,
            device=device
        )

    scalers = {
        'feature': feature_scalers.get('U_L_scalars') if feature_scalers else None,
        'scale': {
            'train': scale_scalers_train,
            'test': None
        }
    }

    return x_final_normed, scalers


def normalize_x_test(x_tensor_list,
                     config,
                     input_feature_names,
                     case_names=None,
                     grid_dicts=None,
                     device=None,
                     scalers=None):
    device = device or x_tensor_list[0].device

    feature_norm_key = config['training']['feature_norm']
    scale_norm_key = config['training']['scale_norm']

    # Decide whether to reuse scalars
    use_feature_scaler = (
        'global' in feature_norm_key and scalers and scalers.get('feature') is not None
    )
    use_scale_scaler = (
        scalers and isinstance(scalers.get('scale'), dict) and
        all(case in scalers['scale'] for case in case_names)
    )

    # Step 1: Feature Norm
    x_feat_normed, _ = apply_feature_norms(
        tensor_list=x_tensor_list,
        input_feature_names=input_feature_names,
        feature_norm_key=feature_norm_key,
        case_names=case_names,
        grid_dicts=grid_dicts,
        external_scalars={'U_L_scalars': scalers['feature']} if use_feature_scaler else None
    )

    # Step 2: Scale Norm
    if scale_norm_key == '':
        x_final_normed = x_feat_normed
        scale_scalers_test = None
    else:
        _, x_final_normed, scale_scalers_test = apply_scale_norm(
            x_train_list=[],
            x_test_list=x_feat_normed,
            mode=scale_norm_key,
            device=device,
            external_scalars=scalers.get('scale', {}).get('test') if use_scale_scaler else None
        )

    if scalers is not None and isinstance(scalers.get('scale'), dict):
        scalers['scale']['test'] = scale_scalers_test

    return x_final_normed, x_feat_normed, x_tensor_list



