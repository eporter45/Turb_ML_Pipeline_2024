import torch

def frobenius_error_single(y_pred, y_true):
    """
    Compute Frobenius norm error between two tensors of shape:
        - (n_points, m) or (n_points, m, m)
    """
    if y_pred.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")

    if y_pred.ndim == 2:
        diff = y_pred - y_true
        error = torch.norm(diff, dim=1)
    elif y_pred.ndim == 3:
        diff = y_pred - y_true
        error = torch.norm(diff, dim=(1, 2))
    else:
        raise ValueError(f"Unsupported tensor shape: {y_pred.shape}")

    return error.mean()

def frobenius_error(y_pred_list, y_true_list):
    """
    Compute the mean Frobenius error across a list of tensors.

    Args:
        y_pred_list: list of tensors or a single tensor
        y_true_list: list of tensors or a single tensor

    Returns:
        float: average Frobenius error across all tensors
    """
    if isinstance(y_pred_list, torch.Tensor) and isinstance(y_true_list, torch.Tensor):
        return frobenius_error_single(y_pred_list, y_true_list).item()

    if len(y_pred_list) != len(y_true_list):
        raise ValueError("Prediction and truth lists must have the same length")

    total_error = 0.0
    for yp, yt in zip(y_pred_list, y_true_list):
        total_error += frobenius_error_single(yp, yt)

    return (total_error / len(y_pred_list)).item()
