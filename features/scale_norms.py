import torch
from typing import List, Dict, Tuple

def normalize_abs(tensor_list: List[torch.Tensor], device=None) -> Tuple[List[torch.Tensor], List[Dict[int, float]]]:
    device = device or tensor_list[0].device
    normed, scalers = [], []
    for t in tensor_list:
        t = t.to(device)
        max_vals = torch.max(torch.abs(t), dim=0).values
        max_vals[max_vals == 0] = 1.0
        normed.append(t / max_vals)
        scalers.append({i: float(max_vals[i]) for i in range(t.shape[1])})
    return normed, scalers

def normalize_minmax(tensor_list: List[torch.Tensor], device=None) -> Tuple[List[torch.Tensor], List[Dict[int, Tuple[float, float]]]]:
    device = device or tensor_list[0].device
    normed, scalers = [], []
    for t in tensor_list:
        t = t.to(device)
        min_vals = torch.min(t, dim=0).values
        max_vals = torch.max(t, dim=0).values
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1.0
        normed.append((t - min_vals) / range_vals)
        scalers.append({i: (float(min_vals[i]), float(range_vals[i])) for i in range(t.shape[1])})
    return normed, scalers

def apply_scale_norm(x_train_list: List[torch.Tensor],
                     x_test_list: List[torch.Tensor],
                     mode: str = "abs",
                     device=None,
                     external_scalars: Dict[str, List[Dict]] = None):
    """
    Supports 'abs' and 'minmax'. Returns:
    - normalized train tensors
    - normalized test tensors
    - scalers: {'train': [...], 'test': [...]}
    """
    device = device or (x_train_list[0].device if x_train_list else x_test_list[0].device)

    # ─── Test-only normalization ─────────────────────────
    if not x_train_list and external_scalars:
        x_test_normed = []
        test_scalers = external_scalars.get("test", [])
        for t, s in zip(x_test_list, test_scalers):
            t = t.to(device)
            if mode == "abs":
                scale = torch.tensor([s[i] for i in range(len(s))], device=device)
                x_test_normed.append(t / scale)
            elif mode == "minmax":
                min_vals = torch.tensor([s[i][0] for i in range(len(s))], device=device)
                range_vals = torch.tensor([s[i][1] for i in range(len(s))], device=device)
                x_test_normed.append((t - min_vals) / (range_vals + 1e-8))
            else:
                raise ValueError(f"[ERROR] Unknown scale mode: {mode}")
        return x_test_normed, None, {"test": test_scalers}

    # ─── Full normalization ──────────────────────────────
    if mode == "abs":
        x_train_normed, train_scalers = normalize_abs(x_train_list, device)
        x_test_normed, test_scalers = normalize_abs(x_test_list, device)
    elif mode == "minmax":
        x_train_normed, train_scalers = normalize_minmax(x_train_list, device)
        x_test_normed, test_scalers = normalize_minmax(x_test_list, device)
    else:
        raise ValueError(f"[ERROR] Unknown scale mode: {mode}")

    scalers = {"train": train_scalers, "test": test_scalers}
    return x_train_normed, x_test_normed, scalers

