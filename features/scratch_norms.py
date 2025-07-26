'''This script is used to ensure that make_norms functions are working correctly and are interpretable'''

import os
import numpy as np
import torch

from Preprocessing.Load_data import load_case_data, get_case_features
from features.scale_norms import normalize_xy, auto_denormalize_xy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
cases = ['BUMP_h20', 'BUMP_h26', 'PHLL_case_1p0', 'PHLL_case_0p8']  # Test mixed geometry too
out_features = ['p']
in_features = ['Ux', 'Uy']
norm_mode = 'minmax'  # options: 'abs' or 'minmax'

# === Dynamically find Data folder ===
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")

# === Load data ===
x_tensor_list = []
y_tensor_list = []  # Dummy targets just for scratchwork (can reuse x for now)

for case in cases:
    print(f"[INFO] Loading case: {case} from {data_path}")
    rans_dict, les = load_case_data(case, data_path, mesh_yn='no')

    X = torch.tensor(
        np.vstack([get_case_features(rans_dict, feat) for feat in in_features]).T,
        dtype=torch.float32
    )
    Y = torch.tensor(np.vstack([get_case_features(rans_dict, feat) for feat in out_features]).T, dtype=torch.float32)

    print(f'Case {case}, X_tensor shape: {X.shape}')
    x_tensor_list.append(X)
    y_tensor_list.append(Y)

print(f"\n[INFO] Loaded {len(x_tensor_list)} tensors.")
for i, tensor in enumerate(x_tensor_list):
    print(f"  Case {cases[i]} tensor shape: {tensor.shape}")

# === Normalize ===
print("\n=== Normalizing X and Y tensors ===")
norm_x_train, norm_x_test, x_scalers, norm_y_train, norm_y_test, y_scalers = normalize_xy(
    x_train_list=x_tensor_list,
    x_test_list=x_tensor_list,
    y_train_list=y_tensor_list,
    y_test_list=y_tensor_list,
    train_case_names=cases,
    test_case_names=cases,
    norm_mode=norm_mode,
    device=device
)

# === Denormalize ===
print("\n=== Denormalizing X and Y tensors ===")
denorm_x_train, denorm_y_train = auto_denormalize_xy(
    normed_x_list=norm_x_train,
    normed_y_list=norm_y_train,
    x_scalers=x_scalers,
    y_scalers=y_scalers,
    train_case_names=cases,
    config={'training': {'norm': norm_mode}},
    device=device
)

# === Check if Original == Denormalized ===
print("=== Checking if original tensors match denormalized tensors ===")
for i, (orig_x, orig_y) in enumerate(zip(x_tensor_list, y_tensor_list)):
    x_match = torch.allclose(orig_x, denorm_x_train[i], rtol=1e-4, atol=1e-6)
    y_match = torch.allclose(orig_y, denorm_y_train[i], rtol=1e-4, atol=1e-6)
    print(f"  Case {cases[i]}: X match: {x_match}, Y match: {y_match}")
print("\n=== Final Summary ===")
print(f"X_scalers:\n{x_scalers}")
print(f"Y_scalers:\n{y_scalers}")
