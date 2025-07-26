import torch
import os
import numpy as np
import yaml
from Preprocessing.Load_data import load_case_data, get_case_features
from Preprocessing.load_data_pressure import load_data_pressure_from_config



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Config ===
cases = ['BUMP_h20', 'BUMP_h26', 'PHLL_case_1p0', 'PHLL_case_0p8']  # Test mixed geometry too
out_features = ['p']
in_features = ['Ux', 'Uy']
norm_mode = 'minmax'  # options: 'abs' or 'minmax'
loss_mode = 'tke'

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

from training_utils.weight_case_losses import build_case_loss_weights

loss_weights = build_case_loss_weights(y_tensor_list, device)

print(f'Loss weights shape: {loss_weights.shape}')
print(f'Loss weights: {[ch.item() for ch in loss_weights]}')