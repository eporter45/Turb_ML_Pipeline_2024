import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import gc
import seaborn as sns

from Preprocessing.Load_data import load_case_data, get_case_features
from features.do_full_norms import apply_two_step_normalization, normalize_x_test
from features.add_transport import add_transport_feature
from Trials import TRIALS

# === Config ===
trial = 'bump_inter'
train_cases = TRIALS[trial]['train']
test_cases = TRIALS[trial]['test']

input_feature_names = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
output_feature_names = ['p']

norm_config = {
    'training': {
        'feature_norm': 'nondim_local',
        'transport_norm': '',
        'scale_norm': 'minmax',
        'transport_feature': True
    },
    'features': {
        'input': input_feature_names,
        'output': output_feature_names,
        'transport_feature': True}
}

device = torch.device("cpu")
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")
plot_dir = os.path.join(root_dir, "KDE_plots")
os.makedirs(plot_dir, exist_ok=True)

from Preprocessing.load_stack import load_and_stack_rans

# Load raw data
x_train, y_train, grid_train = load_and_stack_rans(train_cases, input_feature_names, output_feature_names, data_path)
x_test, y_test, grid_test = load_and_stack_rans(test_cases, input_feature_names, output_feature_names, data_path)

# Normalize features
x_train_normed, scalers = apply_two_step_normalization(
    x_train, config=norm_config,
    input_feature_names=input_feature_names,
    case_names=train_cases,
    grid_dicts=grid_train,
    device=device
)

x_test_normed, _, _ = normalize_x_test(
    x_test, config=norm_config,
    input_feature_names=input_feature_names,
    case_names=test_cases,
    grid_dicts=grid_test,
    device=device,
    scalers=scalers
)

# Optionally append transport feature
if norm_config['training']['transport_feature'] == True:
    print("[INFO] Appending transport feature...")
    x_train_normed = add_transport_feature(
        x_train_normed, train_cases, norm_config,
        grid_train, data_path, device
    )
    x_test_normed = add_transport_feature(
        x_test_normed, test_cases, norm_config,
        grid_test, data_path, device
    )
    input_feature_names.append("T_transport")  # Add to names for plotting

# === KDE Plot ===
print(f"[INFO] Plotting KDEs to {plot_dir}")
for feat_idx, feat_name in enumerate(input_feature_names):
    plt.figure()
    for case, xnorm in zip(train_cases, x_train_normed):
        vals = xnorm[:, feat_idx].numpy()
        sns.kdeplot(vals, label=f"{case} (train)", linestyle='-')

    if x_test_normed and len(x_test_normed) == len(test_cases):
        for case, xnorm in zip(test_cases, x_test_normed):
            vals = xnorm[:, feat_idx].numpy()
            sns.kdeplot(vals, label=f"{case} (test)", linestyle='--')

    plt.title(f"{feat_name} KDE\nFeatureNorm: {norm_config['training']['feature_norm']}, "
              f"Transport: {norm_config['training']['transport_norm']}, "
              f"Scale: {norm_config['training']['scale_norm']}")
    plt.xlabel(f"{feat_name} (normalized)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(plot_dir, f"KDE_{feat_name}.png")
    plt.savefig(save_path)
    print(f"[INFO] Saved plot: {save_path}")
    plt.close()
    gc.collect()

# === Final Info ===
print("\n[INFO] Scalars used:")
print("- Feature (U, L):", scalers.get('feature'))
print("- Transport T:", scalers.get('transport'))
print("- Scale:", scalers.get('scale'))

from Plotting.plot_pressure import plot_single_figure_log, plot_single_figure

T_idx = -1
for case, xnorm, grid in zip(test_cases, x_test_normed, grid_test.values()):
    T_abs = xnorm[:, T_idx].numpy()

    fig, ax = plot_single_figure_log(
        grid_dict=grid,
        feature_1d=T_abs,
        feature_name="T_transport (nondim, scaled)",
        case_name=case
    )

    save_path = os.path.join(plot_dir, f"TransportField_{case}.png")
    fig.savefig(save_path)
    print(f"[INFO] Saved transport field plot: {save_path}")
    plt.close(fig)
    gc.collect()

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import gc

def plot_input_kde(x_train_normed, x_test_normed, train_cases, test_cases, config, save_dir):
    """
    Plot KDE for each input feature from normalized x_train/x_test.

    Args:
        x_train_normed (list of tensors): Normalized training inputs (one per case).
        x_test_normed (list of tensors): Normalized test inputs (one per case).
        train_cases (list): Names of training cases.
        test_cases (list): Names of test cases.
        config (dict): Simulation config containing normalization settings and input features.
        save_dir (str): Directory where plots will be saved.
    """

    os.makedirs(save_dir, exist_ok=True)

    input_feature_names = config['features']['input'].copy()
    if config['training'].get('transport_feature'):
        input_feature_names.append('T_transport')

    feature_norm = config['training'].get('feature_norm', '')
    transport_norm = config['training'].get('transport_norm', '')
    scale_norm = config['training'].get('scale_norm', '')

    for feat_idx, feat_name in enumerate(input_feature_names):
        plt.figure()
        for case, xnorm in zip(train_cases, x_train_normed):
            vals = xnorm[:, feat_idx].cpu().numpy()
            sns.kdeplot(vals, label=f"{case} (train)", linestyle='-')

        if x_test_normed and len(x_test_normed) == len(test_cases):
            for case, xnorm in zip(test_cases, x_test_normed):
                vals = xnorm[:, feat_idx].cpu().numpy()
                sns.kdeplot(vals, label=f"{case} (test)", linestyle='--')

        plt.title(f"{feat_name} KDE\nFeatureNorm: {feature_norm}, "
                  f"Transport: {transport_norm}, Scale: {scale_norm}")
        plt.xlabel(f"{feat_name} (normalized)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"KDE_{feat_name}.png")
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")
        plt.close()
        gc.collect()
