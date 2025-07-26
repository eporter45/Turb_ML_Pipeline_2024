import os
import matplotlib.pyplot as plt
import numpy as np
import torch
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
    if config['features'].get('transport_feature', False):
        input_feature_names.append('T_transport')

    feature_norm = config['training'].get('feature_norm', '')
    transport_norm = config['features'].get('transport_norm', '')
    scale_norm = config['training'].get('scale_norm', '')
    print(f'Length ofinput features: {x_train_normed[0].shape}')
    for feat_idx, feat_name in enumerate(input_feature_names):
        plt.figure()
        for case, xnorm in zip(train_cases, x_train_normed):
            print(f"feat_idx: {feat_idx}, feat_name: {feat_name} case: {case}")
            vals = xnorm[:, feat_idx].cpu().numpy()
            sns.kdeplot(vals, label=f"{case} (train)", linestyle='-')

        if x_test_normed and len(x_test_normed) == len(test_cases):
            for case, xnorm in zip(test_cases, x_test_normed):
                vals = xnorm[:, feat_idx].cpu().numpy()
                sns.kdeplot(vals, label=f"{case} (test)", linestyle='--')
        if feat_name == 'T_transport':
            plt.title(f"{feat_name} KDE\n:, "
                      f"Transport: {feature_norm + transport_norm}, Scale: {scale_norm}")
        elif feat_name in input_feature_names:
            plt.title(f"{feat_name} KDE\nFeatureNorm: {feature_norm}, ")
        plt.xlabel(f"{feat_name} (normalized)")
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"KDE_{feat_name}.png")
        plt.savefig(save_path)
        print(f"[INFO] Saved plot: {save_path}")
        plt.close()
        gc.collect()
