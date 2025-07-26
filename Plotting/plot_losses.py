import matplotlib.pyplot as plt
import numpy as np
import torch
def plot_losses(losses, title, trial_name='Train',type=''):
    """
    Plots log(loss) vs. epoch for training or validation.

    Args:
        losses (list or array): Loss values per epoch.
        title (str): Title or name of the run (e.g., from config).
        testortrain (str): 'Train' or 'Test' to label the curve.

    Returns:
        fig (matplotlib.figure.Figure): The figure object for saving or displaying.
    """
    epochs = list(range(1, len(losses) + 1))
    losses = np.array([t.detach().cpu().item() if torch.is_tensor(t) else t for t in losses])
    print(f'[Debug] trial name: {trial_name} title: {title}')
    fig = plt.figure(figsize=(8, 6))
    plt.plot(epochs, losses, linewidth=2.0)
    plt.yscale('log')
    plt.title(f'{trial_name} Loss vs. Epoch - {title} {type}')
    plt.xlabel("Epochs")
    plt.ylabel("Log Loss")
    plt.grid(True, which="both", linestyle='--', linewidth=0.5)

    return fig

import matplotlib.pyplot as plt
import os

def get_total_loss_df(loss_df):
    total_dict = {'total_net_loss': loss_df['total_loss'],
                  'total_phys_loss': loss_df['total_phys'],
                  'total_data_loss': loss_df['total_data']}
    return total_dict




def plot_all_case_losses(losses_df, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)   # keep as a Path or str
    total_dict = get_total_loss_df(losses_df)
    loss_dicts = [total_dict, losses_df['case_net']]
    dict_names = ['total_net_loss', 'case_net_loss']
    if config['training']['loss_terms']['use_phys_loss']== 'true':
        allowed = ['case_phys_loss', 'case_data_loss']
        dict_names = dict_names + allowed
        allowed_losses = [losses_df['case_phys'], losses_df['case_data']]
        loss_dicts = loss_dicts + allowed_losses
    for loss_dict, dict_name in zip(loss_dicts, dict_names):
        if dict_name != 'total_net_loss':
            cur_dir = os.path.join(output_dir, dict_name)
            os.makedirs(cur_dir, exist_ok=True)
        else:
            cur_dir = output_dir
        for case_name, loss_values in loss_dict.items():
            fig = plot_losses(losses=loss_values,
                              title= case_name,
                              trial_name=config['trial_name'],
                              type=dict_name)
            fig_path = os.path.join(cur_dir, f'{case_name}.png')
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"[INFO] Saved loss plot for {case_name} to {fig_path}")



