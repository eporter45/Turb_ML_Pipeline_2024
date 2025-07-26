import matplotlib.pyplot as plt
import os
import numpy as np
from matplotlib.colors import SymLogNorm
from utils.convert_2D import to_grid

# === Utility Functions ===
def compute_cell_edges(cx, cy):
    cx_edges = np.zeros((cx.shape[0] + 1, cx.shape[1] + 1))
    cy_edges = np.zeros((cy.shape[0] + 1, cy.shape[1] + 1))

    cx_edges[:-1, :-1] = cx
    cy_edges[:-1, :-1] = cy

    cx_edges[-1, :-1] = cx[-1, :]
    cx_edges[:, -1] = cx_edges[:, -2]

    cy_edges[:-1, -1] = cy[:, -1]
    cy_edges[-1, :] = cy_edges[-2, :]
    return cx_edges, cy_edges


def plot_single_figure(grid_dict, feature_1d, feature_name, case_name):
    """
    Plot a single scalar field over a 2D grid using pcolormesh, consistent with plot_case_prediction.

    Inputs:
        grid_dict: dict containing 'Cx', 'Cy', and 'case_name'
        feature_1d: 1D array or Series (flattened)
        feature_name: string to label plot
        case_name: name of the case (e.g., 'BUMP_h20')
    """
    from utils.convert_2D import to_grid

    # Convert 1D to 2D grid using case-specific reshaping
    cx = to_grid(grid_dict['Cx'], case_name)
    cy = to_grid(grid_dict['Cy'], case_name)
    feature_2d = to_grid(feature_1d, case_name)

    # Compute cell edges for pcolormesh
    cx_edges, cy_edges = compute_cell_edges(cx, cy)

    # Mask zeros and normalize color scale symmetrically around 0
    feature_masked = np.ma.masked_where(feature_2d == 0, feature_2d)
    abs_max = np.nanmax(np.abs(feature_masked))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad(color='lightgreen')

    pc = ax.pcolormesh(cx_edges, cy_edges, feature_masked, cmap=cmap, norm=norm, shading='auto')
    ax.set_title(f'{case_name}: {feature_name}')
    fig.colorbar(pc, ax=ax)
    ax.axis('off')

    return fig, ax


def plot_single_figure_log(grid_dict, feature_1d, feature_name, case_name, base=10, eps=1e-12):
    """
    Plot a single feature on a symmetric logarithmic color scale.
    Intended for visualizing fields like gradient components with sharp spikes.

    Inputs:
        grid_dict   : dict with 'Cx', 'Cy', and 'case_name'
        feature_1d  : 1D array or Series of flattened values
        feature_name: label for the colorbar and title
        case_name   : name of the case (e.g., 'BUMP_h20')
        eps         : small epsilon to avoid log(0); default 1e-12
    """
    from utils.convert_2D import to_grid
    # Convert inputs to 2D
    cx = to_grid(grid_dict['Cx'], case_name)
    cy = to_grid(grid_dict['Cy'], case_name)
    feature_2d = to_grid(feature_1d, case_name)

    # Compute mesh cell edges
    cx_edges, cy_edges = compute_cell_edges(cx, cy)

    # Mask zeros
    feature_masked = np.ma.masked_where(np.abs(feature_2d) < eps, feature_2d)

    # Log-scale normalization
    vmax = np.nanmax(np.abs(feature_masked))
    linthresh = vmax * 0.01
    norm = SymLogNorm(linthresh=linthresh, vmin=-vmax, vmax=vmax, base=base)

    # Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad(color='lightgreen')

    pc = ax.pcolormesh(cx_edges, cy_edges, feature_masked, cmap=cmap, norm=norm, shading='auto')
    ax.set_title(f'{case_name}: {feature_name} (log scale)')
    fig.colorbar(pc, ax=ax)
    ax.axis('off')

    return fig, ax

def plot_case_prediction(truth, pred, save_dir, case_name, vmin_shared=True, grid_dict=None):
    if grid_dict is None:
        raise ValueError("grid_dict must be provided to plot_case_prediction")

    # Convert 1D flattened grid to 2D grid
    from utils.convert_2D import to_grid

    cx = to_grid(grid_dict['Cx'], case_name)
    cy = to_grid(grid_dict['Cy'], case_name)
    cx_edges, cy_edges = compute_cell_edges(cx, cy)

    # Convert and reshape pressure fields to 2D grids
    print(f'[DEBUG]: Plotting case prediction: pred shape: {pred.shape}')
    print(f'[DEBUG]: Plotting case prediction: truth shape: {truth.shape}')
    truth_2d = to_grid(truth, case_name)
    pred_2d = to_grid(pred, case_name)
    diff_2d = np.abs(truth_2d - pred_2d)

    vmin = min(truth_2d.min(), pred_2d.min()) if vmin_shared else None
    vmax = max(truth_2d.max(), pred_2d.max()) if vmin_shared else None

    fig, axs = plt.subplots(1, 3, figsize=(30, 5))
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='white')

    im0 = axs[0].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(truth_2d == 0, truth_2d),
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    axs[0].set_title("Truth")
    fig.colorbar(im0, ax=axs[0])

    im1 = axs[1].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(pred_2d == 0, pred_2d),
                            cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    axs[1].set_title("Prediction")
    fig.colorbar(im1, ax=axs[1])

    im2 = axs[2].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(diff_2d == 0, diff_2d),
                            cmap=cmap, shading='auto')
    axs[2].set_title("Abs Difference")
    fig.colorbar(im2, ax=axs[2])

    for ax in axs:
        ax.axis('off')

    os.makedirs(save_dir, exist_ok=True)
    fname = f"{case_name}_{'shared' if vmin_shared else 'independent'}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_case_prediction_multi(truth, pred, save_dir, case_name, grid_dict, feature_labels, vmin_shared=True):
    if grid_dict is None:
        raise ValueError("grid_dict must be provided to plot_case_prediction_multi")

    if truth.shape[1] != pred.shape[1]:
        raise ValueError("Mismatch in number of prediction targets between truth and pred")

    num_features = truth.shape[1]
    cx = to_grid(grid_dict['Cx'], case_name)
    cy = to_grid(grid_dict['Cy'], case_name)
    cx_edges, cy_edges = compute_cell_edges(cx, cy)

    fig, axs = plt.subplots(num_features, 3, figsize=(5 * num_features, 5 * num_features))
    axs = np.atleast_2d(axs)
    print(f'[DEBUG]: Plotting case prediction: pred shape: {pred.shape}')
    cmap = plt.get_cmap('jet').copy()
    cmap.set_bad(color='black')

    for i in range(num_features):
        truth_2d = to_grid(truth[:, i], case_name)
        pred_2d = to_grid(pred[:, i], case_name)
        diff_2d = np.abs(truth_2d - pred_2d)

        vmin = min(truth_2d.min(), pred_2d.min()) if vmin_shared else None
        vmax = max(truth_2d.max(), pred_2d.max()) if vmin_shared else None

        im0 = axs[i, 0].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(truth_2d == 0, truth_2d),
                                   cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        axs[i, 0].set_title(f"{feature_labels[i]} Truth")
        fig.colorbar(im0, ax=axs[i, 0])

        im1 = axs[i, 1].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(pred_2d == 0, pred_2d),
                                   cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
        axs[i, 1].set_title(f"{feature_labels[i]} Prediction")
        fig.colorbar(im1, ax=axs[i, 1])

        im2 = axs[i, 2].pcolormesh(cx_edges, cy_edges, np.ma.masked_where(diff_2d == 0, diff_2d),
                                   cmap=cmap, shading='auto')
        axs[i, 2].set_title(f"{feature_labels[i]} Abs Difference")
        fig.colorbar(im2, ax=axs[i, 2])

        for j in range(3):
            axs[i, j].axis('off')

    os.makedirs(save_dir, exist_ok=True)
    fname = f"{case_name}_{'shared' if vmin_shared else 'independent'}.png"
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, fname), dpi=150)
    plt.close()


def plot_all_cases(y_true_list, y_pred_list, out_features, case_names, save_dir, grid_dicts):
    for y_t, y_p, name in zip(y_true_list, y_pred_list, case_names):
        #print("Test cases:", case_names)
        #print("Grid keys:", list(grid_dicts.keys()))
        plot_case_prediction_multi(y_t, y_p, save_dir, name, feature_labels= out_features, vmin_shared=True, grid_dict=grid_dicts[name])
        plot_case_prediction_multi(y_t, y_p, save_dir, name, feature_labels = out_features, vmin_shared=False, grid_dict=grid_dicts[name])



