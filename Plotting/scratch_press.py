import matplotlib.pyplot as plt
import os
import numpy as np

from Preprocessing.Load_data import load_case_data

# === Config ===
case_name = "BUMP_h42"
feature = "p"

# Dynamically find correct data path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(root_dir, "Data")

# === Load data ===
print(f"[INFO] Loading case: {case_name} from {data_path}")
rans_dict, _, _ = load_case_data(case_name, data_path, mesh_yn='yes')

# === Extract the pressure feature ===
if feature not in rans_dict:
    raise KeyError(f"Feature '{feature}' not found in RANS dictionary for case '{case_name}'")

truth_tensor = rans_dict[feature]
pred_tensor = 1.01 * truth_tensor

# === Prepare grid ===
from utils.convert_2D import convert2d_case

x = convert2d_case(rans_dict['Cx'], case_name)
y = convert2d_case(rans_dict['Cy'], case_name)
if 'DUCT' in case_name:
    x = y
    y = convert2d_case(rans_dict['Cz'], case_name)

# === Plotting Utility Function ===
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


def get_bump_edge_coords(grid_tuple, case):
    import pyvista as pv
    cx, cy = grid_tuple
    Z = np.zeros_like(cx)  # dummy z-coord for 2D grid
    grid = pv.StructuredGrid(cx, cy, Z)
    edges = grid.extract_feature_edges(boundary_edges=True)
    edge_coords = edges.points
    x_edges = edge_coords[:, 0]
    y_edges = edge_coords[:, 1]
    return x_edges, y_edges

def plot_single_figure(grid_tuple, feature, feature_name, case):
    cx, cy = grid_tuple
    cx_edges, cy_edges = compute_cell_edges(cx, cy)

    # Apply mask to handle zero values as white
    feature_masked = np.ma.masked_where(feature == 0, feature)

    # Compute normalization
    abs_max = np.nanmax(np.abs(feature))
    norm = plt.Normalize(vmin=-abs_max, vmax=abs_max)

    # Set up colormap with masked values set to white
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad(color='lightgreen')

    fig, ax = plt.subplots(figsize=(14, 6))
    pc = ax.pcolormesh(cx_edges, cy_edges, feature_masked, cmap=cmap, norm=norm, shading='auto')

    ax.set_title(f'{case}: {feature_name}')
    fig.colorbar(pc, ax=ax)

    return fig, ax

# === Plot a single feature ===
from utils.convert_2D import convert2d_case

feature_grid = convert2d_case(truth_tensor, case_name)
fig, ax = plot_single_figure((x, y), feature_grid, feature_name=feature, case=case_name)
plt.show()
