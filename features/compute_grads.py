from utils.convert_2D import to_grid, to_vector

import numpy as np
import pyvista as pv
import torch

def compute_gradients_for_feature(feature_1d, grid_dict, feature_name):
    """
    Compute gradient of a scalar feature w.r.t. x, y, z directions.

    Inputs:
        feature_1d : 1D array or pandas Series
        grid_dict  : dict with keys 'Cx', 'Cy', and 'case_name'
        feature_name : str name used to build output dict keys

    Returns:
        Dictionary with keys:
            'd{feature_name}dx', 'd{feature_name}dy', 'd{feature_name}dz'
    """
    case_name = grid_dict['case_name']

    # 1. Convert coords to 2D
    x2d = to_grid(grid_dict['Cx'], case_name)
    y2d = to_grid(grid_dict['Cy'], case_name)
    z2d = np.zeros_like(x2d)

    # 2. Create PyVista mesh
    mesh = pv.StructuredGrid(x2d, y2d, z2d)

    # 3. Convert feature to 2D, then flatten
    field_2d = to_grid(feature_1d, case_name)
    mesh.point_data[feature_name] = field_2d.flatten(order='F')  # for PyVista

    # 4. Compute gradients
    mesh_with_grad = mesh.compute_derivative(scalars=feature_name)
    grad = mesh_with_grad['gradient']  # shape (n_points, 3)

    # 5. Sort back to RANS order using to_vector
    grad_dx = to_vector(grad[:, 0].reshape(field_2d.shape, order='F'), case_name)
    grad_dx = np.nan_to_num(grad_dx, nan=0.0, posinf=0.0, neginf=0.0)

    grad_dy = to_vector(grad[:, 1].reshape(field_2d.shape, order='F'), case_name)
    grad_dy = np.nan_to_num(grad_dy, nan=0.0, posinf=0.0, neginf=0.0)

    grad_dz = to_vector(grad[:, 2].reshape(field_2d.shape, order='F'), case_name)
    grad_dz = np.nan_to_num(grad_dz, nan=0.0, posinf=0.0, neginf=0.0)

    return {    f'd{feature_name}_dx': grad_dx,
                f'd{feature_name}_dy': grad_dy,
                f'd{feature_name}_dz': grad_dz  }




