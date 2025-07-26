# this file was written by n_kozak
# it converts case data from a vector to a 2D matrix for ease of plotting
import torch
import numpy as np

def convert2d_BUMP(Var):
    ## ------------------------------------------------------------------------  ##
    " Load Packages "
    import numpy as np
    ## ------------------------------------------------------------------------  ##
    ny = 175
    var2d = Var.to_numpy()
    Var1 = np.reshape(var2d[0:ny * 40], (ny, -1), order='C')
    Var2 = np.reshape(var2d[ny * 40:(40 + 12) * ny], (ny, -1), order='C')
    Var3 = np.reshape(var2d[(40 + 12) * ny:(40 + 12 + 200) * ny], (ny, -1), order='C')
    Var4 = np.reshape(var2d[(40 + 12 + 200) * ny:(40 + 12 + 200 + 40) * ny], (ny, -1), order='C')
    Var5 = np.reshape(var2d[(40 + 12 + 200 + 40) * ny::], (ny, -1), order='C')
    var2d = np.hstack([Var1, Var2, Var3, Var4, Var5])

    return var2d

def convert2d_PHIL(Var):
## ------------------------------------------------------------------------  ##
    " Load Packages "
    import numpy as np
    import pandas as pd
## ------------------------------------------------------------------------  ##
    ny = 99
    var2d = Var.to_numpy()
    var2d = np.transpose(np.reshape(var2d ,(ny,-1),order='F'))
    return var2d


def convert2d_DUCT(Var):
    ## ------------------------------------------------------------------------  ##
    " Load Packages "
    import numpy as np
    ## ------------------------------------------------------------------------  ##
    ny = 48

    var2d = Var.to_numpy()
    s_i = (48 ** 2) * 0
    e_i = (48 ** 2) * (0 + 1)
    Var_1 = np.reshape(var2d[s_i:e_i], (ny, -1), order='F')
    s_i = (48 ** 2) * 1
    e_i = (48 ** 2) * (1 + 1)
    Var_region = np.reshape(var2d[s_i:e_i], (ny, -1), order='F')
    Var_1 = np.fliplr(np.vstack([Var_1, Var_region]))

    s_i = (48 ** 2) * 2
    e_i = (48 ** 2) * (2 + 1)
    Var_2 = np.reshape(var2d[s_i:e_i], (ny, -1), order='F')
    s_i = (48 ** 2) * 3
    e_i = (48 ** 2) * (3 + 1)
    Var_region = np.reshape(var2d[s_i:e_i], (ny, -1), order='F')
    Var_2 = np.fliplr(np.vstack([Var_2, Var_region]))

    var2d = np.hstack([Var_1, Var_2])

    return var2d

def convert2d_CNDV(Var):
## ------------------------------------------------------------------------  ##
    " Load Packages "
    import numpy as np
## ------------------------------------------------------------------------  ##
    ny = 175
    var2d = Var.to_numpy()
    Var1 = np.reshape(var2d[0:50*ny] ,(ny,-1),order='C')
    Var2 = np.reshape(var2d[50*ny:450*ny] ,(ny,-1),order='C')
    Var3 = np.reshape(var2d[450*ny::] ,(ny,-1),order='C')
    var2d = np.hstack([Var1,Var2,Var3])
    return var2d


def convert2d_case(Var, test_case):
    if 'bump' in test_case.lower():
        return convert2d_BUMP(Var)
    elif 'phll' in test_case.lower():
        return convert2d_PHIL(Var)
    elif 'duct' in test_case.lower():
        return convert2d_DUCT( Var)
    elif 'cndv' in test_case.lower():
        return convert2d_CNDV(Var)

def convert2d_cases(Var, test_cases):
    cases_data = []
    for case in test_cases:
        cases_data.append(convert2d_case(Var, case))
    return cases_data

# this function was written by
def to_grid(data, case_name):
    """
    Converts tensor or NumPy data to a 2D grid using case-specific reshaping.
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy().flatten()
    elif isinstance(data, np.ndarray):
        data = data.flatten()
    import pandas as pd
    return convert2d_case(pd.Series(data), case_name)


#==============Start flattening back to vector funcitons==========
def bump_to_vector(grid2d):
    ny = 175
    widths = [40, 12, 200, 40, 120]  # horizontal widths of each region
    slices = []
    start = 0
    for w in widths:
        end = start + w
        block = grid2d[:, start:end]
        slices.append(block.flatten(order='C'))
        start = end
    return np.concatenate(slices)

def phll_to_vector(grid2d):
    return grid2d.T.flatten(order='F')

def duct_to_vector(grid2d):
    ny = 48
    nx_total = grid2d.shape[1]
    nx_half = nx_total // 2

    block1 = np.fliplr(grid2d[:, :nx_half])
    block2 = np.fliplr(grid2d[:, nx_half:])

    block1_main = block1[:ny, :]
    block1_region = block1[ny:, :]
    block2_main = block2[:ny, :]
    block2_region = block2[ny:, :]

    flat1 = block1_main.flatten(order='F')
    flat1_region = block1_region.flatten(order='F')
    flat2 = block2_main.flatten(order='F')
    flat2_region = block2_region.flatten(order='F')

    return np.concatenate([flat1, flat1_region, flat2, flat2_region])

def cndv_to_vector(grid2d):
    ny = 175
    widths = [50, 400, 50]
    slices = []
    start = 0
    for w in widths:
        end = start + w
        block = grid2d[:, start:end]
        slices.append(block.flatten(order='C'))
        start = end
    return np.concatenate(slices)

def to_vector(grid2d, case_name):
    """
    Converts a 2D grid back into a 1D vector matching the original RANS data order.
    Uses geometry-specific logic.
    """
    geom = case_name.split('_')[0].lower()

    if geom == 'bump':
        return bump_to_vector(grid2d)
    elif geom == 'phll':
        return phll_to_vector(grid2d)
    elif geom == 'duct':
        return duct_to_vector(grid2d)
    elif geom == 'cndv':
        return cndv_to_vector(grid2d)
    else:
        raise ValueError(f"[ERROR] Unrecognized case geometry: {geom}")

