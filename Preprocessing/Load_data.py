# import necessary packages
import os
import numpy as np
import torch

# initialize the directories we are working with
cwd = os.getcwd()
pwd = os.path.dirname(cwd)
data_path = os.path.join(pwd, 'data')

# load data for current case
def loadRANS_DICT_ext(case, path):
    filename = path + '/' + case + '_DICT'
    with open(f'{filename}.pkl', 'rb') as file:
        data = np.load(file, allow_pickle=True)
    return data

def load_case_data(case, path, mesh_yn):
    def make_dataPaths(data_filepath):
        LES_dict = data_filepath + '/LES_Dict'
        RANS = data_filepath + '/RANS'
        RANS_dict_ext = data_filepath + '/RANS_Dict_ext'
        RANS_mesh = data_filepath + '/RANS_MeshInfo'
        return LES_dict, RANS, RANS_dict_ext, RANS_mesh

    def loadLES_Dict(case, path):
        filename = path + '/' + case + '_DICT'
        with open(f'{filename}.pkl', 'rb') as file:
            data = np.load(file, allow_pickle=True)
        return data

    def loadRANS_mesh_info(case, path):
        filename = path + '/' + case + '_MeshInfo'
        with open(f'{filename}.pkl', 'rb') as file:
            data = np.load(file, allow_pickle=True)
        return data

    def loadRANS_DICT_ext(case, path):
        filename = path + '/' + case + '_DICT'
        with open(f'{filename}.pkl', 'rb') as file:
            data = np.load(file, allow_pickle=True)
        return data

    LES_d_path, RANS_path, RANS_dict_ext, RANS_MESH_PATH = make_dataPaths(path)
    LES_dict = loadLES_Dict(case, LES_d_path)
    #print(f"[DEBUG] LES keys for {case}: {list(LES_dict.keys())}")
    rans_dict = loadRANS_DICT_ext(case, RANS_dict_ext)
    if mesh_yn == 'yes':
        rans_mesh = loadRANS_mesh_info(case, RANS_MESH_PATH)
        return rans_dict, LES_dict, rans_mesh
    else:
        return rans_dict, LES_dict

def get_case_features(RANS, feature):
    return RANS[feature].to_numpy()

def getReynoldsFeatures(LES_dict):
    tau_xx = LES_dict['uu']
    tau_xy = LES_dict['uv']
    tau_xz = LES_dict['uw']
    tau_yy = LES_dict['vv']
    tau_yz = LES_dict['vw']
    tau_zz = LES_dict['ww']
    return [tau_xx, tau_xy, tau_xz, tau_yy, tau_yz, tau_zz]

def load_ReynoldsTensor(LES):
    tau_vec = getReynoldsFeatures(LES)
    Reynolds_stresses = np.vstack(tau_vec)
    return Reynolds_stresses

def make_grids_dict(path, cases):
    def make_grid_dict(case):
        RANS = load_case_data(case, path, 'no')  # no mesh needed for grid
        Cx = RANS['Cx'].to_numpy()
        Cy = RANS['Cy'].to_numpy()
        if 'DUCT' in case:
            Cx = Cy
            Cy = RANS['Cz'].to_numpy()
        return {'Cx': Cx, 'Cy': Cy}

    grid_dict = {}
    for case in cases:
        grid_dict[case] = make_grid_dict(case)
    return grid_dict

def load_data_from_config(config, train_cases, test_cases, path):
    input_features = config["features"]["input"]
    output_features = config["features"]["output"]
    model_type = config["model"]["type"].lower()
    use_mesh = any(kw in model_type for kw in ["gcn", "gat", "gated", "graph", "mpnn"])

    def extract_grid_dict(case, rans):
        Cx = rans["Cx"].to_numpy()
        Cy = rans["Cy"].to_numpy()
        if "DUCT" in case:
            Cx = Cy
            Cy = rans["Cz"].to_numpy()
        return {"Cx": Cx, "Cy": Cy}

    data_bundle = {
        "X_train": [], "y_train": [],
        "X_test": [], "y_test": [],
        "grid_dicts_train": {}, "grid_dicts_test": {}
    }

    if use_mesh:
        data_bundle["edge_index_train"] = []
        data_bundle["edge_index_test"] = []

    # === Load training cases ===
    for case in train_cases:
        loaded = load_case_data(case, path, 'yes' if use_mesh else 'no')
        rans, les, *mesh = loaded if isinstance(loaded, tuple) else (loaded, None)

        X = torch.tensor(
            np.vstack([get_case_features(rans, feat) for feat in input_features]).T,
            dtype=torch.float32
        )
        if config['features']['output_type'] == 'reynolds':
            y = torch.tensor(load_ReynoldsTensor(les).T, dtype=torch.float32)
        else:
            y = torch.tensor(
                np.vstack([get_case_features(les, feat) for feat in output_features]).T,
                dtype=torch.float32
            )

        data_bundle["X_train"].append(X)
        data_bundle["y_train"].append(y)
        data_bundle["grid_dicts_train"][case] = extract_grid_dict(case, rans)

        if use_mesh and mesh:
            edge_index = torch.tensor(mesh[0]['E_list'].T, dtype=torch.long)
            data_bundle["edge_index_train"].append(edge_index)

    # === Load testing cases ===
    for case in test_cases:
        loaded = load_case_data(case, path, 'yes' if use_mesh else 'no')
        rans, les, *mesh = loaded if isinstance(loaded, tuple) else (loaded, None)

        X = torch.tensor(
            np.vstack([get_case_features(rans, feat) for feat in input_features]).T,
            dtype=torch.float32
        )
        if config['features']['output_type'] == 'rst':
            y = torch.tensor(load_ReynoldsTensor(les).T, dtype=torch.float32)
        else:
            y = torch.tensor(
                np.vstack([get_case_features(les, feat) for feat in output_features]).T,
                dtype=torch.float32
            )

        data_bundle["X_test"].append(X)
        data_bundle["y_test"].append(y)
        data_bundle["grid_dicts_test"][case] = extract_grid_dict(case, rans)

        if use_mesh and mesh:
            edge_index = torch.tensor(mesh[0]['E_list'].T, dtype=torch.long)
            data_bundle["edge_index_test"].append(edge_index)

    return data_bundle

