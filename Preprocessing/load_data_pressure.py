import os
import numpy as np
import torch
import pandas as pd


def load_case_data(case, path, mesh_yn):
    def make_dataPaths(data_filepath):
        LES_dict = os.path.join(data_filepath, 'LES_Dict')
        RANS = os.path.join(data_filepath, 'RANS')
        RANS_dict_ext = os.path.join(data_filepath, 'RANS_Dict_ext')
        RANS_mesh = os.path.join(data_filepath, 'RANS_MeshInfo')
        return LES_dict, RANS, RANS_dict_ext, RANS_mesh

    def loadRANS_DICT_ext(case, path):
        filename = f'{path}/{case}_DICT.pkl'
        with open(filename, 'rb') as file:
            return pd.read_pickle(file)

    def loadRANS_mesh_info(case, path):
        filename = f'{path}/{case}_MeshInfo.pkl'
        with open(filename, 'rb') as file:
            return np.load(file, allow_pickle=True)

    LES_d_path, RANS_path, RANS_dict_ext, RANS_MESH_PATH = make_dataPaths(path)
    rans_dict = loadRANS_DICT_ext(case, RANS_dict_ext)
    if mesh_yn == 'yes':
        rans_mesh = loadRANS_mesh_info(case, RANS_MESH_PATH)
        return rans_dict, rans_mesh
    else:
        return rans_dict


def get_case_features(RANS, feature):
    return RANS[feature].to_numpy()


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


def load_data_pressure_from_config(config, train_cases, test_cases, path):
    input_features = config['features']['input']
    output_features = config['features']['output']
    use_mesh = config.get('model', {}).get('type', '').lower() not in ['fcn', 'fcn_ten']

    def build_tensor_list(cases, input_features, output_features, path):
        X_list = []
        y_list = []

        for case in cases:
            loaded = load_case_data(case, path, 'yes' if use_mesh else 'no')
            RANS = loaded[0] if isinstance(loaded, tuple) else loaded  # Always get the RANS dict

            # DEBUG statements to catch issues early
            '''print(f"[DEBUG] Case: {case}")
            print(f"[DEBUG] Type of loaded: {type(loaded)}")
            if isinstance(RANS, dict):
                print(f"[DEBUG] RANS keys: {list(RANS.keys())}")
            else:
                print(f"[DEBUG] Unexpected RANS format: {RANS}")'''

            # Convert features
            RANS_data = [get_case_features(RANS, feat) for feat in input_features]
            X_tensor = torch.tensor(np.vstack(RANS_data).T, dtype=torch.float32)
            X_list.append(X_tensor)

            # Only one output feature is expected for pressure (e.g., 'p')
            p_data = get_case_features(RANS, output_features[0])
            y_tensor = torch.tensor(p_data.reshape(-1, 1), dtype=torch.float32)
            y_list.append(y_tensor)

        return X_list, y_list

    X_train, y_train = build_tensor_list(train_cases, input_features, output_features, path)
    X_test, y_test = build_tensor_list(test_cases, input_features, output_features, path)
    grid_dicts_test = make_grids_dict(path, test_cases)
    grid_dicts_train = make_grids_dict(path, train_cases)
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "grid_dicts_test": grid_dicts_test,
        "grid_dicts_train": grid_dicts_train,
    }
