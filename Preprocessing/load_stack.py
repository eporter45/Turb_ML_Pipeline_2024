import torch
import numpy as np
from Preprocessing.Load_data import load_case_data, get_case_features

def load_and_stack_rans(cases, input_feature_names, output_feature_names, data_path):
    x_list, y_list, grid_dict = [], [], {}
    for case in cases:
        #print(f"[INFO] Loading case: {case}")
        rans_dict, _ = load_case_data(case, data_path, mesh_yn='no')

        X = torch.tensor(
            np.vstack([get_case_features(rans_dict, feat) for feat in input_feature_names]).T,
            dtype=torch.float32
        )

        Y = torch.tensor(
            np.vstack([get_case_features(rans_dict, feat) for feat in output_feature_names]).T,
            dtype=torch.float32
        )

        x_list.append(X)
        y_list.append(Y)

        Cx = rans_dict['Cx'].to_numpy()
        Cy = rans_dict['Cy'].to_numpy()
        if 'DUCT' in case:
            Cx = Cy
            Cy = rans_dict['Cz'].to_numpy()
        grid_dict[case] = {'Cx': Cx, 'Cy': Cy}
        #print(f"  - X shape: {X.shape}, Y shape: {Y.shape}")
    return x_list, y_list, grid_dict

def load_and_stack_les(cases, input_feature_names, output_feature_names, data_path):
    x_list, y_list, grid_dict = [], [], {}
    for case in cases:
        #print(f"[INFO] Loading case: {case}")
        rans_dict, les_dict = load_case_data(case, data_path, mesh_yn='no')

        X = torch.tensor(
            np.vstack([get_case_features(rans_dict, feat) for feat in input_feature_names]).T,
            dtype=torch.float32
        )

        Y = torch.tensor(
            np.vstack([get_case_features(les_dict, feat) for feat in output_feature_names]).T,
            dtype=torch.float32
        )

        x_list.append(X)
        y_list.append(Y)

        Cx = rans_dict['Cx'].to_numpy()
        Cy = rans_dict['Cy'].to_numpy()
        if 'DUCT' in case:
            Cx = Cy
            Cy = rans_dict['Cz'].to_numpy()
        grid_dict[case] = {'Cx': Cx, 'Cy': Cy}
        #print(f"  - X shape: {X.shape}, Y shape: {Y.shape}")
    return x_list, y_list, grid_dict
