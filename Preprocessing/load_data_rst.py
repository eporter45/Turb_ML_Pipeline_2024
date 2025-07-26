import os
import numpy as np
import torch
import pandas as pd
from Preprocessing.load_stack import load_and_stack_les
from typing import List, Tuple



def load_data_rst_from_config(config, train_cases, test_cases, path):

    # initialize output of function
    data_bundle = {
        "X_train": [], "y_train": [],
        "X_test": [], "y_test": [],
        "grid_dicts_train": {}, "grid_dicts_test": {}
    }
    input_features = config['features']['input']
    output_features = config['features']['output']
    out_feat = []
    if output_features is not List:
        out_feat = ['uu', 'uv', 'uw', 'vv', 'vw', 'ww']
    else:
        out_feat = output_features
    use_mesh = config.get('model', {}).get('type', '').lower() not in ['fcn', 'fcn_ten']
    if use_mesh:
       data_bundle["edge_index_train"] = []
       data_bundle["edge_index_test"] = []

    #load and store in output bundle
    x_train, y_train, grid_train = load_and_stack_les(train_cases, input_features, out_feat, data_path=path)
    data_bundle["X_train"], data_bundle['y_train'], data_bundle['grid_dicts_train'] = x_train, y_train, grid_train
    x_test, y_test, grid_test = load_and_stack_les(test_cases, input_features, out_feat, data_path=path)
    data_bundle["X_test"], data_bundle['y_test'], data_bundle['grid_dicts_test'] = x_test, y_test, grid_test
    print('[DEBUG] x_train length:', len(x_train) )
    print('[DEBUG] x_train[0] shape:', x_train[0].shape)
    print('[DEBUG] y_train length:', len(y_train) )
    print('[DEBUG] y_train[0] shape:', y_train[0].shape)
    print('[DEBUG] x_test length:', len(x_test) )
    print('[DEBUG] x_test[0] shape:', x_test[0].shape)
    print('[DEBUG] y_test length:', len(y_test) )
    print('[DEBUG] y_test[0] shape:', y_test[0].shape)
    return data_bundle

