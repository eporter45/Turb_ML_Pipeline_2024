import numpy as np
import torch
from Preprocessing.load_stack import load_and_stack_rans

def calc_tke_loss(y_pred, y_true):
    tke_pred = 0.5 * (y_pred[:, 0] + y_pred[:, 3] + y_pred[:, 5])
    tke_true = 0.5 * (y_true[:, 0] + y_true[:, 3] + y_true[:, 5])
    return torch.mean((tke_pred - tke_true) ** 2)

def add_tke_loss(y_pred, y_true, config):
    weight = config['training']['loss_terms']['tke_weight']
    tke_var = calc_tke_loss(y_pred, y_true)
    return weight * tke_var

def make_tke_loss(y_pred, y_true, config):
    tke_loss = 0
    if config['training']['loss_terms']['tke']:
        tke_loss = add_tke_loss(y_pred, y_true, config)
    return tke_loss

