""" This script takes in a configuration and returns the predictions"""
"""Based on the model architexture"""

from Models.FCN import make_predictions
import torch

def make_preds(model, X_test):
    model.eval()
    outputs = []
    with torch.no_grad():
        for X in X_test:
            pred = model(X)
            outputs.append(pred)
    return outputs