import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from Trials import TRIALS
from training_utils.weight_case_losses import build_case_loss_weights
from training_utils.initialize_scheduler import initialize_scheduler
import random
import numpy as np
from Models.FCN import get_activation, train_model_with_scheduler, train_model, make_predictions

class BranchFCN(nn.Module):
    def __init__(self, input_size: int,
                 output_size: int,
                 branch_layers: list[int],
                 branch_activation: bool,
                 trunk_layers: list[int],
                 hidden_activation: str = "relu",
                 output_activation: str = "sigmoid",
                 dropout: float = 0.0,
                 seed: int = 42):
        super().__init__()

        if not trunk_layers:
            raise ValueError("trunk_layers must be a non-empty list")

        # seeding ----------
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        random.seed(seed); np.random.seed(seed)

        h_act = get_activation(hidden_activation)
        self.branch_do = nn.Dropout(dropout)
        self.trunk_do  = nn.Dropout(dropout)

        # ----- branches
        self.branches = nn.ModuleList()
        for _ in range(input_size):
            mods, in_dim = [], 1
            for j, out_dim in enumerate(branch_layers):
                if branch_activation:
                    mods += [nn.Linear(in_dim, out_dim), h_act]
                else:
                    mods += [nn.Linear(in_dim, out_dim)]
                if j == len(branch_layers) - 1:
                    mods.append(self.branch_do)
                in_dim = out_dim
            self.branches.append(nn.Sequential(*mods))

        # ----- trunk
        concat_dim = branch_layers[-1] * input_size
        mods, in_dim = [], concat_dim
        for j, out_dim in enumerate(trunk_layers):
            mods += [nn.Linear(in_dim, out_dim), h_act]
            if j < len(trunk_layers) - 1:
                mods.append(self.trunk_do)
            in_dim = out_dim
        self.trunk = nn.Sequential(*mods)

        self.output_layer = nn.Linear(trunk_layers[-1], output_size)
        self.output_activation = (
            get_activation(output_activation) if output_activation else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pieces = [branch(x[:, i : i + 1]) for i, branch in enumerate(self.branches)]
        h = torch.cat(pieces, dim=-1)
        h = self.trunk(h)
        h = self.output_layer(h)
        return self.output_activation(h)




def runBranch_model(X_train, y_train, X_test, config, directory, device):
    """
    Mirror of runSimple_model, but builds BranchFCN.

    Expects these keys to be present in config:
        config['model']['branch_layers']   (list[int])
        config['model']['trunk_layers']    (list[int])
        config['model']['hidden_activation']
        config['model']['output_activation']   # optional, default 'sigmoid'
        config['model']['dropout']
    """
    # --- reproducibility -------------------------------------------------
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    # --- instantiate model ----------------------------------------------
    model = BranchFCN(
        input_size        = X_train[0].shape[1],           # # of scalar features
        output_size       = y_train[0].shape[1],           # usually 1
        branch_layers     = config['model']['branch_layers'],
        branch_activation = config['model']['branch_activation'],
        trunk_layers      = config['model']['trunk_layers'],
        hidden_activation = config['model']['hidden_activation'],
        output_activation = config['model'].get('output_activation', 'sigmoid'),
        dropout           = config['model']['dropout'],
        seed              = seed,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of learnable parameters: {num_params}")

    # --- train with or without scheduler -------------------------------
    if config['training']['scheduler']['enabled']:
        loss_hist, best_epoch, best_model, optimizer = \
            train_model_with_scheduler(model, X_train, y_train,
                                       config, directory, device)
    else:
        loss_hist, best_epoch, best_model, optimizer = \
            train_model(model, X_train, y_train,
                        config, directory, device)

    # --- predictions on test cases -------------------------------------
    preds = make_predictions(best_model.to(device),
                             [x.to(device) for x in X_test])

    return (preds, loss_hist, best_model,
            best_epoch, best_model.state_dict(), optimizer)





