import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import copy
from Trials import TRIALS
from training_utils.weight_case_losses import build_case_loss_weights
from training_utils.initialize_scheduler import initialize_scheduler
from training_utils.tke_loss import make_tke_loss
import random
import numpy as np


# util for activation naming
def get_activation(name):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "leakyrelu":
        return nn.LeakyReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

class TENxTEN(nn.Module):
    def __init__(self, dropout, input_size, output_size, activation="leakyrelu", layers=10, width=10, seed=42):
        super(TENxTEN, self).__init__()

        # Set seeds for reproducibility
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Save params
        self.input_size = input_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = int(layers)
        self.width = int(width)

        self.activation_fn = get_activation(activation)
        # Build sequential model
        layers_list = []
        l =  nn.Linear(input_size, width)
        nn.init.xavier_normal_(l.weight)
        layers_list.append(l)

        for i in range(self.num_layers - 2):
            layers_list.append(self.activation_fn)
            if i % 2 == 0:
                layers_list.append(nn.Dropout(self.dropout))
            l = nn.Linear(width, width)
            nn.init.xavier_normal_(l.weight)
            layers_list.append(l)

        layers_list.append(self.activation_fn)
        l = nn.Linear(width, output_size)
        nn.init.xavier_normal_(l.weight)
        layers_list.append(l)
        #layers_list.append(nn.Identity())  # final layer is ungated

        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)

def compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses,
                      net_data_history, net_phys_history, net_loss_history):
    return {
        'case_net': net_case_losses,
        'case_phys': phys_case_losses,
        'case_data': data_case_losses,
        'total_data': net_data_history,
        'total_phys': net_phys_history,
        'total_loss': net_loss_history
    }

def create_loss_history_dict(case_names):
    return ({name: [] for name in case_names},
            {name: [] for name in case_names},
            {name: [] for name in case_names})

def resolve_train_case_names(config):
    trial_val = config['trial_name']
    if isinstance(trial_val, dict):
        return trial_val['train']
    else:
        return TRIALS[trial_val]['train']

def save_model_bundle(model, optimizer, directory, tag='final_model'):
    path = os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))
    torch.save(model.state_dict(), os.path.join(path, 'model_dict.pth'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_dict.pth'))

def make_predictions(model, X_test):
    model.eval()
    outputs = []
    with torch.no_grad():
        for X in X_test:
            pred = model(X)
            outputs.append(pred)
    return outputs

def train_model(model, X_train, y_train, config, directory, device):
    criterion = nn.MSELoss(reduction="mean")
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']), weight_decay=0.0001)
    best_epoch = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)

    # -------- resolve train_case_names robustly ------------
    train_case_names = resolve_train_case_names(config)
    # -------------------------------------------------------
    # net loss dicts
    net_case_losses, data_case_losses, phys_case_losses = create_loss_history_dict(train_case_names)
    # net loss summed from all cases
    net_loss_history, net_phys_history, net_data_history = [], [], []


    # get loss weights
    loss_weights = build_case_loss_weights(config, y_train=y_train, device=device)
    print(f'[INFO] Loss Weights  {loss_weights}')
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_net_sum = 0
        epoch_phys_sum = 0
        epoch_data_sum = 0
        for idx, (X, y) in enumerate(zip(X_train, y_train)):
            w_case = loss_weights[idx]
            name = train_case_names[idx]
            dataset = TensorDataset(X.to(device), y.to(device))
            loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

            mse_sum = 0        #case loss tracking
            phys_sum = 0
            n_points = 0       #num batches count

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                batch_mse = criterion(outputs, y_batch)
                batch_tke_var_loss = make_tke_loss(outputs, y_batch, config)
                loss = (batch_mse + batch_tke_var_loss) * w_case
                #torch operations
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # track net loss points
                bs = X_batch.size(0)
                mse_sum += batch_mse.item() * bs
                phys_sum += batch_tke_var_loss.item() * bs
                n_points += bs

            case_mse_loss = (mse_sum / n_points) * w_case.item()
            case_phys_loss = (phys_sum / n_points) * w_case.item()
            total_case_loss = (case_mse_loss + case_phys_loss)
            #add to histories
            net_case_losses[name].append(total_case_loss)
            phys_case_losses[name].append(case_phys_loss)
            data_case_losses[name].append(case_mse_loss)
            #track total epoch loss
            epoch_net_sum += total_case_loss
            epoch_phys_sum += case_phys_loss
            epoch_data_sum += case_mse_loss

        # add to histories
        net_loss = epoch_net_sum
        net_phys_loss = epoch_phys_sum
        net_data_loss = epoch_data_sum
        net_loss_history.append(epoch_net_sum)
        net_phys_history.append(net_phys_loss)
        net_data_history.append(net_data_loss)

        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.5f} Data Loss: {net_data_loss:.5f}, Physics Loss:{net_phys_loss:.5f}", flush=True)

        if epoch % 250 == 0 and epoch != 0:
            save_model_bundle(best_model, optimizer, directory, f'epoch_{epoch}')

    save_model_bundle(best_model, optimizer, directory, 'best_model')
    save_model_bundle(model, optimizer, directory, 'final_model')

    # make case_losses data struct
    loss_df = compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses,
                                net_data_history, net_phys_history, net_loss_history)
    return loss_df, best_epoch, best_model, optimizer


def train_model_with_scheduler(model, X_train, y_train, config, directory, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(config['training']['lr']), weight_decay=0.0001)
    scheduler = initialize_scheduler(optimizer, config)
    best_epoch = 0
    best_loss = float('inf')
    best_model = copy.deepcopy(model)
    #resolve train_case_names robustly
    train_case_names = resolve_train_case_names(config)
    # net loss dicts
    net_case_losses, data_case_losses, phys_case_losses = create_loss_history_dict(train_case_names)
    # net loss summed from all cases
    net_loss_history, net_phys_history, net_data_history = [], [], []
    #initialize weights
    loss_weights = build_case_loss_weights(config, y_train=y_train, device=device, case_names=train_case_names )
    print(f'[INFO] Loss Weights {loss_weights}')

    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_net_sum = 0
        epoch_phys_sum = 0
        epoch_data_sum = 0

        for idx, (X, y) in enumerate(zip(X_train, y_train)):
            name = train_case_names[idx]
            dataset = TensorDataset(X.to(device), y.to(device))
            loader = DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)

            w_case = loss_weights[idx]
            mse_sum = 0  # case loss tracking
            phys_sum = 0
            n_points = 0  # num batches count
            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                batch_mse = criterion(outputs, y_batch)
                batch_tke_var_loss = make_tke_loss(outputs, y_batch, config)
                loss = (batch_mse + batch_tke_var_loss) * w_case
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # track net loss points
                bs = X_batch.size(0)
                mse_sum += batch_mse * bs
                phys_sum += batch_tke_var_loss * bs
                n_points += bs

            case_mse_loss = (mse_sum / n_points) * w_case.item()
            case_phys_loss = (phys_sum / n_points) * w_case.item()
            total_case_loss = case_mse_loss + case_phys_loss
            # add to histories
            net_case_losses[name].append(total_case_loss)
            phys_case_losses[name].append(case_phys_loss)
            data_case_losses[name].append(case_mse_loss)
            # track total epoch loss
            epoch_net_sum += total_case_loss
            epoch_phys_sum += case_phys_loss
            epoch_data_sum += case_mse_loss

        # compile loss histories
        net_loss = epoch_net_sum
        avg_net_loss = net_loss / len(train_case_names)
        net_phys_loss = epoch_phys_sum
        net_data_loss = epoch_data_sum
        net_loss_history.append(epoch_net_sum)
        net_phys_history.append(net_phys_loss)
        net_data_history.append(net_data_loss)

        if net_loss < best_loss:
            best_loss = net_loss
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.5f} Data Loss: {net_data_loss:.5f}, Physics Loss:{net_phys_loss:.5f}", flush=True)
            print(f"Epoch[{epoch}/{config['training']['epochs']}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        if epoch % 250 == 0 and epoch != 0:
            save_model_bundle(best_model, optimizer, directory, f'epoch_{epoch}')

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(net_loss)
        elif (config['training']['scheduler']['type'] == 'reduce'
              or config['training']['scheduler']['type'] == 'reduce_lr'
              or config['training']['scheduler']['type'] == 'reduce_on_plateau'):
            scheduler.step(avg_net_loss)
        else:
            scheduler.step()


    save_model_bundle(best_model, optimizer, directory, 'best_model')
    save_model_bundle(model, optimizer, directory, 'final_model')
    # make case_losses data struct
    loss_df = compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses,
                                net_data_history, net_phys_history, net_loss_history)

    return loss_df, best_epoch, best_model, optimizer

def runSimple_model(X_train, y_train, X_test, config, directory, device):
    # Set seed for reproducibility
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)

    model = TENxTEN(
        dropout=config['model']['dropout'],
        input_size=X_train[0].shape[1],
        output_size=y_train[0].shape[1],
        activation=config['model']['activation'],
        layers=config['model']['layers'],
        width=config['model']['width'],
    ).to(device)

    print(f'[MODEL] Num of layers: {model.num_layers} ')
    print(f'[MODEL] Width of Layers: {model.width} ')
    weights =  model.state_dict()
    #print(f'[Model] Weights keys: {weights.keys()}')
    #print(f'[MODEL] Model Weights: {weights} ')
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Number of learnable parameters: {total_params}")

    if config['training']['scheduler']['enabled']:
        loss_df, best_epoch, best_model, optimizer = train_model_with_scheduler(
            model, X_train, y_train, config, directory, device
        )
    else:
        loss_df, best_epoch, best_model, optimizer = train_model(
            model, X_train, y_train, config, directory, device
        )

    # Make predictions
    predictions = make_predictions(best_model.to(device), [x.to(device) for x in X_test])
    return predictions, loss_df, best_model, best_epoch, best_model.state_dict(), optimizer
