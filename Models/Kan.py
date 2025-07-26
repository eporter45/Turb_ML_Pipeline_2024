import os
import copy
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from KAN_lib.kan.MultKAN import KAN_me as KAN
from Trials import TRIALS
from training_utils.weight_case_losses import build_case_loss_weights
from training_utils.tke_loss import make_tke_loss

def initialize_kan(input_features, output_features, config, device):
    def sanitize_shape(shape):
        # Convert nested shape like [[4, 0], [5, 0], ...] to [4, 5, ...]
        return [s[0] if isinstance(s, list) and len(s) == 2 and s[1] == 0 else s for s in shape]

    print("[DEBUG] Raw model shape from config:", config['model']['shape'], flush=True)
    print("[DEBUG] Raw type of model shape:", type(config['model']['shape']), flush=True)

    shape = sanitize_shape(config['model']['shape'])

    print("[DEBUG] Sanitized model shape:", shape)

    model = KAN(
        width=shape,
        grid=config['model'].get('grid', 10),
        k=config['model'].get('spline_order', 3),
        seed=config['model'].get('seed', 42),
        grid_range=config['model'].get('grid_range', [-1.2, 1.2]),
        #update_grid=config['model'].get('update_grid', True),
        #use_kan=config['model'].get('use_kan', True)
    ).to(device)

    print("[DEBUG] model width as interpreted by MultKAN:", model.width, flush=True)
    print("[INFO] KAN model initialized", flush=True)
    return model

def try_plot_model(model, save_path, device, input_example=None):
    """
    Attempts to plot the KAN model and save it to the given path.

    Args:
        model (KAN): The trained KAN model.
        save_path (str): Full path where the plot should be saved.
        device (torch.device): Device to send input to.
        input_example (Tensor, optional): Example input tensor for warm-up. If None, skip warm-up.
    """
    try:
        model.eval()

        if input_example is not None:
            _ = model(input_example.to(device))  # Warm-up pass

        # Apply symbolic rules if available
        if hasattr(model, "apply_symbolic_rule"):
            model.apply_symbolic_rule()

        # === Additional Debug Checks ===
        core_model = getattr(model, "model", model)

        print("[DEBUG] symbolic_fun:", getattr(core_model, "symbolic_fun", None))
        print("[DEBUG] cache_data:", getattr(core_model, "cache_data", None))
        print("[DEBUG] acts:", getattr(core_model, "acts", None))
        print("[DEBUG] width:", getattr(core_model, "width", None))

        # Print symbolic expressions for each layer
        if hasattr(core_model, "symbolic_fun"):
            for i, layer in enumerate(core_model.symbolic_fun):
                expr = getattr(layer, 'expr', None)
                print(f"[DEBUG] symbolic layer {i} expr: {expr}")

        fig = model.plot()
        if fig is not None:
            fig.set_size_inches(12, 5)
            fig.savefig(save_path, dpi=150)
            plt.close(fig)
        else:
            print("[WARNING] model.plot() returned None.")
        model.train()

    except Exception as e:
        print(f"[WARNING] Could not save model plot: {e}")

def save_model_bundle(state_dict, optimizer, directory, tag='final_model'):
    """
    Saves the model state_dict and optimizer state.

    Args:
        state_dict (dict): The state dict from the model (model.state_dict()).
        optimizer (Optimizer): The optimizer whose state to save.
        directory (str): Output directory.
        tag (str): Subdirectory name (e.g., 'best_model' or 'final_model').
    """
    path = os.path.join(directory, tag)
    os.makedirs(path, exist_ok=True)
    torch.save(state_dict, os.path.join(path, 'model_dict.pth'))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_dict.pth'))

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

def train_kan_model(model, X_train, y_train, config, directory, device):
    input_size = X_train[0].shape[1]
    output_size = y_train[0].shape[1]
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=0.0001)
    best_epoch = 0
    best_loss = float('inf')
    best_state_dict = copy.deepcopy(model.state_dict())  # <-- store weights only
    # resolve train_case_names robustly
    train_case_names = resolve_train_case_names(config)
    # initialize history dicts and net loss vectors
    net_case_losses, data_case_losses, phys_case_losses = create_loss_history_dict(train_case_names)
    net_loss_history, net_phys_history, net_data_history = [], [], []

    #initialize loss weights
    loss_weights = build_case_loss_weights(config, y_train, device)
    print(f'[INFO] Loss weights: {loss_weights}')
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_net_sum = 0
        epoch_phys_sum = 0
        epoch_data_sum = 0
        for idx, (X, y) in enumerate(zip(X_train, y_train)):
            name = train_case_names[idx]
            w_case = loss_weights[idx]
            mse_sum = 0  # case loss tracking
            phys_sum = 0
            n_points = 0  # num batches count
            dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
            loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=True)
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                batch_mse = criterion(outputs, y_batch)
                batch_tke_var_loss = make_tke_loss(outputs, y_batch, config)
                loss = (batch_mse + batch_tke_var_loss) * w_case
                #torch operations
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #loss tracking
                bs = X_batch.size(0)
                mse_sum += batch_mse.item() * bs
                phys_sum += batch_tke_var_loss.item() * bs
                n_points += bs

            case_mse_loss = (mse_sum / n_points) * w_case
            case_phys_loss = (phys_sum / n_points) * w_case
            total_case_loss = case_mse_loss + case_phys_loss
            # add to histories
            net_case_losses[name].append(total_case_loss)
            phys_case_losses[name].append(case_phys_loss)
            data_case_losses[name].append(case_mse_loss)
            # track total epoch loss
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
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.5f} Data Loss: {net_data_loss:.5f}, Physics Loss:{net_phys_loss:.5f}", flush=True)

        if epoch % 250 == 0 and epoch != 0:
            save_model_bundle(model.to(device).state_dict(), config, train_case_names, tag=f'epoch_{epoch}')
            #try_plot_model(model, os.path.join(checkpoint_dir, "network_plot.png"), device, input_example=X_train[0][:32])

    # Save best model
    best_model = initialize_kan(input_size, output_size, config, device)
    best_model.load_state_dict(best_state_dict)

    save_model_bundle(best_state_dict, optimizer, directory, 'best_model')
    save_model_bundle(model.state_dict(), optimizer, directory, 'final_model')

    #try_plot_model(best_model, os.path.join(best_dir, "network_plot.png"), device, input_example=X_train[0][:32])
    #try_plot_model(model, os.path.join(final_dir, "network_plot.png"), device, input_example=X_train[0][:32])
    # make case_losses data struct
    loss_df = compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses,
                                net_data_history, net_phys_history,net_loss_history)
    return loss_df, best_model, best_epoch, best_state_dict, optimizer

from training_utils.initialize_scheduler import initialize_scheduler

def train_kan_model_with_scheduler(model, X_train, y_train, config, directory, device):
    input_size = X_train[0].shape[1]
    output_size = y_train[0].shape[1]
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=0.0001)
    scheduler = initialize_scheduler(optimizer, config)
    best_epoch = 0
    best_loss = float('inf')
    print(f'[INFO] Initializing best_state_dict', flush=True)
    best_state_dict = copy.deepcopy(model.state_dict())
    print(f'[INFO] Initializing best_state_dict initialized', flush=True)
    #resolve train_case_names robustly
    train_case_names = resolve_train_case_names(config)
    #initialize history dicts and net loss vectors
    net_case_losses, data_case_losses, phys_case_losses = create_loss_history_dict(train_case_names)
    net_loss_history, net_phys_history, net_data_history = [], [], []
    #initialize loss weights
    loss_weights = build_case_loss_weights(config, y_train, device, case_names=train_case_names)
    print(f'[INFO] LOSS WEIGHTS: {loss_weights}', flush=True)
    for epoch in range(config['training']['epochs']):
        model.train()
        epoch_net_sum = 0
        epoch_phys_sum = 0
        epoch_data_sum = 0

        for idx, (X, y) in enumerate(zip(X_train, y_train)):
            name = train_case_names[idx]
            w_case = loss_weights[idx]
            mse_sum = 0  # case data loss tracking
            phys_sum = 0  #case phys loss tracking
            n_points = 0  # num batches count
            #use batching method
            dataset = torch.utils.data.TensorDataset(X.to(device), y.to(device))
            loader = torch.utils.data.DataLoader(dataset, batch_size=config['training']['batch_size'], shuffle=False)

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                batch_mse = criterion(outputs, y_batch)
                batch_tke_var_loss = make_tke_loss(outputs, y_batch, config)
                loss = (batch_mse + batch_tke_var_loss) * w_case
                #input torch commands
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #track batch losses
                bs = X_batch.size(0)
                mse_sum += batch_mse.item() * bs
                phys_sum += batch_tke_var_loss.item() * bs
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
            best_state_dict = copy.deepcopy(model.state_dict())

        if epoch % config['training']['eval_every'] == 0:
            print(f"Epoch [{epoch}/{config['training']['epochs']}], Net Loss: {net_loss:.5f} Data Loss: {net_data_loss:.5f}, Physics Loss:{net_phys_loss:.5f}", flush=True)
            print(f"Epoch[{epoch}/{config['training']['epochs']}], Learning Rate: {scheduler.get_last_lr()[0]:.6f}", flush=True)

        if epoch % 250 == 0 and epoch != 0:
            save_model_bundle(model.to(device).state_dict(), config, train_case_names, tag=f'epoch_{epoch}')

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(net_loss)
        elif (config['training']['scheduler']['type'] == 'reduce'
              or config['training']['scheduler']['type'] == 'reduce_lr'
              or config['training']['scheduler']['type'] == 'reduce_on_plateau'):
            scheduler.step(avg_net_loss)
        else:
            scheduler.step()

    # Save best model
    best_model = initialize_kan(input_size, output_size, config, device)
    best_model.load_state_dict(best_state_dict)

    save_model_bundle(best_state_dict, optimizer, directory, 'best_model')
    save_model_bundle(model.state_dict(), optimizer, directory, 'final_model')
    #try_plot_model(model, os.path.join(final_dir, "network_plot.png"), device, input_example=X_train[0][:32])
    # make case_losses data struct
    loss_df = compile_loss_logs(net_case_losses, phys_case_losses, data_case_losses,
                                net_data_history, net_phys_history, net_loss_history)
    return loss_df, best_model, best_epoch, best_state_dict, optimizer

def train_with_fit(model, X_train, y_train, config, output_dir, device):
    input_size = X_train[0].shape[1]
    output_size = y_train[0].shape[1]
    data = {"train_input": torch.cat(X_train, dim=0),
            "train_label": torch.cat(y_train, dim=0),
            "test_input": torch.cat(X_train, dim=0)[:32],  # dummy test data
            "test_label": torch.cat(y_train, dim=0)[:32]    }

    # === Fit using internal KAN optimizer ===
    steps = config['training'].get('epochs', 500)
    lamb = config['training'].get('lamb', 1e-3)
    optimizer = config['training'].get('opt', 'Adam')
    print(f"[INFO] Fitting KAN with opt={optimizer}, steps={steps}, lamb={lamb}")

    results = model.fit(data, opt=optimizer, steps=steps, lamb=lamb)

    # === Save final model ===
    final_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(final_dir, "model_dict.pth"))
    #try plot
    try_plot_model(model, os.path.join(final_dir, "network_plot.png"), device, input_example=data["train_input"][:32])

    # === Return stub values to match interface ===
    train_loss_history = results['train_loss']
    best_epoch = steps - 1
    best_state_dict = copy.deepcopy(model.state_dict())
    # Save best model
    best_model = initialize_kan(input_size, output_size, config, device)
    best_model.load_state_dict(best_state_dict)
    return train_loss_history, best_model, best_epoch, best_state_dict, optimizer

def runKAN_model(X_train, y_train, X_test, config, output_dir, device):
    input_size = X_train[0].shape[1]
    output_size = y_train[0].shape[1]
    print(f'[DEBUG]: input size {input_size}, output size {output_size}', flush=True)
    model = initialize_kan(input_features=input_size, output_features=output_size, config=config,device=device)
    #print(f'[INFO]: Model Parameters: {model._parameters()}')
    if config.get('training', {}).get('use_fit', False):
        loss_df, best_model, best_epoch, best_state_dict, optimizer = train_with_fit(
            model, X_train, y_train, config, output_dir, device)

    elif config['training']['scheduler']['enabled'] == True:
        loss_df, best_model, best_epoch, best_state_dict, optimizer = train_kan_model_with_scheduler(
            model, X_train, y_train, config, output_dir, device)
    else:
        loss_df, best_model, best_epoch, best_state_dict, optimizer = train_kan_model(
            model, X_train, y_train, config, output_dir, device
        )

    # Make predictions     return case_losses, best_model, best_epoch, best_state_dict, optimizer
    best_model.eval()
    with torch.no_grad():
        predictions = [best_model(x.to(device)) for x in X_test]

    return predictions, loss_df, best_model, best_epoch, best_state_dict, optimizer
