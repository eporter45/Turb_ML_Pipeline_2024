import torch
import torch.optim as optim
from Preprocessing.load_stack import load_and_stack_rans

def is_tensor(x, device):
    return x.to(device) if torch.is_tensor(x) else torch.as_tensor(x, device=device)


def make_loss_weights_from_rans_k(case_names, data_path, device):
    """Load RANS k from each case and compute loss weights based on k magnitude.
    """
    dummy_input = ['Ux']
    dummy_output = ['k']
    x_list, y_list, _ = load_and_stack_rans(case_names, dummy_input, dummy_output, data_path)

    print(f'[DEBUG] shape of y list: {[y.shape for y in y_list]}')

    # Compute per-case mean |k|
    k_mean = torch.tensor([torch.mean(torch.abs(y[:, 0])) for y in y_list], device=device)

    # Invert to give more weight to low-k cases
    weights = 1.0 / (k_mean + 1e-8)

    # Normalize to mean 1 so the total loss scale stays stable
    weights = weights / torch.min(weights)

    return weights


def make_null_weights(y_train, device):
    return torch.ones(len(y_train), dtype=torch.float32, device=device)


def get_weight_mode(config):
    mode = config["training"]['loss_case_scaling']["loss_mode"]
    modes = ['tke', 'tke_sq']
    if mode.lower() in modes:
        return mode.lower()
    return None


def build_case_loss_weights(config, y_train, device,
                            case_names=None):
    mode = get_weight_mode(config)
    data_path = config['paths']['data']
    if mode == "tke" or mode == "tke_sq":
        assert case_names is not None and data_path is not None
        weights = make_loss_weights_from_rans_k(case_names, data_path, device)
        if mode == "tke_sq":
            weights = weights ** 2
        return weights

    # existing modes...
    return make_null_weights(y_train, device)