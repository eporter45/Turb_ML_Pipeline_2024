import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training_utils.weight_case_losses import build_case_loss_weights     # ← same helper

def compute_test_losses(model, X_test, y_test, config):
    """
    Returns
    -------
    net_test_loss   : float   # Σ (weighted per‑case MSE)
    avg_case_loss   : float   # mean over cases
    """
    device     = next(model.parameters()).device
    criterion  = nn.MSELoss(reduction="mean")          # per‑batch mean
    batch_size = config['training'].get('batch_size', 1024)

    # ----- per‑case weights (1/<p>², 1/<k>², or 1) -----------------------
    #case_weights = build_case_loss_weights(config, y_test, device)   # tensor (n_cases,)

    net_loss = 0.0
    model.eval()
    with torch.no_grad():
        for idx, (x_case, y_case) in enumerate(zip(X_test, y_test)):
            #w_case   = case_weights[idx]               # 0‑D tensor on device
            mse_sum  = 0.0                             # Σ(batch_mse * bs)
            n_points = 0                               # Σ batch_size

            loader = DataLoader(TensorDataset(x_case, y_case),
                                batch_size=batch_size, shuffle=False)

            for x_b, y_b in loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                batch_mse = criterion(model(x_b), y_b)   # scalar
                bs        = x_b.size(0)
                mse_sum  += batch_mse.item() * bs        # accumulate un‑weighted sum
                n_points += bs

            case_mse   = mse_sum / n_points
            case_loss  = case_mse    # weighted, for logging
            net_loss  += case_loss

    avg_case_loss = net_loss / len(X_test)               # equal per‑case mean
    return net_loss, avg_case_loss
