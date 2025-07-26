import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

def initialize_scheduler(optimizer, config):
    """
    Builds a learning‑rate scheduler from the YAML config.

    Supported schedulers
    --------------------
    type: "step"
        step_size: int
        gamma:     float
    type: "multistep"
        step_size: [milestone1, milestone2, ...]   (list[int])
        gamma:     float
    type: "linear"
        step_size: [start_epoch, target_pct]       (list[ int, float ])
        gamma:     float   # ignored
    type: "reduce_on_plateau"
        factor: float
        patience: int
        min_lr: float
        cooldown: int
        verbose: bool
    type: "none" | "off" | "false"
        → returns None
    """
    sch_cfg  = config["training"]["scheduler"]
    s_type   = sch_cfg["type"].lower()
    step_sz  = sch_cfg.get("step_size", None)
    gamma    = sch_cfg.get("gamma", 0.1)

    # ───── no scheduler ────────────────────────────────────────────────
    if s_type in ("none", "off", "false", ''):
        return None

    # ───── StepLR ──────────────────────────────────────────────────────
    if s_type in ("step", "steplr", "lrstep"):
        if not isinstance(step_sz, int):
            raise TypeError("'step_size' must be an int for StepLR.")
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_sz, gamma=gamma)

    # ───── MultiStepLR ────────────────────────────────────────────────
    if s_type == "multistep":
        if not (isinstance(step_sz, (list, tuple)) and all(isinstance(x, int) for x in step_sz)):
            raise TypeError("'step_size' must be a list of ints for MultiStepLR.")
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=list(step_sz), gamma=gamma)

    # ───── ReduceLROnPlateau ───────────────────────────────────────────
    if s_type in ("reduce_on_plateau", "reduce"):
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=float(sch_cfg["factor"]),
            patience=sch_cfg["patience"],
            min_lr=float(sch_cfg["min_lr"]),
            cooldown=int(sch_cfg["cooldown"])
            #verbose=sch_cfg.get('verbose', True)
        )

    # ───── Linear decay ───────────────────────────────────────────────
    if s_type == "linear":
        if not (isinstance(step_sz, (list, tuple)) and len(step_sz) == 2):
            raise TypeError("'step_size' must be [start_epoch, target_pct] for Linear scheduler.")
        start_epoch, target_pct = step_sz
        if not isinstance(start_epoch, int):
            raise TypeError("Linear scheduler 'start_epoch' must be an int.")
        if not (0.0 < target_pct <= 1.0):
            raise ValueError("'target_pct' must be in (0, 1].")

        total_epochs = config["training"]["epochs"]
        decay_steps  = total_epochs - start_epoch
        slope = (1.0 - target_pct) / decay_steps    # positive

        def lr_lambda(epoch):
            if epoch < start_epoch:
                return 1.0
            return 1.0 - slope * (epoch - start_epoch)

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ───── unsupported scheduler type ─────────────────────────────────
    raise ValueError(f"Unsupported scheduler type '{s_type}'. "
                     "Use 'none', 'step', 'multistep', 'linear', or 'reduce_on_plateau'.")
