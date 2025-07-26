import os
import yaml
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from utils.make_accuracies_pressure import make_abs_diff_metrics, save_metrics_summary
from Plotting.plot_pressure import plot_all_cases
from Plotting.plot_losses import plot_losses
from Preprocessing.load_data_pressure import make_grids_dict

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "outputs"
DATA_PATH = SCRIPT_DIR.parent / "Data"

# Load grid info once
shared_grids = {
    "interp": make_grids_dict(DATA_PATH, ['BUMP_h42']),
    "extrap": make_grids_dict(DATA_PATH, ['BUMP_h31'])
}

def postprocess_simulation(output_dir, grid_dicts=None):
    output_dir = Path(output_dir)
    pred_dir = output_dir / "predictions"

    print(f"[INFO] Starting postprocessing in {output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    y_pred = torch.load(pred_dir / "y_pred.pt", map_location=device)
    y_test = torch.load(pred_dir / "y_test.pt", map_location=device)

    with open(output_dir / "config_used.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(pred_dir / "test_cases.txt", "r") as f:
        test_cases = [line.strip() for line in f]

    metrics = make_abs_diff_metrics(y_pred, y_test, device=device)
    save_metrics_summary(metrics, output_dir / "accuracy_metrics.txt")

    # Plot predictions
    plot_dir = pred_dir / "plots"
    plot_all_cases(y_test, y_pred, test_cases, plot_dir, grid_dicts=grid_dicts)

    # Plot training loss if available
    train_loss_path = output_dir / "final_model" / "train_loss.pt"
    if train_loss_path.exists():
        train_loss = torch.load(train_loss_path, map_location='cpu')
        fig = plot_losses(train_loss, title=config['paths']['name'], testortrain='Train')
        fig.savefig(output_dir / "train_loss_postprocessed.png", dpi=150)
        plt.close(fig)
        print(f"[INFO] Saved postprocessed training loss plot.")
    else:
        print("[WARNING] No training loss to plot.")

    print(f"[INFO] Finished postprocessing for {output_dir}")

def find_test_dirs(root_dir):
    test_dirs = []
    for pressure_dir in os.listdir(root_dir):
        pressure_path = os.path.join(root_dir, pressure_dir)
        if not os.path.isdir(pressure_path):
            continue
        for model_type in os.listdir(pressure_path):  # old_FCN / old_FCN_scheduler
            model_path = os.path.join(pressure_path, model_type)
            if not os.path.isdir(model_path):
                continue
            for test_run in os.listdir(model_path):  # test_1, test_2, ...
                test_path = os.path.join(model_path, test_run)
                if os.path.isdir(test_path) and test_run.startswith("test_"):
                    test_dirs.append(test_path)
    return test_dirs


from Preprocessing.load_data_pressure import make_grids_dict

DATA_PATH = Path(__file__).resolve().parent.parent / "Data"


def run_all_postprocessing():
    test_dirs = find_test_dirs(ROOT_DIR)
    print(f"[INFO] Found {len(test_dirs)} test directories.")
    for test_dir in test_dirs:
        print(f"[INFO] Processing: {test_dir}")

        # Read test_cases.txt to determine the exact grid(s) needed
        test_cases_file = Path(test_dir) / "predictions" / "test_cases.txt"
        if not test_cases_file.exists():
            print(f"[WARNING] No test_cases.txt found in {test_dir}, skipping...")
            continue

        with open(test_cases_file, "r") as f:
            test_cases = [line.strip() for line in f if line.strip()]

        # Load only the grids relevant to these test cases
        grid_dicts = make_grids_dict(DATA_PATH, test_cases)

        # Postprocess with the right grid dict
        postprocess_simulation(test_dir, grid_dicts=grid_dicts)


if __name__ == "__main__":
    run_all_postprocessing()
