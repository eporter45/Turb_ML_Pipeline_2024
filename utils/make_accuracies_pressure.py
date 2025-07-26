import torch
import numpy as np
import os

def make_abs_diff_metrics(predictions, y_test, device):
    all_diffs   = []
    preds_list  = []
    truths_list = []

    for y_p, y_t in zip(predictions, y_test):
        try:
            y_p = y_p.to(device)
            y_t = y_t.to(device)

            arr_p = y_p.cpu().numpy()
            arr_t = y_t.cpu().numpy()

            # absolute error
            abs_diff = np.abs(arr_p - arr_t)
            all_diffs.append(abs_diff)

            # store for corrcoef/net‐accuracy
            preds_list.append(arr_p)
            truths_list.append(arr_t)

        except RuntimeError as e:
            print(f"[Warning] Tensor mismatch or device issue: {e}")
            continue

    if not all_diffs:
        return {
            "sum_abs_diff": None,
            "mean_per_point": None,
            "sum_per_component": None,
            "avg_per_component": None,
            "median_of_points": None,
            "correlation_coefficient": None,
            "net_accuracy": None,
        }

    # concatenate over all cases
    diffs  = np.concatenate(all_diffs, axis=0)
    preds  = np.concatenate(preds_list,  axis=0).flatten()
    truths = np.concatenate(truths_list, axis=0).flatten()

    sum_abs_true = np.sum(np.abs(truths))

    # Pearson r
    corrcoef = (
        np.corrcoef(preds, truths)[0,1]
        if preds.size and truths.size else np.nan
    )

    # net accuracy = 1 − (sum_abs_diff / sum_abs_true)
    net_acc = (
        1.0 - (np.sum(diffs) / sum_abs_true)
        if sum_abs_true != 0 else np.nan
    )

    return {
        "sum_abs_diff":        float(np.sum(diffs)),
        "mean_per_point":      float(np.mean(diffs)),
        "sum_per_component":   np.sum(diffs, axis=0).tolist(),
        "avg_per_component":   np.mean(diffs, axis=0).tolist(),
        "median_of_points":    np.median(diffs, axis=0).tolist(),
        "correlation_coefficient": float(corrcoef),
        "net_accuracy":            float(net_acc)
    }

def save_metrics_summary(metrics, output_dir, model_name="model"):
    os.makedirs(output_dir, exist_ok=True)
    summary_file = os.path.join(output_dir, f"{model_name}_metrics.txt")

    with open(summary_file, "w") as f:
        f.write("=== Model Metrics Summary ===\n\n")
        for key, value in metrics.items():
            if value is None:
                f.write(f"{key}: None\n\n")
            elif isinstance(value, (list, tuple, np.ndarray)):
                formatted = ", ".join(f"{v:.6f}" for v in value)
                f.write(f"{key}: [{formatted}]\n\n")
            else:
                f.write(f"{key}: {value:.6f}\n\n")

    print(f"[INFO] Saved metrics summary to {summary_file}")
