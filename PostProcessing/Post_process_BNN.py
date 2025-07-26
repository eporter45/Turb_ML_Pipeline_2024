from pathlib import Path
import os
import pandas as pd

from pathlib import Path
import os
import pandas as pd

from pathlib import Path
import pandas as pd


def summarize_bnn_results(output_root, summary_filename="bnn_summary.csv"):
    """
    Summarize BNN metrics from model_metrics.txt files.

    Expected directory structure:
    output_root/
        trial_name/
            architecture_name/
                job_name/
                    metrics/model_metrics.txt
    """
    output_root = Path(output_root)
    summary_path = output_root / summary_filename

    records = []

    for trial_dir in output_root.iterdir():
        if not trial_dir.is_dir():
            continue

        for arch_dir in trial_dir.iterdir():
            if not arch_dir.is_dir():
                continue

            for job_dir in arch_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                metrics_path = job_dir / "metrics" / "model_metrics.txt"
                if not metrics_path.exists():
                    continue

                try:
                    with open(metrics_path, 'r') as f:
                        lines = f.readlines()
                        metric_vals = {}
                        for line in lines:
                            if ":" in line:
                                key, val = line.split(":", 1)
                                key = key.strip()
                                val = val.strip().strip("[]")

                                # Parse as list or float
                                if "," in val:
                                    try:
                                        metric_vals[key] = [float(x) for x in val.split(",")]
                                    except:
                                        metric_vals[key] = val
                                else:
                                    try:
                                        metric_vals[key] = float(val)
                                    except:
                                        metric_vals[key] = val

                        valid = True
                        final_train_loss = metric_vals.get("sum_abs_diff", float('nan'))
                        final_test_loss = metric_vals.get("mean_per_point", float('nan'))
                        net_accuracy = metric_vals.get("net_accuracy", float('nan'))

                except Exception:
                    final_train_loss, final_test_loss, net_accuracy = float('nan'), float('nan'), float('nan')
                    valid = False
                    metric_vals = {}

                record = {
                    "trial": trial_dir.name,
                    "architecture": arch_dir.name,
                    "job_name": job_dir.name,
                    "final_train_loss": final_train_loss,
                    "final_test_loss": final_test_loss,
                    "net_accuracy": net_accuracy,
                    "valid_run": valid,
                    "path_to_output": str(job_dir)
                }
                record.update(metric_vals)
                records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(summary_path, index=False)
    return df


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent / "Pressure_from_velocity" / "outputs" / "BNN_trials2"
summarize_bnn_results(ROOT_DIR)
