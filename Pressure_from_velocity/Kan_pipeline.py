#!/usr/bin/env python
# pipeline_kan_grad_sweep.py
import os
import yaml
import subprocess
from itertools import product, islice
import pandas as pd

MAX_JOBS_PER_BATCH = 2


CONFIG_TEMPLATE = "Pressure_from_velocity/configs/kan_config.yaml"
OUTPUT_ROOT = "Pressure_from_velocity/outputs/run2"
SUMMARY_FILE = "kan_validation_summary.csv"
SBATCH_SCRIPT = "run_kan.slurm"
trials_names = ['single_bump', 'single_phill']

# Sweep parameters

from Models.kan_architectures import arch_types
arch_key = "mult3"
architectures = arch_types[arch_key]
# Optional deeper case
learning_rates = [0.01, 0.001, 0.0005]
loss_mode = ''
norm_types = ['minmax', 'abs']
feature_set = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
epochs = 650
scheduler = True
sched_type = 'linear'
sched_step = [int(0.1 * epochs), .1]
sched_gamma = 0.7


def update_config_file(config_path, updated_config):
    with open(config_path, 'w') as f:
        yaml.dump(updated_config, f)

def generate_all_jobs():
    jobs = []
    for trial, norm, (arch_name, shape), lr in product(
                     trials_names, norm_types, architectures.items(), learning_rates):

        name    = f"kan_{trial}_{norm}_{arch_name}_lr{lr}"
        out_dir = os.path.join(OUTPUT_ROOT, f"{trial}_{norm}", arch_name)

        # 1) load the template
        with open(CONFIG_TEMPLATE, 'r') as f:
            config = yaml.safe_load(f)

        # 2) now you can safely read from it
        input_dim  = len(feature_set)
        output_dim = len(config['features']['output'])

        # 3) overwrite whatever you need
        config['features']['input']  = feature_set
        config['features']['output'] = ['p']
        config['trial_name']          = trial
        config['paths']['name']       = name
        config['paths']['output_dir'] = out_dir
        config['model']['type']       = "KAN"
        config['model']['shape'] = [[input_dim, 0]] + shape + [[output_dim, 0]]
        config['training']['norm']    = norm
        config['training']['epochs']  = epochs
        config['training']['lr']      = lr
        config['training']['loss_mode'] = loss_mode
        config['training']['scheduler']['enabled']   = scheduler
        config['training']['scheduler']['type']   = sched_type
        config['training']['scheduler']['step_size'] = sched_step
        config['training']['scheduler']['gamma']     = sched_gamma



        job_config_path = os.path.join("Pressure_from_velocity/configs", f"temp_{name}.yaml")
        os.makedirs(out_dir, exist_ok=True)
        update_config_file(job_config_path, config)
        jobs.append(job_config_path)

    return jobs

def run_job_batch(job_paths):
    for job_config_path in job_paths:
        name = os.path.basename(job_config_path).replace("temp_", "").replace(".yaml", "")
        print(f"[INFO] Submitting KAN job: {name}")
        subprocess.run([
            "sbatch",
            f"--export=CONFIG_PATH={job_config_path}",
            SBATCH_SCRIPT
        ])

def run_batch(batch_idx):
    all_jobs = generate_all_jobs()
    start = batch_idx * MAX_JOBS_PER_BATCH
    end = start + MAX_JOBS_PER_BATCH
    selected_jobs = list(islice(all_jobs, start, end))
    run_job_batch(selected_jobs)

def summarize_results(output_root, summary_file):
    records = []
    for root, dirs, files in os.walk(output_root):
        for name in dirs:
            path = os.path.join(root, name)
            metrics_path = os.path.join(path, "accuracy_metrics.txt")
            try:
                with open(metrics_path, 'r') as f:
                    lines = f.readlines()
                    final_train_loss = float(lines[0].split()[-1])
                    final_test_loss = float(lines[1].split()[-1])

                    metric_vals = {}
                    for line in lines[3:]:
                        if ':' in line:
                            key, val = line.split(":")
                            key = key.strip()
                            val = val.strip().strip('[]')
                            if "," in val:
                                metric_vals[key] = [float(x) for x in val.split(",")]
                            elif val.lower() != 'none':
                                metric_vals[key] = float(val)
                            else:
                                metric_vals[key] = None

                    valid = True
            except Exception:
                final_train_loss, final_test_loss = float('nan'), float('nan')
                valid = False
                metric_vals = {}

            record = {
                "name": name,
                "final_train_loss": final_train_loss,
                "final_test_loss": final_test_loss,
                "valid_run": valid,
                "path_to_output": path
            }
            record.update(metric_vals)
            records.append(record)

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(output_root, summary_file), index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help="Which batch index to run (0-based)")
    args = parser.parse_args()

    print(f"[INFO] Submitting batch {args.batch} of KAN validation jobs...")
    run_batch(args.batch)
    print("[INFO] To summarize results later, run summarize_results() manually or as a separate job.")
