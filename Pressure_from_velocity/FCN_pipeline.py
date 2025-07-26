#!/usr/bin/env python
# call notes
###python pipeline_fcn_grad_sweep.py --batch 0
##python pipeline_fcn_grad_sweep.py --batch 1
...
# pipeline_fcn_grad_sweep.py
import os
import yaml
import subprocess
from itertools import product, islice
import pandas as pd

CONFIG_TEMPLATE = "Pressure_from_velocity/configs/nn_config.yaml"
OUTPUT_ROOT = "Pressure_from_velocity/outputs/FCN_single_v3"
SUMMARY_FILE = "fcn_validation_summary.csv"
SBATCH_SCRIPT = "run_fcn.slurm"
trials_names = ['single_bump', 'single_phill']
# Sweep parameters
from Models.FCN_architectures import architectures
architectures = architectures['x20']
MAX_JOBS_PER_BATCH = 30
activations = ['leaky_relu']
learning_rates = [0.05]
loss_mode = ''
norm_types = ['minmax']
feature_set = ['Ux', 'Uy', 'dUx_dx', 'dUx_dy', 'dUy_dx', 'dUy_dy']
epochs = 4000
dropout = 0.05
scheduler = True
sched_type = 'reduce'
sched_gamma = 0.8
# this is for linear sched
sched_start = 0.1 * epochs
sched_end_pct = 0.01
sched_lin = [sched_start, sched_end_pct]
#this sched step is for step LR
step_every = 300
step_gamma = 0.15
sched_step = [step_every, step_gamma]
#these factors are for reduce on plateu
sched_factor = 0.8
sched_patience = 70
sched_minlr = float(5e-7)
sched_cooldown = 50

def update_config_file(config_path, updated_config):
    with open(config_path, 'w') as f:
        yaml.dump(updated_config, f)

def generate_all_jobs():
    jobs = []
    for activation, trial, norm, (arch_name, shape), lr in product(
                             activations, trials_names, norm_types,
                            architectures.items(), learning_rates):


        name    = f"FCN_{trial}_{norm}_{arch_name}_{activation}_lr{lr}"
        out_dir = os.path.join(OUTPUT_ROOT, f"{trial}_{norm}", activation, arch_name)

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
        cfg_md = config['model']
        cfg_md['type']       = "FCN"
        cfg_md['layers'] = shape[1]
        cfg_md['width'] = shape[0]
        cfg_md['activation'] = activation
        cfg_tr = config['training']
        cfg_tr['epochs']  = epochs
        cfg_tr['norm'] = norm
        cfg_tr['lr'] = lr
        cfg_tr['loss_mode'] = loss_mode
        cfg_scd = cfg_tr['scheduler']
        cfg_scd['enabled'] = scheduler
        cfg_scd['type'] = sched_type
        if sched_type.lower() == 'reduce':
            cfg_scd['factor'] = sched_factor
            cfg_scd['patience'] = sched_patience
            cfg_scd['min_lr'] = sched_minlr
            cfg_scd['cooldown'] = sched_cooldown
        elif sched_type.lower() == 'step':
            cfg_scd['step_size'] = sched_step[0]
            cfg_scd['gamma'] = sched_step[1]
        elif sched_type.lower() == 'linear':
            cfg_scd['step_size'] = sched_lin

        job_config_path = os.path.join("Pressure_from_velocity/configs", f"temp_{name}.yaml")
        os.makedirs(out_dir, exist_ok=True)
        update_config_file(job_config_path, config)
        jobs.append(job_config_path)

    return jobs

def run_job_batch(job_paths):
    for job_config_path in job_paths:
        name = os.path.basename(job_config_path).replace("temp_", "").replace(".yaml", "")
        print(f"[INFO] Submitting FCN job: {name}")
        subprocess.run(["sbatch", f"--export=ALL,CONFIG_PATH={job_config_path}", SBATCH_SCRIPT], check=True)

def run_batch(batch_idx):
    all_jobs = generate_all_jobs()
    start = batch_idx * MAX_JOBS_PER_BATCH
    end = start + MAX_JOBS_PER_BATCH
    selected_jobs = list(islice(all_jobs, start, end))
    run_job_batch(selected_jobs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=0, help="Which batch index to run (0-based)")
    args = parser.parse_args()

    print(f"[INFO] Submitting batch {args.batch} of FCN validation jobs...", flush=True)
    run_batch(args.batch)
    print("[INFO] To summarize results later, run summarize_results() manually or as a separate job.", flush=True)
