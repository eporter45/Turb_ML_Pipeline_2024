#!/bin/bash
#SBATCH --job-name=fcn_job
#SBATCH --partition=gpu
#SBATCH --nodelist=oat-01,oat-02,oat-06,oat-07
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-user=eporter4@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Activate virtual environment
source ~/Code/venvs/venv_gpu/bin/activate

# Set PYTHONPATH so it can find local modules like Models/
export PYTHONPATH=$PYTHONPATH:/home/users/eporter4/Code

# Run your simulation
python Pressure_from_velocity/Simulation.py --config "$CONFIG_PATH"
