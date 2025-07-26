#!/bin/bash
# Script to run the Pressure_from_velocity simulation locally with PYTHONPATH set

export PYTHONPATH=.
python Pressure_from_velocity/Simulation.py --config Pressure_from_velocity/configs/nn_config.yaml
