#!/bin/bash

#SBATCH -w lambda4        # specify the machine to run on
#SBATCH -c 2              # specify the number of CPUs
#SBATCH --gres=gpu:1      # specify the number of GPUs

# Navigate to your 'project' folder
cd /home/hay.e/DeepLearning/Project/mini_project/project

# Run the Python script
python3 main.py
