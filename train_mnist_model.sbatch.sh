#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --gpus-per-task=1
#SBATCH --job-name=vae
#SBATCH --ntasks=1
#SBATCH --output=vae-sbatch-%j.log
#SBATCH --partition=shared

/dgx-storage/c/user_data/jordan/anaconda3/envs/rl_copula_policy_py36/bin/python /dgx-storage/c/user_data/jordan/projects/variational-autoencoder/vae/train_mnist_vae.py

