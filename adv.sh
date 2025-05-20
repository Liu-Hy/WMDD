#!/bin/bash
#SBATCH --job-name="1001"
#SBATCH --output="stdout/1001.log"
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # could be 1 for py-torch
#SBATCH --cpus-per-task=64   # spread out to use 1 core per numa, set to 64 if tasks is 1
#SBATCH --constraint="scratch"
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bcac-delta-gpu
#SBATCH --exclusive  # dedicated node for this job
#SBATCH --no-requeue
#SBATCH -t 24:00:00

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
module purge
module reset  # load the default Delta modules

source activate hl
bash run.sh -x 1 -y 1 -d tiny-imagenet -b 10.0 -p -C -h 3.0 -l 100 -r '/home/hl57/data'
