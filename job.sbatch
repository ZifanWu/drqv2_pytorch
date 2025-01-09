#!/bin/bash

#SBATCH --mem=64g
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4    # <- match to OMP_NUM_THREADS
#SBATCH --partition=soc-gpu-np
#SBATCH --account=soc-gpu-np    # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=drqv2
#SBATCH --time=12:00:00      # hh:mm:ss for the job
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --output=./slurm_output/%x_%A_%a.out
#SBATCH --error=./slurm_output/%x_%A_%a.err
### GPU options ###
##SBATCH --gpus-per-node=1
#SBATCH --gres=gpu

#SBATCH --array=0-4


# module load deeplearning/2023.3;
source activate /uufs/chpc.utah.edu/common/home/u1520755/miniconda3/envs/drqv2
# SCRDIR=/scratch/general/<file-system>/$USER/$SLURM_JOB_ID
# mkdir -p $SCRDIR
# cd $SCRDIR

# copy data to scratch (if needed)
# cp <input-files> $SCRDIR

python train.py index=${SLURM_ARRAY_TASK_ID}
