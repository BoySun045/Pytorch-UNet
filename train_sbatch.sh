#!/bin/bash -l

#SBATCH --job-name=gen3c_multi_mono
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=48g
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle
module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_train_env

python3 train.py --amp --epochs 1000 -b 16 -s 0.67 -c 7 -v 10 -l 5e-5 -rw 8.0 --head_mode df_seg # --dataset_name gen3c_single

# mono depth 
# python3 train.py --amp --epochs 500 -b 16 -s 0.67 -c 7 -v 10 -l 5e-5 -rw 8.0 -umd --head_mode df_seg --dataset_name gen3c_multi