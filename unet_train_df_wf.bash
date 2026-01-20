#!/bin/bash


sbatch --time=24:00:00 -A ls_polle --ntasks=1 --nodes=1 --cpus-per-task=16 --mem-per-cpu=8G --gpus=rtx_4090:1 --gres=gpumem:24g --wrap \
"python3 /cluster/project/cvg/boysun/Unet-refine/train.py --amp --epochs 100 -b 64 -s 0.67 -c 11 -v 10 -l 5e-5 -rw 8.0 -ud --head_mode df_seg"

