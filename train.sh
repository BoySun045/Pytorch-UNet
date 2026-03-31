#!/bin/bash

module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_train_env

python3 train.py --amp --epochs 500 -b 16 -s 0.67 -c 7 -v 10 -l 5e-5 -rw 8.0 --head_mode df_seg --use_mono_depth --dataset_name gt