#!/bin/bash -l
#SBATCH --job-name=gen_data
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --mem-per-cpu=48g
#SBATCH --ntasks=1
#SBATCH --time=48:00:00
#SBATCH --gpus=rtx_4090:1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/gen3c_env

# 1. Assign the first argument to ROOT_DIR, default to current dir if empty
# ROOT_DIR="${1:-.}" 
# T="${2:-48}"
ROOT_DIR="/cluster/project/cvg/students/shangwu/Pytorch-UNet/dummy"
T=48

/cluster/project/cvg/students/shangwu/GEN3C/tmp/result_0_0

# 2. Use "$ROOT_DIR"/* (with the $ sign and double quotes)
for subdir in "$ROOT_DIR"/video/*; do
    echo $subdir
    # Check if it is a directory
    dir_name=$(basename "$subdir")
    if [[ -d "$subdir" && "$dir_name" != "depth" && "$dir_name" != "img" && "$dir_name" != "weighted_mask" && "$dir_name" != "video" && "$dir_name" != "outputs" ]]; then
        echo "Processing: $subdir"
        # bash /cluster/project/cvg/students/shangwu/GEN3C/utility/multi_T_variance/run.sh multi 48 backward "$subdir/outputs_multi" "$subdir" intersection rgb priority 15 1
        python3 /cluster/project/cvg/students/shangwu/GEN3C/utility/post_process/post_process.py --camera_npz "$subdir/result_0_0/camera_data.npz" --data_root "$subdir" --T 48 --multi
        # rm "$subdir/outputs"/*.npz
    fi
done