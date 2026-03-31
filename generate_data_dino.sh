#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/gen3c_env

# 1. Assign the first argument to ROOT_DIR, default to current dir if empty
ROOT_DIR="${1:-.}" 
T="${2:-48}"

start_time=$(date +%s)

# 2. Use "$ROOT_DIR"/* (with the $ sign and double quotes)
for subdir in "$ROOT_DIR"/video/*; do
    # Check if it is a directory
    dir_name=$(basename "$subdir")
    if [[ -d "$subdir" && "$dir_name" != "depth" && "$dir_name" != "img" && "$dir_name" != "weighted_mask" && "$dir_name" != "video" ]]; then
        echo "Processing: $subdir"
        bash /cluster/project/cvg/students/shangwu/GEN3C/utility/multi_T_variance/run.sh multi 48 forward "$subdir/outputs" "$subdir" intersection dino priority
        bash /cluster/project/cvg/students/shangwu/GEN3C/utility/multi_T_variance/run.sh multi 48 backward "$subdir/outputs" "$subdir" intersection dino priority
        python3 /cluster/project/cvg/students/shangwu/GEN3C/utility/post_process/post_process.py --camera_npz "$subdir/result_0_0/camera_data.npz" --data_root "$subdir" --T 48
        rm "$subdir/outputs"/*.npz
    fi
done

end_time=$(date +%s)
cost_time=$((end_time - start_time))

printf "total time: %02d:%02d:%02d\n" $((cost_time/3600)) $((cost_time%3600/60)) $((cost_time%60))
