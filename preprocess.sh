#!/bin/bash -l

#SBATCH --job-name=preprocess
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --gpus=rtx_4090:1
#SBATCH --account=ls_polle
#SBATCH --time=48:00:00

module load eth_proxy

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /cluster/project/cvg/students/shangwu/ftnet_train_env

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT=/cluster/project/cvg/students/shangwu/Pytorch-UNet
FEATURES_DIR=/cluster/project/cvg/students/shangwu/GEN3C/features_analysis
RGB_ROOT=/cluster/project/cvg/students/shangwu/GEN3C/assets/diffusion/dataset_all

cd "$PROJECT"

# $SCRATCH is set automatically on the ETH cluster per-job
IDP_DIR=$SCRATCH/cache_idp_features_100
DISOC_DIR=$SCRATCH/disocclusion_output_100
GMM_DIR=$SCRATCH/gmm_output_100

echo "starting preprocessing with IDP_DIR=$IDP_DIR, DISOC_DIR=$DISOC_DIR, GMM_DIR=$GMM_DIR"
# ── Step 1: Calculate IDP maps ───────────────────────────────────────────────
python scripts/calculate_idp.py \
    --npz_dir      "$FEATURES_DIR/filtered_dit_features_100" \
    --out_dir      "$IDP_DIR" \
    --layer        27 \
    --frame        6 \
    --total_steps  8 \
    --out_hw       544 720
echo "IDP calculation completed, output saved to $IDP_DIR"
# ── Step 2: Project IDP to 3D and extract frontier maps ──────────────────────
python scripts/project_3d_idp.py \
    --cache_dir         "$IDP_DIR" \
    --video_root        "$FEATURES_DIR/outputs" \
    --out_dir           "$DISOC_DIR" \
    --forward_dist      1.0 \
    --near_depth_thresh 0.5 \
    --depth_threshold   0.5
echo "3D projection and frontier extraction completed, output saved to $DISOC_DIR"
# ── Step 3: Fit GMMs on frontier maps ────────────────────────────────────────
python scripts/fit_gmm.py \
    --data_dir      "$DISOC_DIR/frontier" \
    --out_dir       "$GMM_DIR" \
    --k_max         8 \
    --n_samples     5000 \
    --weight_thresh 0.02
echo "GMM fitting completed, output saved to $GMM_DIR"