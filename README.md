# FrontierNet training code 


### Adapt from https://github.com/milesial/Pytorch-UNet

### Install

Install dependencies
```bash
pip install -r requirements.txt
```

### This repo trains the WM-FrontierNet given the extracted DiT features and videos.
This inludes:

1. IDP generation
2. depth48 estimation
3. IDP reprojection
4. 3D clustering
5. Model training

### Training

Train both distance field and info gain segmentation head (recommanded): 

```console
python3 /cluster/project/cvg/boysun/Pytorch-UNet-latest/train.py --amp --epochs 100 -b 16 -s 0.67 -c 7 -v 10 -l 5e-5 -rw 8.0 -ud --head_mode df_seg
```

Notes:

- Best performance comes from "df_seg" mode.
- Change the "dir_path" in train.py to your dataset folder. 
- Check the args setup in train.py for details about the arguments. 
- Change the number of classes (-c in args) to fit your segmentation setup, and provide your corresponding "multi_class_weights_path" in train.py, which gives threshold for each class. 
- Data Preprocessing always does crop and rescale - the input to the model is always rescaled squared image.
- RGBD as input gives much better results (i.e. with -ud), if you need to train with RGB only then remove the arugment, might have some unknown issues. 

### Data Preprocessing Pipeline

The three scripts under `scripts/` prepare training data from raw DiT features and videos. Run them in order.

**Expected inputs:**
- `dit_features/` — DiT feature `.npz` files (named `seed_86_Actmap_MH3D_*.npz`)
- `videos/` — per-sample video directories (named `Actmap_MH3D_*`), each containing `result_0_0/video_86.mp4`
- An RGB image dataset with a flat `dataset_shard_*/` layout (passed via `--rgb_root`)

---

**Step 1 — Calculate IDP maps** (`scripts/calculate_idp.py`)

Computes the Iterative Denoising Progress (IDP) map for each DiT feature file and saves it as a `(H, W)` float32 array.

```bash
python scripts/calculate_idp.py \
    --npz_dir  dit_features \
    --out_dir  $SCRATCH/cache_idp_features \
    --layer    27 \
    --frame    6 \
    --total_steps 8 \
    --out_hw   544 720
```

Optionally add `--vis_dir $SCRATCH/cache_idp_features_vis` to save side-by-side IDP + RGB visualisations, or `--rgb_root /path/to/dataset_all` to include the RGB panel.

---

**Step 2 — Project IDP to 3D and extract frontier maps** (`scripts/project_3d_idp.py`)

Runs UniK3D depth estimation on each RGB frame, computes a forward-projection disocclusion mask, splits the IDP into covered/disoccluded regions, then extracts the frontier (far disoccluded pixels) using the depth of frame 48 from the corresponding video.

```bash
python scripts/project_3d_idp.py \
    --cache_dir  $SCRATCH/cache_idp_features \
    --rgb_root   /path/to/dataset_all \
    --video_root videos \
    --out_dir    $SCRATCH/disocclusion_output \
    --forward_dist      1.0 \
    --near_depth_thresh 0.5
```

Output layout under `$SCRATCH/disocclusion_output/`:
```
masks/       <stem>_mask.npy           bool  (H,W)
covered/     <stem>_covered.npy        float (H,W)
disoccluded/ <stem>_disoccluded.npy    float (H,W)
frontier/    <stem>_frontier.npy       float (H,W)   ← input to step 3
             <stem>_depth48.npy        float (H,W)   ← input to step 3
             <stem>_frontier.png
vis/         <stem>_vis.png
```

Add `--regen_vis` to force-regenerate visualisations for already-processed samples.

---

**Step 3 — Fit GMMs on frontier maps** (`scripts/fit_gmm.py`)

Samples pixel (or 3D camera-space) coordinates from each frontier map weighted by activation, fits a Bayesian Dirichlet-Process GMM, and saves the active component centres.

```bash
python scripts/fit_gmm.py \
    --data_dir     $SCRATCH/disocclusion_output/frontier \
    --out_dir      $SCRATCH/gmm_output \
    --k_max        10 \
    --n_samples    5000 \
    --weight_thresh 0.02
```

If `<stem>_depth48.npy` is present alongside the frontier file the GMM is fitted in 3D camera space and centres are projected back to pixel coordinates for visualisation. Output is `gmm_centres.npz` (keys: `stems`, `centres`) plus per-sample `<stem>_gmm.png` visualisations.

---

### Dataset Generation/Transformation

- Check dataset/data_gen.py as an example. 
- It refine the binary frontier mask with depth discontinuity. Modify the "get_frontier_line_mask()" function and remove this refinment if you won't need it. 

