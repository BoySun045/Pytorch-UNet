"""
calculate_idp.py

Generate IDP maps from DiT feature .npz files and save them
to $SCRATCH/cache_idp_features/ (or --out_dir).

Values are the sum of weighted cosine-distance progress over diffusion steps
0→6, comparable across images.

Usage:
    python calculate_idp.py \
        [--npz_dir  /path/to/dit_features] \
        [--out_dir  $SCRATCH/cache_idp_features] \
        [--vis_dir  $SCRATCH/cache_idp_features_vis] \
        [--layer    27] \
        [--frame    6] \
        [--total_steps 8] \
        [--alpha    1.0] \
        [--out_hw   544 720]
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

def _cosine_dist_upsampled(
    layer_data: np.ndarray,
    t: int,
    ref_t: int,
    frame: int,
    out_hw: Tuple[int, int],
) -> np.ndarray:
    """Cosine distance between timestep t and ref_t at a given frame, upsampled to out_hw."""
    x_t = torch.from_numpy(layer_data[t, frame].squeeze(2)).float()
    x_ref = torch.from_numpy(layer_data[ref_t, frame].squeeze(2)).float()
    cos_dist = 1.0 - F.cosine_similarity(x_t, x_ref, dim=-1, eps=1e-8)
    upsampled = F.interpolate(cos_dist[None, None], size=out_hw, mode="bilinear", align_corners=False)
    return upsampled[0, 0].cpu().numpy().astype(np.float32)

def compute_idp(
    layer_data: np.ndarray,
    frame: int,
    total_steps: int,
    alpha: float,
    out_hw: tuple,
) -> np.ndarray:
    """Compute IDP as sum of weighted cosine-distance progress over diffusion steps."""
    if total_steps >= layer_data.shape[0]:
        total_steps = layer_data.shape[0] - 1
    if total_steps < 2:
        raise ValueError(f"Need at least 3 timesteps, got {layer_data.shape[0]}")

    ref_t = total_steps
    deltas = np.stack(
        [_cosine_dist_upsampled(layer_data, t=t, ref_t=ref_t, frame=frame, out_hw=out_hw)
         for t in range(total_steps)],
        axis=0,
    )  # [T, H, W]

    progress = deltas[:-1] - deltas[1:]
    weights = ((np.arange(total_steps - 1, 0, -1) / (total_steps - 1)) ** alpha).reshape(-1, 1, 1)
    weighted_progress = np.maximum(progress, 0.0) * weights

    idp = np.sum(weighted_progress[:6], axis=0)
    return idp.astype(np.float32)                  # [H, W]


def build_rgb_index(rgb_root: str) -> dict:
    index = {}
    for shard in sorted(os.listdir(rgb_root)):
        shard_dir = os.path.join(rgb_root, shard)
        if not os.path.isdir(shard_dir):
            continue
        for fname in os.listdir(shard_dir):
            if fname.lower().endswith((".jpg", ".png")):
                stem = os.path.splitext(fname)[0]
                index[stem] = os.path.join(shard_dir, fname)
    return index


def save_vis(idp: np.ndarray, stem: str, vis_dir: str,
             rgb_path: str = None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arr = idp.squeeze()          # (H, W)

    n_panels = 2 if rgb_path is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(stem, fontsize=9)

    col = 0
    if rgb_path is not None:
        rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
        axes[col].imshow(rgb)
        axes[col].set_title("RGB", fontsize=11)
        axes[col].axis("off")
        col += 1

    im0 = axes[col].imshow(arr, cmap="hot", vmin=arr.min(), vmax=arr.max())
    axes[col].set_title("IDP", fontsize=11)
    plt.colorbar(im0, ax=axes[col], fraction=0.046, pad=0.04, label="IDP")
    axes[col].axis("off")

    plt.tight_layout()
    out_path = os.path.join(vis_dir, f"{stem}_idp.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def stem_from_npz(npz_path: str) -> str:
    stem = os.path.splitext(os.path.basename(npz_path))[0]
    # Strip leading "seed_NN_" prefix to match existing cache naming
    if stem.startswith("seed_"):
        parts = stem.split("_", 2)
        stem = parts[2] if len(parts) >= 3 else stem
    return stem


def main():
    default_npz_dir = os.path.join(os.path.dirname(__file__), "..", "dit_features")
    default_out_dir = os.path.join(os.environ.get("SCRATCH", "/tmp"), "cache_idp_features")

    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir",      default=default_npz_dir)
    parser.add_argument("--out_dir",      default=default_out_dir)
    parser.add_argument("--vis_dir",      default=None,
                        help="If set, save side-by-side PNG visualizations here")
    parser.add_argument("--vis_only",     action="store_true",
                        help="Skip .npz processing; visualize existing .npy files in --out_dir")
    parser.add_argument("--rgb_root",     default=None,
                        help="Root of dataset_shard_* dirs for RGB images (adds RGB panel to vis)")
    parser.add_argument("--layer",        type=int,   default=27)
    parser.add_argument("--frame",        type=int,   default=6)
    parser.add_argument("--total_steps",  type=int,   default=8)
    parser.add_argument("--alpha",        type=float, default=1.0)
    parser.add_argument("--out_hw",       type=int,   nargs=2, default=[544, 720],
                        metavar=("H", "W"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = args.vis_dir or (args.out_dir + "_vis")
    os.makedirs(vis_dir, exist_ok=True)
    out_hw = tuple(args.out_hw)
    key = f"cond:blocks.block{args.layer}.blocks.2.block.layer2"

    rgb_index = {}
    if args.rgb_root:
        print(f"Indexing RGB images in {args.rgb_root} ...")
        rgb_index = build_rgb_index(args.rgb_root)
        print(f"Indexed {len(rgb_index)} RGB images")

    # ------------------------------------------------------------------
    # vis_only: visualize existing .npy files in out_dir
    # ------------------------------------------------------------------
    if args.vis_only:
        npy_files = sorted(
            os.path.join(args.out_dir, f)
            for f in os.listdir(args.out_dir)
            if f.endswith(".npy")
        )
        print(f"Found {len(npy_files)} .npy files in {args.out_dir}")
        print(f"Visualizations → {vis_dir}")
        n_done = n_skipped = n_error = 0
        pbar = tqdm(npy_files, unit="file")
        for npy_path in pbar:
            stem = os.path.splitext(os.path.basename(npy_path))[0]
            vis_path = os.path.join(vis_dir, f"{stem}_idp.png")
            if os.path.exists(vis_path):
                n_skipped += 1
                pbar.set_postfix(done=n_done, skipped=n_skipped, errors=n_error)
                continue
            try:
                idp = np.load(npy_path)
                save_vis(idp, stem, vis_dir, rgb_path=rgb_index.get(stem))
                n_done += 1
            except Exception as e:
                tqdm.write(f"ERROR {stem}: {e}")
                n_error += 1
            pbar.set_postfix(done=n_done, skipped=n_skipped, errors=n_error)
        print(f"\nFinished. done={n_done}  skipped={n_skipped}  errors={n_error}")
        return

    # ------------------------------------------------------------------
    # Normal mode: generate .npy from .npz (and optionally visualize)
    # ------------------------------------------------------------------
    npz_files = sorted(
        os.path.join(args.npz_dir, f)
        for f in os.listdir(args.npz_dir)
        if f.endswith(".npz")
    )
    print(f"Found {len(npz_files)} .npz files in {args.npz_dir}")
    print(f"Output → {args.out_dir}")
    if args.vis_dir:
        print(f"Visualizations → {vis_dir}")

    n_done = n_skipped = n_error = 0
    pbar = tqdm(npz_files, unit="file")
    for npz_path in pbar:
        stem = stem_from_npz(npz_path)
        out_path = os.path.join(args.out_dir, f"{stem}.npy")

        if os.path.exists(out_path):
            n_skipped += 1
            pbar.set_postfix(done=n_done, skipped=n_skipped, errors=n_error)
            continue

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                if key not in data:
                    raise KeyError(f"Key '{key}' not found")
                layer_data = data[key]
            idp = compute_idp(layer_data, args.frame, args.total_steps,
                              args.alpha, out_hw)
            np.save(out_path, idp)
            if args.vis_dir:
                save_vis(idp, stem, vis_dir, rgb_path=rgb_index.get(stem))
            n_done += 1
        except Exception as e:
            tqdm.write(f"ERROR {stem}: {e}")
            n_error += 1
            continue

        pbar.set_postfix(done=n_done, skipped=n_skipped, errors=n_error)

    print(f"\nFinished. done={n_done}  skipped={n_skipped}  errors={n_error}")


if __name__ == "__main__":
    main()
