"""
project_3d_idp.py

Batch-process all IDP .npy files in $SCRATCH/cache_idp_features/ and produce
per-sample disocclusion masks + split IDP regions projected into 3D.

Frame 0 of video_86.mp4 is used as the RGB for depth estimation and visualisation.

Output layout under $SCRATCH/disocclusion_output/:
    masks/          <stem>_mask.npy           bool  (H,W)  True=disoccluded
    covered/        <stem>_covered.npy        float (H,W)  IDP at covered pixels
    disoccluded/    <stem>_disoccluded.npy    float (H,W)  IDP at disoccluded pixels
    vis/            <stem>_vis.png            7-panel RGB/IDP/covered/texture/frontier/frame48/depth48
    frontier/       <stem>_frontier.npy       float (H,W)  IDP at frontier pixels
                    <stem>_frontier.png       side-by-side frontier heatmap + frame48
                    <stem>_depth48.npy        float (H,W)  UniK3D depth at frame 48

UniK3D is loaded once and reused across all samples.
Already-processed files are skipped (resume-safe).

Usage:
    python project_3d_idp.py \
        [--cache_dir  $SCRATCH/cache_idp_features] \
        [--out_dir    $SCRATCH/disocclusion_output] \
        [--forward_dist 1.0]
"""

import argparse
import os
import sys

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# Camera constants (fixed for all HM3D exploration runs)
# ---------------------------------------------------------------------------
CAM_H = 544
CAM_W = 720
CAM_F = 300.0
CX    = CAM_W / 2.0 - 0.5   # 359.5
CY    = CAM_H / 2.0 - 0.5   # 271.5
FORWARD_DIST = 1.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def stem_from_npy(npy_path: str) -> str:
    """
    Extract the shared RGB stem from a npy filename.

    e.g. "Actmap_MH3D_00005_437_1.npy" → "Actmap_MH3D_00005_437_1"
    """
    return os.path.splitext(os.path.basename(npy_path))[0]



def load_unik3d_model():
    import torch
    from unik3d.models import UniK3D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UniK3D.from_pretrained("lpiccinelli/unik3d-vitl").to(device)
    model.eval()
    print(f"UniK3D loaded on {device}")
    return model, device


def estimate_depth(rgb: np.ndarray, K: np.ndarray, model, device) -> np.ndarray:
    import torch
    from unik3d.utils.camera import OPENCV  # noqa: F401

    rgb_tensor = (
        torch.from_numpy(rgb.astype(np.float32))
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )
    cam_params = torch.tensor(
        [K[0,0], K[1,1], K[0,2], K[1,2]] + [0.0]*12,
        dtype=torch.float32,
    ).to(device)
    camera = eval("OPENCV")(params=cam_params)
    with torch.no_grad():
        pred = model.infer(rgb_tensor, camera)
    return pred["depth"].squeeze().cpu().numpy().astype(np.float32)


def check_explorable(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    forward_dist: float = 1.0,
    depth_threshold: float = 0.5,
    explorable_ratio: float = 0.70,
) -> tuple[bool, bool]:
    """
    Check whether a scene is safe and explorable after moving forward.

    Returns
    -------
    (explorable, collision)
      collision  : True if any valid pixel in the centre 5%×5% ROI has depth < 1 m
      explorable : False if ≥ explorable_ratio of visible pixels (after forward
                   projection) have depth < depth_threshold
    """
    H, W = depth.shape

    # --- Collision: centre 5%×5% ROI, depth < 1 m ---------------------------
    cy0, cy1 = int(H * 0.475), int(H * 0.525)
    cx0, cx1 = int(W * 0.475), int(W * 0.525)
    roi = depth[cy0:cy1, cx0:cx1]
    roi_valid = roi > 0
    if roi_valid.any() and (roi[roi_valid] < 1.0).any():
        return False, True

    # --- Forward projection: build depth buffer at new camera position -------
    z0    = depth.astype(np.float32)
    z_new = z0 - forward_dist
    valid = (z0 > 0) & (z_new > 0)

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    x_cam = (us - cx) * z0 / fx
    y_cam = (vs - cy) * z0 / fy
    u_new = (fx * x_cam / z_new)[valid]
    v_new = (fy * y_cam / z_new)[valid]
    z_pts = z_new[valid]

    ui = np.round(u_new).astype(np.int32)
    vi = np.round(v_new).astype(np.int32)
    in_bounds = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    ui, vi, z_pts = ui[in_bounds], vi[in_bounds], z_pts[in_bounds]

    # z-buffer: far → near so nearest point wins
    order        = np.argsort(z_pts)[::-1]
    depth_buffer = np.zeros((H, W), dtype=np.float32)
    depth_buffer[vi[order], ui[order]] = z_pts[order]

    # --- Explorable check ----------------------------------------------------
    visible_mask = depth_buffer > 0
    n_visible    = int(visible_mask.sum())
    if n_visible == 0:
        return False, False
    n_near     = int((depth_buffer[visible_mask] < depth_threshold).sum())
    near_ratio = n_near / n_visible
    explorable = near_ratio < explorable_ratio
    return explorable, False


def compute_disocclusion_mask(
    depth: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    forward_dist: float = 1.0,
) -> np.ndarray:
    H, W = depth.shape
    z0  = depth.astype(np.float32)
    z48 = z0 - forward_dist
    valid = (z0 > 0) & (z48 > 0)

    u_grid, v_grid = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    x_cam = (u_grid - cx) * z0 / fx
    y_cam = (v_grid - cy) * z0 / fy
    u48_f = np.where(valid, fx * x_cam / z48 + cx, -2.0)
    v48_f = np.where(valid, fy * y_cam / z48 + cy, -2.0)

    coverage = np.zeros((H, W), dtype=np.uint8)
    for du, dv in ((0, 0), (1, 0), (0, 1), (1, 1)):
        u48_i = np.floor(u48_f).astype(np.int32) + du
        v48_i = np.floor(v48_f).astype(np.int32) + dv
        m = valid & (u48_i >= 0) & (u48_i < W) & (v48_i >= 0) & (v48_i < H)
        coverage[v48_i[m], u48_i[m]] = 1

    coverage = cv2.dilate(coverage, np.ones((3, 3), np.uint8), iterations=1)
    return ~coverage.astype(bool)


def load_video_frame(video_path: str, frame_idx: int):
    """Extract a frame (0-indexed) from a video. Returns RGB ndarray or None."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def save_vis(rgb, actmap, actmap_covered, actmap_disoccluded, out_path,
             frame48=None, depth48=None, near_depth_thresh=0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    H, W = actmap.shape

    # Use a shared IDP scale across all actmap panels so colorbars are comparable
    idp_vmin = float(actmap.min())
    idp_vmax = float(actmap.max())

    # Split disoccluded into texture (depth48 < thresh) and frontier (depth48 >= thresh)
    if depth48 is not None:
        depth48_resized = cv2.resize(depth48, (W, H), interpolation=cv2.INTER_LINEAR)
        near_mask = depth48_resized < near_depth_thresh
        actmap_texture  = np.where( near_mask, actmap_disoccluded, 0.0).astype(np.float32)
        actmap_frontier = np.where(~near_mask, actmap_disoccluded, 0.0).astype(np.float32)
    else:
        actmap_texture  = np.zeros_like(actmap_disoccluded)
        actmap_frontier = actmap_disoccluded

    fig, axes = plt.subplots(1, 7, figsize=(38, 5))

    # panel 0: RGB (no colorbar)
    axes[0].imshow(rgb)
    axes[0].set_title("RGB", fontsize=11)
    axes[0].axis("off")

    # panels 1-4: IDP / Actmap maps — shared scale with colorbar
    idp_panels = [
        (axes[1], actmap,           "Actmap (full)"),
        (axes[2], actmap_covered,   "Covered region"),
        (axes[3], actmap_texture,   f"Texture region\n(depth48 < {near_depth_thresh} m)"),
        (axes[4], actmap_frontier,  f"Frontier region\n(depth48 ≥ {near_depth_thresh} m)"),
    ]
    for ax, data, title in idp_panels:
        im = ax.imshow(data, cmap="hot", vmin=idp_vmin, vmax=idp_vmax)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="IDP")
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    # panel 5: frame 48 RGB
    axes[5].imshow(frame48 if frame48 is not None else np.zeros_like(rgb))
    axes[5].set_title("Frame 48 (video_86)", fontsize=11)
    axes[5].axis("off")

    # panel 6: depth of frame 48
    if depth48 is not None:
        im = axes[6].imshow(depth48, cmap="plasma")
        plt.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04, label="m")
    else:
        axes[6].imshow(np.zeros_like(rgb))
    axes[6].set_title("Depth — Frame 48 (UniK3D)", fontsize=11)
    axes[6].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_frontier(actmap_disoccluded, depth48, H, W, near_depth_thresh):
    """Return actmap_frontier given depth48 (any resolution) and actmap shape (H, W)."""
    if depth48 is None:
        return actmap_disoccluded.copy()
    depth48_resized = cv2.resize(depth48, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.where(depth48_resized >= near_depth_thresh, actmap_disoccluded, 0.0).astype(np.float32)


def save_frontier_png(actmap_frontier, out_path, frame48=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = plt.get_cmap("hot")
    a_norm = (actmap_frontier - actmap_frontier.min()) / (actmap_frontier.max() - actmap_frontier.min() + 1e-8)
    frontier_img = (cm(a_norm)[..., :3] * 255).astype(np.uint8)

    if frame48 is not None:
        H, W = frontier_img.shape[:2]
        frame48_resized = cv2.resize(frame48, (W, H), interpolation=cv2.INTER_LINEAR)
        combined = np.concatenate([frontier_img, frame48_resized], axis=1)
    else:
        combined = frontier_img

    cv2.imwrite(out_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))


def _load_depth48(video_root, stem, model, device):
    """Load frame 48 from video_86.mp4 and estimate its depth. Returns (frame48, depth48)."""
    video_path = os.path.join(video_root, stem, "result_0_0", "video_86.mp4")
    frame48 = load_video_frame(video_path, 48)
    if frame48 is None:
        return None, None
    H48, W48 = frame48.shape[:2]
    K48 = np.array([
        [CAM_F * (W48 / CAM_W), 0, CX * (W48 / CAM_W)],
        [0, CAM_F * (H48 / CAM_H), CY * (H48 / CAM_H)],
        [0, 0, 1],
    ], dtype=np.float64)
    depth48 = estimate_depth(frame48, K48, model, device)
    return frame48, depth48


def process_one(npy_path, out_dirs, forward_dist, model, device,
                video_root, near_depth_thresh=0.25, regen_vis=False,
                depth_threshold=0.5):
    stem = stem_from_npy(npy_path)
    video_path = os.path.join(video_root, stem, "result_0_0", "video_86.mp4")

    mask_path        = os.path.join(out_dirs["masks"],       stem + "_mask.npy")
    covered_path     = os.path.join(out_dirs["covered"],     stem + "_covered.npy")
    disoccluded_path = os.path.join(out_dirs["disoccluded"], stem + "_disoccluded.npy")
    vis_path         = os.path.join(out_dirs["vis"],         stem + "_vis.png")
    frontier_path     = os.path.join(out_dirs["frontier"],    stem + "_frontier.png")
    frontier_npy_path = os.path.join(out_dirs["frontier"],    stem + "_frontier.npy")
    depth48_npy_path  = os.path.join(out_dirs["frontier"],    stem + "_depth48.npy")

    npys_exist = all(os.path.exists(p) for p in (mask_path, covered_path, disoccluded_path))

    # If .npy files exist but vis/frontier is missing (or regen forced), regenerate
    if npys_exist and (not os.path.exists(vis_path) or not os.path.exists(frontier_path) or regen_vis):
        actmap_covered     = np.load(covered_path)
        actmap_disoccluded = np.load(disoccluded_path)
        actmap = actmap_covered + actmap_disoccluded
        H, W = actmap.shape

        rgb = load_video_frame(video_path, 0)
        if rgb is None:
            return "missing_video"
        if rgb.shape[1] != W or rgb.shape[0] != H:
            rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

        frame48, depth48 = _load_depth48(video_root, stem, model, device)
        if frame48 is None:
            print(f"  WARNING: could not read frame 48 for {stem}")

        actmap_frontier = compute_frontier(actmap_disoccluded, depth48, H, W, near_depth_thresh)
        np.save(frontier_npy_path, actmap_frontier)
        if depth48 is not None:
            np.save(depth48_npy_path, depth48)
        save_frontier_png(actmap_frontier, frontier_path, frame48=frame48)
        save_vis(rgb, actmap, actmap_covered, actmap_disoccluded, vis_path,
                 frame48=frame48, depth48=depth48, near_depth_thresh=near_depth_thresh)
        return "done"

    # Skip if everything already exists
    if npys_exist and os.path.exists(vis_path) and os.path.exists(frontier_path):
        return "skipped"

    # Full processing: depth estimation + mask + save all
    actmap = np.load(npy_path)   # (H, W) float32
    H, W = actmap.shape

    fx = CAM_F * (W / CAM_W)
    fy = CAM_F * (H / CAM_H)
    cx = CX    * (W / CAM_W)
    cy = CY    * (H / CAM_H)
    K  = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    rgb = load_video_frame(video_path, 0)
    if rgb is None:
        return "missing_video"
    if rgb.shape[1] != W or rgb.shape[0] != H:
        rgb = cv2.resize(rgb, (W, H), interpolation=cv2.INTER_LINEAR)

    depth = estimate_depth(rgb, K, model, device)
    if depth.shape[1] != W or depth.shape[0] != H:
        depth = cv2.resize(depth, (W, H), interpolation=cv2.INTER_LINEAR)

    explorable, collision = check_explorable(
        depth, fx, fy, cx, cy, forward_dist, depth_threshold)
    if collision:
        return "collision"
    if not explorable:
        return "not_explorable"

    mask               = compute_disocclusion_mask(depth, fx, fy, cx, cy, forward_dist)
    actmap_covered     = np.where(~mask, actmap, 0.0).astype(np.float32)
    actmap_disoccluded = np.where( mask, actmap, 0.0).astype(np.float32)

    frame48, depth48 = _load_depth48(video_root, stem, model, device)
    if frame48 is None:
        print(f"  WARNING: could not read frame 48 for {stem}")

    actmap_frontier = compute_frontier(actmap_disoccluded, depth48, H, W, near_depth_thresh)

    np.save(mask_path,         mask)
    np.save(covered_path,      actmap_covered)
    np.save(disoccluded_path,  actmap_disoccluded)
    np.save(frontier_npy_path, actmap_frontier)
    if depth48 is not None:
        np.save(depth48_npy_path, depth48)
    save_frontier_png(actmap_frontier, frontier_path, frame48=frame48)
    save_vis(rgb, actmap, actmap_covered, actmap_disoccluded, vis_path,
             frame48=frame48, depth48=depth48, near_depth_thresh=near_depth_thresh)
    return "done"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir",    default=os.path.join(os.environ.get("SCRATCH",""), "cache_idp_features"))
    parser.add_argument("--video_root",   default=os.path.join(os.path.dirname(__file__), "..", "videos"))
    parser.add_argument("--out_dir",      default=os.path.join(os.environ.get("SCRATCH",""), "disocclusion_output"))
    parser.add_argument("--forward_dist",      type=float, default=FORWARD_DIST)
    parser.add_argument("--near_depth_thresh", type=float, default=0.5,
                        help="Depth threshold (m) to split disoccluded into texture vs frontier")
    parser.add_argument("--depth_threshold", type=float, default=0.5,
                        help="Explorable check: depth threshold (m) for near-pixel ratio")
    parser.add_argument("--regen_vis", action="store_true",
                        help="Force regeneration of existing vis PNGs")
    args = parser.parse_args()

    # Create output subdirectories
    out_dirs = {
        "masks":       os.path.join(args.out_dir, "masks"),
        "covered":     os.path.join(args.out_dir, "covered"),
        "disoccluded": os.path.join(args.out_dir, "disoccluded"),
        "vis":         os.path.join(args.out_dir, "vis"),
        "frontier":    os.path.join(args.out_dir, "frontier"),
    }
    for d in out_dirs.values():
        os.makedirs(d, exist_ok=True)

    # Collect npy files
    npy_files = sorted(
        os.path.join(args.cache_dir, f)
        for f in os.listdir(args.cache_dir)
        if f.endswith(".npy")
    )
    print(f"Found {len(npy_files)} .npy files in {args.cache_dir}")

    # Load model once
    model, device = load_unik3d_model()

    # Process
    n_done = n_skipped = n_missing = n_error = n_collision = n_not_explorable = 0
    for i, npy_path in enumerate(npy_files):
        try:
            stem = stem_from_npy(npy_path)
        except ValueError as e:
            print(f"[{i+1}/{len(npy_files)}] PARSE ERROR: {e}")
            n_error += 1
            continue

        try:
            status = process_one(
                npy_path, out_dirs, args.forward_dist, model, device,
                video_root=args.video_root,
                near_depth_thresh=args.near_depth_thresh,
                regen_vis=args.regen_vis,
                depth_threshold=args.depth_threshold,
            )
        except Exception as e:
            print(f"[{i+1}/{len(npy_files)}] ERROR {stem}: {e}")
            n_error += 1
            continue

        if status == "skipped":
            n_skipped += 1
        elif status == "missing_video":
            print(f"[{i+1}/{len(npy_files)}] MISSING VIDEO: {stem}")
            n_missing += 1
        elif status == "collision":
            n_collision += 1
        elif status == "not_explorable":
            n_not_explorable += 1
        else:
            n_done += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(npy_files):
            print(f"[{i+1}/{len(npy_files)}]  done={n_done}  skipped={n_skipped}  "
                  f"collision={n_collision}  not_explorable={n_not_explorable}  "
                  f"missing={n_missing}  errors={n_error}")

    print(f"\nFinished. done={n_done}  skipped={n_skipped}  "
          f"collision={n_collision}  not_explorable={n_not_explorable}  "
          f"missing={n_missing}  errors={n_error}")
    print(f"Outputs written to: {args.out_dir}")


if __name__ == "__main__":
    main()
