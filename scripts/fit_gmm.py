"""
fit_gmm.py

Fit a Bayesian GMM (Dirichlet Process) in 3D camera space on frontier activation maps.
A depth map (*_depth48.npy) is required for each frontier map.
The number of active components is determined automatically up to --k_max.

For each *_frontier.npy in --data_dir:
  - Load the corresponding *_depth48.npy (skips if missing)
  - Sample (x, y, z) camera-space coordinates proportional to frontier activation values
  - Fit a Bayesian GMM; components with negligible weight are pruned automatically
  - Save a visualisation: frontier heatmap + projected GMM centres + depth panel
  - Save centres to gmm_centres.npz: {"stems": [...], "centres": ragged list of (K_i, 3)}

Usage:
    python fit_gmm.py \
        [--data_dir   $SCRATCH/disocclusion_output/frontier] \
        [--out_dir    $SCRATCH/gmm_output] \
        [--k_max      8] \
        [--n_samples  2000] \
        [--min_val    0.01] \
        [--weight_thresh 0.02]
"""

import argparse
import os

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture

# ---------------------------------------------------------------------------
# Camera constants (must match batch_gen_disocclusion_masks.py)
# ---------------------------------------------------------------------------
CAM_H = 544
CAM_W = 720
CAM_F = 300.0
CX    = CAM_W / 2.0 - 0.5   # 359.5
CY    = CAM_H / 2.0 - 0.5   # 271.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_intrinsics(H: int, W: int):
    """Scale camera intrinsics to a given image resolution."""
    fx = CAM_F * (W / CAM_W)
    fy = CAM_F * (H / CAM_H)
    cx = CX    * (W / CAM_W)
    cy = CY    * (H / CAM_H)
    return fx, fy, cx, cy


def sample_coords_3d(
    frontier: np.ndarray,
    depth: np.ndarray,
    n_samples: int,
    min_val: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray | None:
    """
    Sample (x, y, z) camera-space coordinates proportional to frontier activation.
    depth must already be resized to match frontier shape.
    Returns array of shape (n_samples, 3) or None if not enough active pixels.
    """
    vs, us = np.where(frontier > min_val)
    if len(us) < 2:
        return None

    d_vals = depth[vs, us].astype(np.float64)
    valid  = d_vals > 0
    vs, us, d_vals = vs[valid], us[valid], d_vals[valid]
    if len(us) < 2:
        return None

    weights = frontier[vs, us].astype(np.float64)
    weights /= weights.sum()
    idx = np.random.choice(len(us), size=n_samples, replace=True, p=weights)

    u_s = us[idx].astype(np.float64) + np.random.uniform(-0.5, 0.5, n_samples)
    v_s = vs[idx].astype(np.float64) + np.random.uniform(-0.5, 0.5, n_samples)
    d_s = d_vals[idx] + np.random.uniform(-0.01, 0.01, n_samples)
    d_s = np.maximum(d_s, 1e-3)

    x = (u_s - cx) * d_s / fx
    y = (v_s - cy) * d_s / fy
    return np.stack([x, y, d_s], axis=1)


def project_to_2d(means_3d: np.ndarray, fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Project 3-D (x, y, z) camera-space means to 2-D (u, v) pixel coords."""
    x, y, z = means_3d[:, 0], means_3d[:, 1], means_3d[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    return np.stack([u, v], axis=1)


def active_components(gmm: BayesianGaussianMixture, weight_thresh: float):
    """Return indices of components whose weight exceeds the threshold."""
    return np.where(gmm.weights_ > weight_thresh)[0]



def save_gmm_vis(frontier: np.ndarray, gmm: BayesianGaussianMixture,
                 active_idx: np.ndarray, out_path: str, stem: str,
                 means_2d: np.ndarray,
                 depth: np.ndarray):
    """
    Visualise 3-D GMM on the frontier heatmap alongside the depth map.
    means_2d: projected (u, v) pixel coordinates of active component centres.
    depth: depth map shown in a second panel.
    """
    K = len(active_idx)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax = axes[0]
    ax.imshow(frontier, cmap="hot", origin="upper")

    colours = plt.cm.cool(np.linspace(0, 1, max(K, 1)))
    for ci, k in enumerate(active_idx):
        mu_uv = means_2d[ci]
        label = f"#{ci} w={gmm.weights_[k]:.2f}\nz={gmm.means_[k][2]:.2f}m"
        ax.plot(*mu_uv, marker="x", color=colours[ci], markersize=10, markeredgewidth=2)
        ax.annotate(label, xy=mu_uv, color=colours[ci],
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_title(f"{stem}  (active K={K})", fontsize=9)
    ax.axis("off")

    im = axes[1].imshow(depth, cmap="plasma", origin="upper")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="m")
    axes[1].set_title("Depth — Frame 48 (UniK3D)", fontsize=9)
    axes[1].axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",      default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "disocclusion_output", "frontier"))
    parser.add_argument("--out_dir",       default=os.path.join(os.environ.get("SCRATCH", "/tmp"), "gmm_output"))
    parser.add_argument("--k_max",         type=int,   default=10,    help="Upper bound on number of components")
    parser.add_argument("--n_samples",     type=int,   default=5000, help="Coordinates sampled per map")
    parser.add_argument("--min_val",       type=float, default=0.01, help="Minimum frontier value to include")
    parser.add_argument("--weight_thresh", type=float, default=0.02, help="Min weight to count a component as active")
    parser.add_argument("--seed",          type=int,   default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    npy_files = sorted(
        os.path.join(args.data_dir, f)
        for f in os.listdir(args.data_dir)
        if f.endswith("_frontier.npy")
    )
    print(f"Found {len(npy_files)} frontier .npy files in {args.data_dir}")

    all_stems   = []
    all_centres = []   # list of variable-length (K_i, 2) arrays

    n_done = n_skip = n_error = 0
    for i, npy_path in enumerate(npy_files):
        stem = os.path.basename(npy_path).replace("_frontier.npy", "")
        vis_path = os.path.join(args.out_dir, stem + "_gmm.png")

        try:
            frontier = np.load(npy_path)   # (H, W) float32
            H, W = frontier.shape

            # --- 3-D mode: load depth48 and back-project to camera space ---
            depth48_path = os.path.join(args.data_dir, stem + "_depth48.npy")
            if not os.path.exists(depth48_path):
                print(f"  [{i+1}] SKIP {stem} — depth map not found ({depth48_path})")
                n_skip += 1
                continue

            means_2d = None
            depth48 = np.load(depth48_path)
            if depth48.shape != (H, W):
                depth48 = cv2.resize(depth48, (W, H), interpolation=cv2.INTER_LINEAR)
            fx, fy, cx, cy = _get_intrinsics(H, W)
            coords = sample_coords_3d(frontier, depth48, args.n_samples,
                                      args.min_val, fx, fy, cx, cy)

            if coords is None:
                print(f"  [{i+1}] SKIP {stem} — not enough active pixels")
                n_skip += 1
                continue

            gmm = BayesianGaussianMixture(
                n_components=args.k_max,
                covariance_type="full",
                weight_concentration_prior_type="dirichlet_process",
                weight_concentration_prior=1.0 / args.k_max,
                random_state=args.seed,
                max_iter=500,
            )
            gmm.fit(coords)

            active_idx = active_components(gmm, args.weight_thresh)
            centres = gmm.means_[active_idx]   # (K_i, 2) or (K_i, 3)

            # For 3-D GMMs project centres to pixel space for visualisation
            if centres.shape[1] == 3:
                means_2d = project_to_2d(centres, fx, fy, cx, cy)

            save_gmm_vis(frontier, gmm, active_idx, vis_path, stem,
                         means_2d=means_2d, depth=depth48)

            all_stems.append(stem)
            all_centres.append(centres)
            n_done += 1

        except Exception as e:
            print(f"  [{i+1}] ERROR {stem}: {e}")
            n_error += 1
            continue

        if (i + 1) % 50 == 0 or (i + 1) == len(npy_files):
            k_str = f"K={len(active_idx)}" if n_done > 0 else ""
            print(f"  [{i+1}/{len(npy_files)}]  done={n_done}  skipped={n_skip}  errors={n_error}  {k_str}")

    # Save centres (ragged: each entry has a different K_i)
    centres_path = os.path.join(args.out_dir, "gmm_centres.npz")
    np.savez(
        centres_path,
        stems=np.array(all_stems),
        centres=np.array(all_centres, dtype=object),
    )
    print(f"\nDone. Visualisations + centres → {args.out_dir}/")
    print(f"  gmm_centres.npz: {len(all_stems)} samples")


if __name__ == "__main__":
    main()
