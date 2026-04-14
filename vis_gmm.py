"""
vis_gmm.py

Visualize a GMM .npz file together with the original image's 3D point cloud.

Point cloud: estimated via MoGE depth from --image.
GMM:         loaded from --npz (centres_3d, covariances, weights).

Each Gaussian is shown as:
  - a sphere at the 3D center, sized by weight
  - a semi-transparent 1-sigma ellipsoid (from the covariance)

Usage:
    python vis_gmm.py \
        --npz   path/to/Actmap_..._gmm.npz \
        --image path/to/Actmap_....jpg \
        [--port 8080]

Open the printed URL in a browser (SSH-forward port if on a remote cluster).
MoGE requires the gen3c_env or an env with moge installed.
"""

import argparse
import os
import sys
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import viser

# ---------------------------------------------------------------------------
# MoGE depth + unprojection  (copied from GEN3C/data_filtering.py)
# ---------------------------------------------------------------------------

def run_moge(image_rgb: np.ndarray, device: str, moge_model=None):
    orig_h, orig_w = image_rgb.shape[:2]
    moge_h, moge_w = 720, 1280

    img_resized = cv2.resize(image_rgb, (moge_w, moge_h))
    img_tensor = (
        torch.tensor(img_resized / 255.0, dtype=torch.float32, device=device)
        .permute(2, 0, 1)
    )

    if moge_model is None:
        from moge.model.v1 import MoGeModel
        print("Loading MoGE model …")
        moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
        moge_model.eval()

    with torch.no_grad():
        out = moge_model.infer(img_tensor)

    depth_moge = out["depth"].cpu()
    intr_norm  = out["intrinsics"].cpu()
    mask_moge  = out["mask"].cpu()

    K_moge = intr_norm.clone().numpy().astype(np.float64)
    K_moge[0, 0] *= moge_w
    K_moge[1, 1] *= moge_h
    K_moge[0, 2] *= moge_w
    K_moge[1, 2] *= moge_h

    depth_orig = F.interpolate(
        depth_moge.unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="bilinear", align_corners=False
    ).squeeze().numpy().astype(np.float32)

    mask_orig = F.interpolate(
        mask_moge.float().unsqueeze(0).unsqueeze(0),
        size=(orig_h, orig_w), mode="nearest"
    ).squeeze().numpy().astype(bool)

    K = K_moge.copy()
    K[0, 0] *= orig_w / moge_w
    K[1, 1] *= orig_h / moge_h
    K[0, 2] *= orig_w / moge_w
    K[1, 2] *= orig_h / moge_h

    return depth_orig, K, mask_orig


def unproject(image_rgb: np.ndarray, depth: np.ndarray,
              K: np.ndarray, mask: np.ndarray):
    H, W = depth.shape
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us, vs = np.meshgrid(np.arange(W, dtype=np.float32),
                         np.arange(H, dtype=np.float32))
    X = (us - cx) * depth / fx
    Y = (vs - cy) * depth / fy
    Z = depth

    pts    = np.stack([X, Y, Z], axis=-1)[mask]
    colors = image_rgb[mask]
    return pts.astype(np.float32), colors


# ---------------------------------------------------------------------------
# GMM helpers
# ---------------------------------------------------------------------------

def covariance_to_wxyz_scale(cov: np.ndarray):
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 1e-8, None)
    scale = np.sqrt(eigvals)

    R = eigvecs
    if np.linalg.det(R) < 0:
        R[:, 0] *= -1

    t = R[0, 0] + R[1, 1] + R[2, 2]
    if t > 0:
        s = 0.5 / np.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return np.array([w, x, y, z]), scale


def _unit_sphere_vertices(n_lat=16, n_lon=16):
    verts = []
    for i in range(n_lat + 1):
        lat = np.pi * (-0.5 + i / n_lat)
        for j in range(n_lon):
            lon = 2 * np.pi * j / n_lon
            verts.append([np.cos(lat) * np.cos(lon),
                           np.cos(lat) * np.sin(lon),
                           np.sin(lat)])
    return np.array(verts, dtype=np.float32)


def _unit_sphere_faces(n_lat=16, n_lon=16):
    faces = []
    for i in range(n_lat):
        for j in range(n_lon):
            a = i * n_lon + j
            b = i * n_lon + (j + 1) % n_lon
            c = (i + 1) * n_lon + (j + 1) % n_lon
            d = (i + 1) * n_lon + j
            faces += [[a, b, c], [a, c, d]]
    return np.array(faces, dtype=np.uint32)


def to_viser(pts: np.ndarray) -> np.ndarray:
    return pts.astype(np.float32)


def crop_to_content(img: np.ndarray, pad: int = 16) -> np.ndarray:
    """Crop blank background borders from a viser screenshot.

    Estimates background colour from the four corner patches, then returns
    the bounding box of pixels that differ from it, plus `pad` pixels margin.
    """
    h, w = img.shape[:2]
    corners = np.concatenate([
        img[:8,  :8 ].reshape(-1, 3),
        img[:8,  -8:].reshape(-1, 3),
        img[-8:, :8 ].reshape(-1, 3),
        img[-8:, -8:].reshape(-1, 3),
    ], axis=0)
    bg   = np.median(corners, axis=0)
    diff = np.abs(img.astype(np.int32) - bg).max(axis=2)
    mask = diff > 15

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return img

    r0 = max(rows[0]  - pad, 0);  r1 = min(rows[-1] + pad + 1, h)
    c0 = max(cols[0]  - pad, 0);  c1 = min(cols[-1] + pad + 1, w)
    return img[r0:r1, c0:c1]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz",   required=True, help="Path to _gmm.npz file")
    parser.add_argument("--image", default=None,  help="Path to corresponding RGB image for point cloud")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--port",  type=int, default=8080)
    parser.add_argument("--point_size", type=float, default=0.02,
                        help="Point cloud point size (default: 0.02)")
    parser.add_argument("--max_points", type=int, default=0,
                        help="Subsample point cloud to at most this many points (0 = no limit)")
    parser.add_argument("--gmm_scale", type=float, default=1.0,
                        help="Scale multiplier for GMM spheres and ellipsoids (default: 1.0)")
    parser.add_argument("--save_views", action="store_true",
                        help="Automatically save two rendered views and exit")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save rendered images (default: same directory as --npz)")
    args = parser.parse_args()

    # --- load GMM ---
    data    = np.load(args.npz)
    centres = data["centres_3d"]    # (K, 3)
    covs    = data["covariances"]   # (K, 3, 3)
    weights = data["weights"]       # (K,)
    K       = len(weights)
    w_norm  = weights / weights.max()

    # hot colormap for Gaussians
    cmap = np.zeros((K, 3), dtype=np.uint8)
    for i, w in enumerate(w_norm):
        cmap[i] = [
            int(np.clip(w * 3,     0, 1) * 255),
            int(np.clip(w * 3 - 1, 0, 1) * 255),
            int(np.clip(w * 3 - 2, 0, 1) * 255),
        ]

    # --- optionally build point cloud ---
    pts_vis = colors_vis = None
    if args.image is not None:
        print(f"Loading image: {args.image}")
        img_bgr   = cv2.imread(args.image)
        image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        depth, K_cam, mask = run_moge(image_rgb, args.device)
        pts, colors = unproject(image_rgb, depth, K_cam, mask)

        # subsample
        if args.max_points > 0 and len(pts) > args.max_points:
            idx = np.random.choice(len(pts), args.max_points, replace=False)
            pts, colors = pts[idx], colors[idx]

        pts_vis    = to_viser(pts)
        colors_vis = colors
        print(f"Point cloud: {len(pts_vis):,} points")

    # --- start viser ---
    server = viser.ViserServer(port=args.port)
    print(f"\nOpen your browser at:  http://localhost:{args.port}\n")

    # Compute scene bounds for initial camera position
    all_pts_cam = np.concatenate([centres] + ([pts_vis] if pts_vis is not None else []), axis=0)
    scene_cx = float(all_pts_cam[:, 0].mean())
    scene_cy = float(all_pts_cam[:, 1].mean())
    scene_cz = float(all_pts_cam[:, 2].mean())
    scene_depth = float(all_pts_cam[:, 2].max() - all_pts_cam[:, 2].min())

    @server.on_client_connect
    def _on_connect(client: viser.ClientHandle) -> None:
        client.camera.position     = np.array([scene_cx, scene_cy, scene_cz - scene_depth * 1.5])
        client.camera.look_at      = np.array([scene_cx, scene_cy, scene_cz])
        client.camera.up_direction = np.array([0.0, -1.0, 0.0])

    stem = args.npz.split("/")[-1].replace("_gmm.npz", "")

    # point cloud
    if pts_vis is not None:
        print(f"pts_vis dtype={pts_vis.dtype} shape={pts_vis.shape} "
              f"X=[{pts_vis[:,0].min():.2f},{pts_vis[:,0].max():.2f}] "
              f"Y=[{pts_vis[:,1].min():.2f},{pts_vis[:,1].max():.2f}] "
              f"Z=[{pts_vis[:,2].min():.2f},{pts_vis[:,2].max():.2f}]")
        server.scene.add_point_cloud(
            name="pointcloud",
            points=pts_vis,
            colors=colors_vis,
            point_size=args.point_size,
            point_shape="circle",
            precision="float32",
        )

    # GMM Gaussians
    for i in range(K):
        pos   = to_viser(centres[i:i+1]).squeeze(0)
        wxyz, scale = covariance_to_wxyz_scale(covs[i])

        # flip scale axes to match viser coord change (Y, Z flipped)
        scale_vis = scale.copy()
        # covariance is in camera coords; for display just keep magnitudes
        color = tuple(int(v) for v in cmap[i])

        server.scene.add_icosphere(
            name=f"gmm/{stem}/center_{i}",
            radius=float(0.03 + 0.06 * w_norm[i]) * args.gmm_scale,
            position=pos.astype(float),
            color=color,
        )
        server.scene.add_mesh_simple(
            name=f"gmm/{stem}/ellipsoid_{i}",
            vertices=(_unit_sphere_vertices() * scale_vis * args.gmm_scale).astype(np.float32),
            faces=_unit_sphere_faces(),
            wxyz=wxyz.astype(np.float32),
            position=pos.astype(np.float32),
            color=color,
            opacity=0.25,
            flat_shading=False,
        )

    server.scene.add_frame("origin", axes_length=0.3, axes_radius=0.01)

    print(f"Showing {K} Gaussians from {args.npz}")
    if args.image:
        print(f"Point cloud from: {args.image}")

    if args.save_views:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.npz))
        os.makedirs(out_dir, exist_ok=True)

        # Unit sphere for ellipsoid wireframes (reused per component)
        _u = np.linspace(0, 2 * np.pi, 20)
        _v = np.linspace(0,     np.pi, 20)
        sx = np.outer(np.cos(_u), np.sin(_v))
        sy = np.outer(np.sin(_u), np.sin(_v))
        sz = np.outer(np.ones_like(_u), np.cos(_v))
        sphere_pts = np.stack([sx.ravel(), sy.ravel(), sz.ravel()], axis=1)  # (400,3)

        hot_colors = plt.cm.hot(np.linspace(0.3, 1.0, K))

        # (view_name, elev_deg, azim_deg)
        views = [
            ("view_from_origin", 0,   -90),   # front view, looking along +Z
            ("view_birdseye",   -90,  -90),   # bird's eye, looking along +Y (down)
        ]

        for view_name, elev, azim in views:
            fig = plt.figure(figsize=(16, 9), dpi=120)
            ax  = fig.add_subplot(111, projection="3d")

            # Point cloud (subsample for speed)
            if pts_vis is not None:
                n_show = min(len(pts_vis), 50_000)
                idx    = np.random.choice(len(pts_vis), n_show, replace=False)
                pc     = pts_vis[idx]
                col    = colors_vis[idx].astype(np.float32) / 255.0
                ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2],
                           c=col, s=0.3, alpha=0.4, linewidths=0)

            # GMM components
            for i in range(K):
                c   = centres[i]
                col = hot_colors[i]

                ax.scatter(*c, s=120 * (0.5 + 0.5 * w_norm[i]),
                           c=[col], marker="o",
                           label=f"#{i}  z={c[2]:.1f}m  w={w_norm[i]:.2f}")

                eigvals, eigvecs = np.linalg.eigh(covs[i])
                eigvals = np.clip(eigvals, 1e-8, None)
                scale   = np.sqrt(eigvals) * args.gmm_scale

                R = eigvecs.copy()
                if np.linalg.det(R) < 0:
                    R[:, 0] *= -1

                ell = (R @ (sphere_pts * scale).T).T + c
                ex  = ell[:, 0].reshape(20, 20)
                ey  = ell[:, 1].reshape(20, 20)
                ez  = ell[:, 2].reshape(20, 20)
                ax.plot_wireframe(ex, ey, ez,
                                  color=col, alpha=0.25, linewidth=0.6)

            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]"); ax.set_zlabel("Z [m]")
            ax.set_title(view_name.replace("_", " "))
            ax.legend(fontsize=7, loc="upper left")

            plt.tight_layout()
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)

            img = crop_to_content(img)
            save_path = os.path.join(out_dir, f"{stem}_{view_name}.png")
            cv2.imwrite(save_path, img[:, :, ::-1])
            print(f"Saved: {save_path}")

        print("Done. Press Ctrl-C to stop the server.")

    print("Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
