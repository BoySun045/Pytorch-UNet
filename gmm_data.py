"""
gmm_data.py

Dataset for samples produced by FrontierNet/inspect_gmm_samples.py.

Directory layout expected:
    gmm_dir/   {stem}_gmm.npz   — Gaussian mixture parameters
    depth_dir/ {stem}.png       — metric depth at frame 48 [H, W], uint16 mm

Optionally:
    image_dir/ {stem}.{ext}     — RGB image

__getitem__ returns:
    {
        "image":     [3, H, W] float32
        "depth_map": [1, H, W] float32  metres  (None when depth_dir not set)
        "gmm":       dict  — raw GMM arrays + "occlusion" (K,) bool when depth_dir set
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Image extensions recognised when searching image_dir
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Sample discovery helpers
# ---------------------------------------------------------------------------

def _find_gmm_files(gmm_dir: Path) -> List[Path]:
    """Return sorted list of *_gmm.npz paths."""
    return sorted(gmm_dir.glob("*_gmm.npz"))


def _stem_from_gmm(gmm_path: Path) -> str:
    """Strip the '_gmm' suffix to recover the original sample stem."""
    name = gmm_path.stem          # e.g. "Actmap_MH3D_00000_101_0_gmm"
    assert name.endswith("_gmm"), f"Unexpected GMM file name: {gmm_path}"
    return name[:-4]              # e.g. "Actmap_MH3D_00000_101_0"


def _find_image(stem: str, image_dir: Path) -> Optional[Path]:
    for ext in _IMG_EXTS:
        cand = image_dir / f"{stem}{ext}"
        if cand.exists():
            return cand
    return None


# ---------------------------------------------------------------------------
# GmmDataset
# ---------------------------------------------------------------------------

class GmmDataset(Dataset):
    """Loads FrontierNet GMM samples for DETR training.

    Parameters
    ----------
    gmm_dir:
        Directory containing ``{stem}_gmm.npz`` files.
    image_dir:
        Optional directory with RGB images named ``{stem}.{ext}``.
    depth_dir:
        Optional directory with ``{stem}.png`` depth images (uint16, mm).
        When provided, per-centre occlusion labels and the dense depth map
        tensor are computed and returned.
    depth_scale:
        Divisor applied to raw depth PNG values to convert to metres.
        Default 1000 (mm → m).
    image_size:
        Target (H, W).  Defaults to (544, 720).
    augment:
        Apply random horizontal flip.
    """

    def __init__(
        self,
        gmm_dir: str,
        image_dir: Optional[str] = None,
        depth_dir: Optional[str] = None,
        depth_scale: float = 1000.0,
        image_size: Tuple[int, int] = (544, 720),
        augment: bool = False,
    ) -> None:
        self.gmm_dir = Path(gmm_dir)
        self.image_dir = Path(image_dir) if image_dir else None
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.depth_scale = depth_scale
        self.image_size = image_size
        self.augment = augment

        self.samples = _find_gmm_files(self.gmm_dir)
        if not self.samples:
            raise RuntimeError(f"No *_gmm.npz files found in {self.gmm_dir}")


    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        gmm_path = self.samples[idx]
        stem = _stem_from_gmm(gmm_path)

        # ---- Load scene depth (frame 48) — done first so occlusion
        #      check uses original frame-48 z before the correction below
        depth_map_np: Optional[np.ndarray] = self._load_depth(stem)

        # ---- GMM parameters -------------------------------------------
        gmm = self._load_gmm(gmm_path)

        

        # ---- Frame-48 → frame-0 z correction --------------------------
        # GMMs are fitted in frame-48 camera space; approximate transform
        # to frame 0 by adding ~1 m (camera moves forward ~1 m in 48 frames).
        if "z" in gmm:
            gmm["z"] = gmm["z"].astype(np.float64) + 1.0
        if "centres_3d" in gmm:
            gmm["centres_3d"] = gmm["centres_3d"].astype(np.float64)
            gmm["centres_3d"][:, 2] += 1.0
        
        # ---- Per-centre occlusion labels (after z adjustment) ---------
        if depth_map_np is not None:
            gmm["occlusion"] = self._compute_occlusion(depth_map_np, gmm)

        # ---- RGB image ------------------------------------------------
        image = self._load_image(stem)   # [3, H, W] float32

        # ---- Convert scene depth to tensor [1, H, W] ------------------
        depth_map: Optional[torch.Tensor] = None
        if depth_map_np is not None:
            depth_map = torch.from_numpy(depth_map_np).unsqueeze(0)  # [1, H, W]

        # ---- Optional augmentation ------------------------------------
        if self.augment and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            if depth_map is not None:
                depth_map = torch.flip(depth_map, dims=[2])

        return {
            "image":     image,
            "depth_map": depth_map,
            "gmm":       gmm,
        }



    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_gmm(self, path: Path) -> Dict[str, np.ndarray]:
        """Return raw GMM arrays as a plain dict (not memory-mapped)."""
        with np.load(path) as data:
            return {k: data[k].copy() for k in data.files}

    def _load_depth(self, stem: str) -> Optional[np.ndarray]:
        """Load the depth image for *stem* and return it in metres as float32.

        Looks for ``{depth_dir}/{stem}.png``.  Returns ``None`` if not found.
        Raw values are divided by ``self.depth_scale`` to convert to metres
        (default 1000 for mm-encoded depth PNGs).
        """
        if self.depth_dir is None:
            return None
        depth_path = self.depth_dir / f"{stem}.png"
        if not depth_path.exists():
            return None
        depth_raw = np.array(Image.open(depth_path), dtype=np.float32)
        return depth_raw / self.depth_scale  # metres

    def _compute_occlusion(self, depth_m: np.ndarray, gmm: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a boolean (K,) array indicating occluded GMM centres.

        A centre is marked occluded when its camera-space z depth exceeds the
        observed scene depth at the corresponding image pixel.

        Parameters
        ----------
        depth_m:
            Scene depth array in metres, shape (H, W).
        gmm:
            GMM parameter dict containing ``centres_2d`` (K, 2) and either
            ``z`` (K,) or ``centres_3d`` (K, 3).

        Returns
        -------
        occlusion : (K,) bool ndarray
            ``True`` for each centre whose z is greater than the scene depth
            at its projected pixel.  Centres whose pixel falls outside the
            depth image or whose depth pixel is zero (invalid) are marked
            non-occluded (``False``).
        """
        centres_2d = gmm.get("centres_2d")  # (K, 2)
        if centres_2d is None:
            return np.zeros(0, dtype=bool)

        K = len(centres_2d)

        # Prefer the dedicated per-centre z array; fall back to centres_3d[:, 2]
        if "z" in gmm:
            z_vals = gmm["z"].astype(np.float64)        # (K,)
        elif "centres_3d" in gmm:
            z_vals = gmm["centres_3d"][:, 2].astype(np.float64)
        else:
            return np.zeros(K, dtype=bool)

        depth_H, depth_W = depth_m.shape
        occlusion = np.zeros(K, dtype=bool)

        for k in range(K):
            u, v = centres_2d[k]
            col = int(round(u))
            row = int(round(v))

            # Skip out-of-bounds pixels
            if not (0 <= row < depth_H and 0 <= col < depth_W):
                continue

            scene_depth = float(depth_m[row, col])
            if scene_depth <= 0.0:  # invalid depth pixel — treat as non-occluded
                continue

            occlusion[k] = z_vals[k] > scene_depth

        return occlusion

    def _load_image(self, stem: str) -> torch.Tensor:
        """Return [3, H, W] float32.  Uses an RGB file when available, otherwise
        replicates the single-channel frontier map across 3 channels."""
        if self.image_dir is not None:
            img_path = _find_image(stem, self.image_dir)
            if img_path is not None:
                img = Image.open(img_path).convert("RGB")
                img = img.resize(
                    (self.image_size[1], self.image_size[0]), Image.Resampling.BILINEAR
                )
                return torch.from_numpy(
                    np.asarray(img, dtype=np.float32) / 255.0
                ).permute(2, 0, 1)
        # Fallback: return a black image
        return torch.zeros(3, *self.image_size, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Collate function (compatible with train.py's df_seg branch)
# ---------------------------------------------------------------------------

def collate_gmm(batch: List[Dict]) -> Dict:
    """Stack spatial tensors; keep GMM dicts as a list (variable K per sample)."""
    out: Dict = {
        "image": torch.stack([x["image"] for x in batch]),
        "gmm":   [x["gmm"] for x in batch],   # list of dicts, K varies
    }

    # depth_map is Optional[Tensor] — only stack when all samples have it
    depth_maps = [x.get("depth_map") for x in batch]
    if all(d is not None for d in depth_maps):
        stacked = []
        for d in depth_maps:
            if isinstance(d, torch.Tensor):
                stacked.append(d if d.dim() == 3 else d.unsqueeze(0))
            else:
                stacked.append(torch.from_numpy(np.asarray(d, dtype=np.float32)).unsqueeze(0))
        out["depth_map"] = torch.stack(stacked)
    else:
        out["depth_map"] = None

    return out


# ---------------------------------------------------------------------------
# Manual inspection entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Inspect GmmDataset samples")
    parser.add_argument("--gmm_dir",    default="gmm_dummy/gmm")
    parser.add_argument("--image_dir",  default="gmm_dummy/image")
    parser.add_argument("--depth_dir",  default="gmm_dummy/depth",
                        help="Directory with {stem}.png depth images for occlusion check")
    parser.add_argument("--depth_scale", type=float, default=1000.0,
                        help="Divisor to convert raw depth PNG values to metres (default: 1000)")
    parser.add_argument("--idx", type=int, default=None,
                        help="Sample index to show (default: show all)")
    parser.add_argument("--out_dir", default="gmm_dummy/tmp",
                        help="Save figures here instead of displaying them")
    args = parser.parse_args()

    ds = GmmDataset(
        gmm_dir=args.gmm_dir,
        image_dir=args.image_dir,
        depth_dir=args.depth_dir,
        depth_scale=args.depth_scale,
    )
    print(f"Dataset: {len(ds)} samples")

    indices = [args.idx] if args.idx is not None else list(range(len(ds)))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        gmm_path = ds.samples[i]
        stem = _stem_from_gmm(gmm_path)
        sample = ds[i]

        image     = sample["image"].numpy()   # [3, H, W]
        depth_map = sample["depth_map"]       # [1, H, W] tensor or None
        gmm       = sample["gmm"]

        centres_2d = gmm.get("centres_2d")
        centres_3d = gmm.get("centres_3d")
        weights    = gmm.get("weights")
        occlusion  = gmm.get("occlusion")    # (K,) bool or None
        K = len(centres_2d) if centres_2d is not None else 0

        # ---- text summary --------------------------------------------
        print(f"\n{'─'*60}")
        print(f"[{i}]  stem : {stem}")
        print(f"      image : {image.shape}  range [{image.min():.3f}, {image.max():.3f}]")
        if depth_map is not None:
            dm = depth_map[0].numpy()
            valid = dm > 0
            print(f"  depth_map : range [{dm[valid].min() if valid.any() else 0:.2f},"
                  f" {dm[valid].max() if valid.any() else 0:.2f}] m  ({valid.mean():.1%} valid)")
        if K > 0:
            n_occ = int(occlusion.sum()) if occlusion is not None else "n/a"
            print(f"  occlusion : {n_occ}/{K} centres occluded")
            scene_d = ds._load_depth(stem)
            for k in range(K):
                u, v = centres_2d[k]
                w    = weights[k] if weights is not None else 1.0
                z    = centres_3d[k, 2] if centres_3d is not None else float("nan")
                occ_str = ""
                if occlusion is not None:
                    if scene_d is not None:
                        col, row = int(round(u)), int(round(v))
                        dH, dW = scene_d.shape
                        sd = scene_d[row, col] if (0 <= row < dH and 0 <= col < dW) else float("nan")
                        occ_str = f"  scene_depth={sd:.3f}m  {'OCCLUDED' if occlusion[k] else 'visible'}"
                    else:
                        occ_str = f"  {'OCCLUDED' if occlusion[k] else 'visible'}"
                print(f"          #{k}  (u,v)=({u:.1f},{v:.1f})  z={z:.3f}m  w={w:.4f}{occ_str}")

        # ---- figure --------------------------------------------------
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(stem, fontsize=9)
        axes = axes.ravel()

        # Panel 0 – RGB image with GMM centre markers
        ax = axes[0]
        ax.imshow(image.transpose(1, 2, 0).clip(0, 1))
        if K > 0 and centres_2d is not None:
            colors = plt.cm.Set1(np.linspace(0, 1, max(K, 1)))
            for k in range(K):
                u, v   = centres_2d[k]
                w      = weights[k] if weights is not None else 1.0
                z      = centres_3d[k, 2] if centres_3d is not None else float("nan")
                is_occ = bool(occlusion[k]) if occlusion is not None else False
                marker = "X" if is_occ else "o"
                edge_c = "red" if is_occ else "white"
                ax.scatter(u, v, s=120, c=[colors[k]], marker=marker,
                           linewidths=2, edgecolors=edge_c,
                           label=f"#{k} w={w:.2f} z={z:.1f}m{'  OCC' if is_occ else ''}")
            ax.legend(fontsize=6, loc="upper right")
        occ_note = f"  ({int(occlusion.sum())}/{K} occ)" if occlusion is not None and K > 0 else ""
        ax.set_title(f"RGB image{occ_note}")
        ax.axis("off")

        # Panel 1 – scene depth with GMM centres annotated by occlusion
        ax = axes[1]
        scene_depth = ds._load_depth(stem)
        if scene_depth is not None:
            im = ax.imshow(scene_depth, cmap="plasma")
            fig.colorbar(im, ax=ax, fraction=0.03, label="m")
            if K > 0 and centres_2d is not None:
                for k in range(K):
                    u, v   = centres_2d[k]
                    z      = centres_3d[k, 2] if centres_3d is not None else float("nan")
                    is_occ = bool(occlusion[k]) if occlusion is not None else False
                    color  = "red" if is_occ else "lime"
                    marker = "X" if is_occ else "o"
                    ax.scatter(u, v, s=120, c=color, marker=marker,
                               edgecolors="white", linewidths=0.8, zorder=5)
                    ax.text(u + 4, v - 4, f"#{k}\nz={z:.2f}m",
                            fontsize=5, color="white",
                            bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5))
            vis_patch = plt.Line2D([0], [0], marker="o", color="w",
                                   markerfacecolor="lime", markersize=7, label="visible")
            occ_patch = plt.Line2D([0], [0], marker="X", color="w",
                                   markerfacecolor="red",  markersize=7, label="occluded")
            ax.legend(handles=[vis_patch, occ_patch], fontsize=6, loc="upper right")
            ax.set_title("scene depth [m]  (lime=visible, red=occluded)")
        else:
            ax.text(0.5, 0.5, "no depth available", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("scene depth")
        ax.axis("off")

        # Panel 2 – depth map tensor (from dataset __getitem__)
        ax = axes[2]
        if depth_map is not None:
            dm = depth_map[0].numpy()
            im = ax.imshow(np.where(dm > 0, dm, np.nan), cmap="plasma")
            fig.colorbar(im, ax=ax, fraction=0.03, label="m")
            ax.set_title("depth_map tensor [m]")
        else:
            ax.text(0.5, 0.5, "no depth_map", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title("depth_map tensor")
        ax.axis("off")

        plt.tight_layout()

        if args.out_dir:
            out_path = out_dir / f"{stem}_inspect.png"
            plt.savefig(out_path, dpi=120, bbox_inches="tight")
            print(f"      saved → {out_path}")
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
