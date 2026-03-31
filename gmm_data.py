"""
gmm_data.py

Dataset for samples produced by FrontierNet/inspect_gmm_samples.py.

Directory layout expected:
    gmm_dir/   {stem}_gmm.npz          — Gaussian mixture parameters
    data_dir/  {stem}_frontier.npy     — frontier activation map [H, W], float32 [0,1]
               {stem}_depth48.npy      — metric depth at frame 48 [H, W], float32

Optionally:
    image_dir/ {stem}.{ext}            — RGB image at frame 0

The frontier map is used directly as the continuous interest (supervision) signal,
mirroring the role of the IDP map in InterestDataset.

__getitem__ returns a dict compatible with the df_seg training branch in train.py:
    {
        "image":      [3, H, W] float32  — RGB or frontier replicated to 3ch
        "interest":   [1, H, W] float32  — frontier activation in [0, 1]
        "valid":      [1, H, W] bool
        "label_mask": [H, W]    int64
        "depth":      [1, H, W] float32  — depth48 (present when data_dir given)
        "gmm":        dict               — raw GMM arrays from the .npz file
    }
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
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
    """Loads FrontierNet GMM samples for UNet training.

    Parameters
    ----------
    gmm_dir:
        Directory containing ``{stem}_gmm.npz`` files produced by
        ``inspect_gmm_samples.py``.
    data_dir:
        Directory containing ``{stem}_frontier.npy`` and
        ``{stem}_depth48.npy`` files.  If *None*, frontier/depth loading
        is skipped and an all-zero image placeholder is returned.
    image_dir:
        Optional directory with RGB images named ``{stem}.{ext}``.
        When provided and a matching file is found, the RGB image is
        returned as ``"image"``; otherwise the frontier map is repeated
        across 3 channels.
    image_size:
        Target (H, W) for all spatial tensors.  Defaults to (544, 720),
        which matches the camera constants in inspect_gmm_samples.py.
    min_interest:
        Lower threshold for the valid mask (pixels below this are ignored).
    num_classes:
        Number of discretisation classes for ``label_mask`` (including the
        background / invalid class 0).
    seg_bin_edges:
        Monotone thresholds that split ``[0, 1]`` into ``num_classes - 1``
        foreground buckets.  Length must equal ``num_classes - 1``.
    augment:
        Apply random horizontal flip.
    """

    def __init__(
        self,
        gmm_dir: str,
        data_dir: Optional[str] = None,
        image_dir: Optional[str] = None,
        image_size: Tuple[int, int] = (544, 720),
        min_interest: float = 1e-3,
        num_classes: int = 7,
        seg_bin_edges: Tuple[float, ...] = (0.05, 0.15, 0.3, 0.45, 0.6, 0.8),
        augment: bool = False,
    ) -> None:
        self.gmm_dir = Path(gmm_dir)
        self.data_dir = Path(data_dir) if data_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.image_size = image_size
        self.min_interest = min_interest
        self.num_classes = num_classes
        self.seg_bin_edges = seg_bin_edges
        self.augment = augment

        gmm_files = _find_gmm_files(self.gmm_dir)
        if not gmm_files:
            raise RuntimeError(f"No *_gmm.npz files found in {self.gmm_dir}")

        # Build list of (stem, gmm_path) keeping only samples whose frontier
        # map exists in data_dir (when data_dir is provided).
        self.samples: List[Tuple[str, Path]] = []
        for gmm_path in gmm_files:
            stem = _stem_from_gmm(gmm_path)
            if self.data_dir is not None:
                frontier_path = self.data_dir / f"{stem}_frontier.npy"
                if not frontier_path.exists():
                    continue  # skip samples missing frontier data
            self.samples.append((stem, gmm_path))

        if not self.samples:
            raise RuntimeError(
                f"No usable samples found. gmm_dir={self.gmm_dir}, "
                f"data_dir={self.data_dir}"
            )

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        stem, gmm_path = self.samples[idx]

        # ---- GMM parameters -------------------------------------------
        gmm = self._load_gmm(gmm_path)

        # ---- Frontier / interest map ----------------------------------
        if self.data_dir is not None:
            frontier = self._load_frontier(stem)   # [1, H, W] float32
            depth = self._load_depth(stem)          # [1, H, W] float32 or None
        else:
            H, W = self.image_size
            frontier = torch.zeros(1, H, W, dtype=torch.float32)
            depth = None

        interest = frontier  # alias: frontier IS the interest signal

        # ---- RGB image ------------------------------------------------
        image = self._load_image(stem, frontier)   # [3, H, W] float32

        # ---- Valid mask & label mask ----------------------------------
        valid = (interest > self.min_interest)                # [1, H, W] bool
        label_mask = self._generate_label_mask(interest, valid)  # [H, W] int64

        # ---- Optional augmentation ------------------------------------
        if self.augment and torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[2])
            interest = torch.flip(interest, dims=[2])
            valid = torch.flip(valid, dims=[2])
            label_mask = torch.flip(label_mask, dims=[1])
            if depth is not None:
                depth = torch.flip(depth, dims=[2])

        sample: Dict[str, object] = {
            "image": image,
            "interest": interest,
            "valid": valid,
            "label_mask": label_mask,
            "gmm": gmm,
        }
        if depth is not None:
            sample["depth"] = depth
        return sample

    # ------------------------------------------------------------------
    # Loading helpers
    # ------------------------------------------------------------------

    def _load_gmm(self, path: Path) -> Dict[str, np.ndarray]:
        """Return raw GMM arrays as a plain dict (not memory-mapped)."""
        with np.load(path) as data:
            return {k: data[k].copy() for k in data.files}

    def _load_frontier(self, stem: str) -> torch.Tensor:
        """Load frontier activation map → [1, H, W] float32, values in [0, 1]."""
        path = self.data_dir / f"{stem}_frontier.npy"
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)          # (H, W)
        t = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        if t.shape[1:] != tuple(self.image_size):
            t = F.interpolate(
                t.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return t

    def _load_depth(self, stem: str) -> Optional[torch.Tensor]:
        """Load depth48 map → [1, H, W] float32, or None if file missing."""
        path = self.data_dir / f"{stem}_depth48.npy"
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        t = torch.from_numpy(arr).unsqueeze(0)  # [1, H, W]
        if t.shape[1:] != tuple(self.image_size):
            t = F.interpolate(
                t.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False
            ).squeeze(0)
        return t

    def _load_image(self, stem: str, frontier: torch.Tensor) -> torch.Tensor:
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
        # Fallback: tile frontier to 3 channels
        return frontier.expand(3, -1, -1).clone()

    # ------------------------------------------------------------------
    # Label mask
    # ------------------------------------------------------------------

    def _generate_label_mask(
        self, interest: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        """Discretise interest into integer classes; 0 = invalid background."""
        base = interest[0]  # [H, W]
        edges = torch.tensor(self.seg_bin_edges, dtype=base.dtype, device=base.device)
        labels = torch.bucketize(base, edges) + 1          # 1 … num_classes-1
        labels = torch.where(valid[0], labels, torch.zeros_like(labels))
        return labels.clamp(max=self.num_classes - 1).long()


# ---------------------------------------------------------------------------
# Collate function (compatible with train.py's df_seg branch)
# ---------------------------------------------------------------------------

def collate_gmm(batch: List[Dict]) -> Dict:
    """Stack spatial tensors; keep GMM dicts as a list (variable K per sample)."""
    output = {
        "image":      torch.stack([x["image"] for x in batch]),
        "interest":   torch.stack([x["interest"] for x in batch]),
        "valid":      torch.stack([x["valid"] for x in batch]),
        "label_mask": torch.stack([x["label_mask"] for x in batch]),
        "gmm":        [x["gmm"] for x in batch],   # list of dicts, K varies
    }
    if all("depth" in x for x in batch):
        output["depth"] = torch.stack([x["depth"] for x in batch])
    return output


# ---------------------------------------------------------------------------
# Manual inspection entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import Normalize

    parser = argparse.ArgumentParser(description="Inspect GmmDataset samples")
    parser.add_argument(
        "--gmm_dir", default="/cluster/project/cvg/students/shangwu/FrontierNet/gmm_output",
    )
    parser.add_argument(
        "--data_dir", default="/cluster/project/cvg/students/shangwu/FrontierNet/data",
    )
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--idx", type=int, default=None,
                        help="Sample index to show (default: show all)")
    parser.add_argument("--out_dir", default=None,
                        help="Save figures here instead of displaying them")
    args = parser.parse_args()

    ds = GmmDataset(
        gmm_dir=args.gmm_dir,
        data_dir=args.data_dir,
        image_dir=args.image_dir,
    )
    print(f"Dataset: {len(ds)} samples")

    indices = [args.idx] if args.idx is not None else list(range(len(ds)))

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

    for i in indices:
        stem, gmm_path = ds.samples[i]
        sample = ds[i]

        image      = sample["image"].numpy()          # [3, H, W]
        interest   = sample["interest"][0].numpy()    # [H, W]
        valid      = sample["valid"][0].numpy()       # [H, W] bool
        label_mask = sample["label_mask"].numpy()     # [H, W] int64
        gmm        = sample["gmm"]
        depth      = sample["depth"][0].numpy() if "depth" in sample else None

        K = len(gmm["weights"])
        centres_2d = gmm["centres_2d"]   # (K, 2)  u, v  [pixels]
        centres_3d = gmm["centres_3d"]   # (K, 3)  x, y, z [m]
        weights    = gmm["weights"]      # (K,)
        covs       = gmm["covariances"]  # (K, 3, 3)

        # ---- text summary --------------------------------------------
        print(f"\n{'─'*60}")
        print(f"[{i}]  stem : {stem}")
        print(f"      image : {image.shape}  range [{image.min():.3f}, {image.max():.3f}]")
        print(f"   interest : {interest.shape}  range [{interest.min():.4f}, {interest.max():.4f}]"
              f"  valid_frac={valid.mean():.3f}")
        print(f" label_mask : unique classes = {sorted(set(label_mask.flatten().tolist()))}")
        if depth is not None:
            valid_depth = depth[depth > 0]
            print(f"      depth : {depth.shape}  range [{valid_depth.min():.2f}, {valid_depth.max():.2f}] m")
        print(f"        GMM : K={K} active components")
        for k in range(K):
            u, v = centres_2d[k]
            x, y, z = centres_3d[k]
            w = weights[k]
            print(f"          #{k}  (u,v)=({u:.1f},{v:.1f})  z={z:.2f}m  w={w:.4f}")
            print(f"              cov diag = [{covs[k,0,0]:+.5f}  {covs[k,1,1]:+.5f}  {covs[k,2,2]:+.5f}]")

        # ---- figure --------------------------------------------------
        ncols = 4 if depth is None else 5
        fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
        fig.suptitle(stem, fontsize=9)

        # Panel 0 – image
        ax = axes[0]
        ax.imshow(image.transpose(1, 2, 0).clip(0, 1))
        ax.set_title("image (frontier ×3)" if args.image_dir is None else "RGB image")
        ax.axis("off")

        # Panel 1 – frontier / interest
        ax = axes[1]
        im = ax.imshow(interest, cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"interest (frontier)\nmax={interest.max():.3f}")
        ax.axis("off")

        # Panel 2 – label mask
        ax = axes[2]
        n_cls = ds.num_classes
        im = ax.imshow(label_mask, cmap="hot", vmin=0, vmax=n_cls - 1)
        fig.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title(f"label_mask (0–{n_cls-1})")
        ax.axis("off")

        # Panel 3 – interest with GMM centres overlaid
        ax = axes[3]
        ax.imshow(interest, cmap="viridis", vmin=0, vmax=1)
        colors = plt.cm.Set1(np.linspace(0, 1, max(K, 1)))
        for k in range(K):
            u, v = centres_2d[k]
            ax.scatter(u, v, s=120, c=[colors[k]], marker="x", linewidths=2,
                       label=f"#{k} w={weights[k]:.3f} z={centres_3d[k,2]:.1f}m")
        if K > 0:
            ax.legend(fontsize=6, loc="upper right")
        ax.set_title("GMM centres on frontier")
        ax.axis("off")

        # Panel 4 (optional) – depth
        if depth is not None:
            ax = axes[4]
            disp = np.where(depth > 0, depth, np.nan)
            im = ax.imshow(disp, cmap="plasma")
            fig.colorbar(im, ax=ax, fraction=0.03)
            ax.set_title("depth48 [m]")
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
