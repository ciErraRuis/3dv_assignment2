"""Native reconstruction systems; do not replace their solvers with COLMAP."""

from __future__ import annotations

import itertools
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from .core import Geometry, PairMatches, ReconstructionMethod, draw_matches, save_json, verify


class Mast3rSfm(ReconstructionMethod):
    def __init__(self, sources: Path, weights: Path):
        import torch

        sys.path.insert(0, str(sources / "mast3r"))
        import mast3r.utils.path_to_dust3r  # noqa: F401
        from mast3r.model import AsymmetricMASt3R

        self.torch = torch
        self.model = AsymmetricMASt3R.from_pretrained(str(weights / "mast3r")).eval().cuda()

    def reconstruct(self, images: Path, output: Path) -> Geometry:
        from dust3r.utils.image import load_images
        from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
        from mast3r.image_pairs import make_pairs
        from mast3r.utils.misc import hash_md5

        paths = sorted(images.glob("*.png"))
        names = [str(p) for p in paths]
        # square_ok is an upstream option: the default would crop 512x512 to
        # 512x384. This experiment preserves every source pixel instead.
        imgs = load_images(names, size=512, square_ok=True)
        if any(tuple(im["img"].shape[-2:]) != (512, 512) for im in imgs):
            raise ValueError("MASt3R unexpectedly changed source field of view")
        pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
        cache = output / "native_cache"
        cache.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        scene = sparse_global_alignment(
            names,
            pairs,
            str(cache),
            self.model,
            lr1=0.07,
            niter1=500,
            lr2=0.014,
            niter2=200,
            device="cuda",
            shared_intrinsics=True,
            opt_depth=True,
            matching_conf_thr=5.0,
        )
        # Export the actual correspondences used by sparse_global_alignment.
        summaries = []
        for index, (a, b) in enumerate(itertools.combinations(paths, 2)):
            ha, hb = hash_md5(str(a)), hash_md5(str(b))
            file = cache / "corres_conf=desc_conf_subsample=8" / f"{ha}-{hb}.pth"
            reverse = False
            if not file.exists():
                file = file.with_name(f"{hb}-{ha}.pth")
                reverse = True
            score, (xy1, xy2, confidence) = self.torch.load(
                file, map_location="cpu", weights_only=False
            )

            def arr(x):
                return x.numpy() if hasattr(x, "numpy") else np.asarray(x)

            if reverse:
                xy1, xy2 = xy2, xy1
            result = PairMatches(arr(xy1), arr(xy2), arr(confidence))
            mask = verify(result)
            (output / "matches").mkdir(exist_ok=True)
            np.savez_compressed(
                output / "matches" / f"{a.stem}__{b.stem}.npz",
                first=result.first,
                second=result.second,
                scores=result.scores,
                inliers=mask,
            )
            summaries.append(
                {
                    "first": a.name,
                    "second": b.name,
                    "matches": len(mask),
                    "inliers": int(mask.sum()),
                }
            )
            if index in [0, 1, 4, 8, 17, 34]:
                draw_matches(a, b, result, mask, output / "visuals" / f"pair_{index:03d}.png")
        # Geometry is produced by native sparse global alignment. RGB-only white
        # background filtering is applied to initialization points, not inputs.
        pts = np.concatenate([p.detach().cpu().numpy() for p in scene.get_sparse_pts3d()])
        rgb = np.concatenate(scene.get_pts3d_colors())
        keep = np.isfinite(pts).all(1) & np.any(rgb < 0.96, axis=1)
        geometry = Geometry(
            [p.name for p in paths],
            scene.get_im_poses().detach().cpu().numpy(),
            scene.intrinsics.detach().cpu().numpy(),
            pts[keep],
            (rgb[keep].clip(0, 1) * 255).round().astype(np.uint8),
            {
                "method": "mast3r_sfm",
                "registered": len(paths),
                "input_views": len(paths),
                "registration_rate": 1.0,
                "gt_cameras_consumed": False,
                "pose_source": "native_sparse_global_alignment",
                "square_ok": True,
                "coarse_iterations": 500,
                "refine_iterations": 200,
                "native_seconds": time.monotonic() - start,
            },
        )
        geometry.save(output)
        save_json(
            output / "matching.json",
            {
                "method": "mast3r_sfm",
                "pairs": summaries,
                "inlier_visualization": "post-hoc RANSAC diagnostic; native solver uses its own confidence",
            },
        )
        # Raw correspondences and final geometry persist; multi-GB forward caches
        # are no longer needed once the reconstruction/export has succeeded.
        cache_bytes = sum(p.stat().st_size for p in cache.rglob("*") if p.is_file())
        del scene
        shutil.rmtree(cache)
        save_json(output / "cleanup.json", {"removed": "native_cache", "bytes": cache_bytes})
        return geometry
