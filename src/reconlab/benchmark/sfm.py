"""A traditional SfM baseline for pairwise matchers; native methods bypass it."""

from __future__ import annotations

import itertools
import time
from pathlib import Path

import numpy as np

from .core import Geometry, ReconstructionMethod, TrackGraph, draw_matches, save_json, verify


def from_colmap(model, method: str) -> Geometry:
    images = sorted((im for im in model.images.values() if im.has_pose), key=lambda x: x.name)
    points = list(model.points3D.values())
    poses, intrinsics = [], []
    for im in images:
        w2c = np.eye(4)
        w2c[:3] = im.cam_from_world().matrix()
        poses.append(np.linalg.inv(w2c))
        intrinsics.append(model.cameras[im.camera_id].calibration_matrix())
    return Geometry(
        [im.name for im in images],
        np.asarray(poses),
        np.asarray(intrinsics),
        np.asarray([p.xyz for p in points]).reshape(-1, 3),
        np.asarray([p.color for p in points], dtype=np.uint8).reshape(-1, 3),
        {
            "method": method,
            "registered": len(images),
            "points3d": len(points),
            "reprojection_error_px": model.compute_mean_reprojection_error(),
            "mean_track_length": model.compute_mean_track_length(),
            "pose_source": "estimated_from_train_RGB",
            "gt_cameras_consumed": False,
        },
    )


class IncrementalMapper:
    def __init__(self, num_threads=4):
        self.num_threads = num_threads

    def reconstruct(self, graph: TrackGraph, images: Path, output: Path) -> Geometry:
        import pycolmap

        output.mkdir(parents=True, exist_ok=True)
        db_path = output / "database.db"
        if db_path.exists():
            raise FileExistsError(db_path)
        # Explicit names avoid import_images resolving AFS symlinks into relative
        # paths outside the isolated train folder. No EXIF or GT calibration.
        from PIL import Image

        width, height = Image.open(images / graph.names[0]).size
        with pycolmap.Database.open(db_path) as db:
            camera = pycolmap.Camera(
                model="SIMPLE_PINHOLE",
                width=width,
                height=height,
                params=[1.2 * max(width, height), width / 2, height / 2],
            )
            camera.has_prior_focal_length = False
            camera_id = db.write_camera(camera)
            for name in graph.names:
                db.write_image(pycolmap.Image(name=name, camera_id=camera_id))
            ids = {im.name: im.image_id for im in db.read_all_images()}
            for name in graph.names:
                # COLMAP pixel centers are offset by +0.5 from OpenCV coordinates.
                db.write_keypoints(
                    ids[name], np.asarray(graph.points[name], dtype=np.float32).reshape(-1, 2) + 0.5
                )
            for (a, b), matches in graph.pairs.items():
                if len(matches) >= 15:
                    db.write_matches(ids[a], ids[b], matches)
        pairs = output / "pairs.txt"
        pairs.write_text("".join(f"{a} {b}\n" for (a, b), m in graph.pairs.items() if len(m) >= 15))
        pycolmap.verify_matches(str(db_path), str(pairs))
        models = pycolmap.incremental_mapping(
            db_path,
            images,
            output / "models",
            options={
                "num_threads": self.num_threads,
                "min_model_size": 3,
                "ba_refine_focal_length": True,
                "ba_refine_principal_point": False,
                "ba_refine_extra_params": False,
            },
        )
        if not models:
            raise RuntimeError("NO_RECONSTRUCTION: no geometrically consistent model")
        best = max(models.values(), key=lambda m: m.num_reg_images())
        best.write(output)
        geometry = from_colmap(best, "incremental_sfm")
        geometry.metadata["components"] = len(models)
        return geometry


class PairwiseReconstruction(ReconstructionMethod):
    def __init__(self, matcher, name, mapper=None):
        self.matcher, self.name = matcher, name
        self.mapper = mapper or IncrementalMapper()

    def reconstruct(self, images: Path, output: Path) -> Geometry:
        paths = sorted(images.glob("*.png"))
        if len(paths) < 3:
            raise ValueError("At least three training images are required")
        total_pairs = len(paths) * (len(paths) - 1) // 2
        graph = TrackGraph([p.name for p in paths])
        match_dir = output / "matches"
        match_dir.mkdir(parents=True, exist_ok=True)
        self.matcher.evidence_root = output / "model_inputs"
        start = time.monotonic()
        self.matcher.prepare(paths)
        feature_seconds = time.monotonic() - start
        summaries = []
        for index, (a, b) in enumerate(itertools.combinations(paths, 2)):
            match_start = time.monotonic()
            result = self.matcher.match(a, b)
            cached_mask = getattr(self.matcher, "cached_inliers", None)
            mask = verify(result) if cached_mask is None else cached_mask
            graph.add(a.name, b.name, result, mask)
            np.savez_compressed(
                match_dir / f"{a.stem}__{b.stem}.npz",
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
                    "seconds": time.monotonic() - match_start,
                }
            )
            # Fixed visualization pairs shared across methods, selected without GT.
            if index in [0, 1, 4, 8, 17, 34]:
                from PIL import Image

                from .core import PairMatches

                displays = getattr(self.matcher, "display_paths", {})
                da, db = displays.get(a, a), displays.get(b, b)
                wa, ha = Image.open(da).size
                wb, hb = Image.open(db).size
                shown = PairMatches(
                    result.first * np.array([wa / 512, ha / 512]),
                    result.second * np.array([wb / 512, hb / 512]),
                    result.scores,
                )
                draw_matches(da, db, shown, mask, output / "visuals" / f"pair_{index:03d}.png")
            if index % 30 == 0:
                save_json(
                    output / "progress.json",
                    {
                        "stage": "matching",
                        "pairs": index + 1,
                        "total_pairs": total_pairs,
                        "elapsed_seconds": time.monotonic() - start,
                    },
                )
                print(f"{self.name}: {index + 1}/{total_pairs} pairs", flush=True)
        save_json(
            output / "matching.json",
            {
                "method": self.name,
                "feature_seconds": feature_seconds,
                "total_seconds": time.monotonic() - start,
                "pairs": summaries,
                "track_consolidation": "nearest original-image pixel; one-to-one per pair",
                "reused_pairs": getattr(self.matcher, "reused", 0),
                "timing_note": "Measured current work; reused pair inference time is excluded.",
            },
        )
        geometry = self.mapper.reconstruct(graph, images, output / "sfm")
        geometry.metadata.update(
            method=self.name,
            input_views=len(paths),
            registration_rate=len(geometry.names) / len(paths),
        )
        return geometry
