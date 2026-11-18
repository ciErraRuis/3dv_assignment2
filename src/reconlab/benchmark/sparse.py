"""Sparse input manifests, pair-local reuse, and six-image GT triangulation."""

from __future__ import annotations

import itertools
import json
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .core import Geometry, ImageMatcher, PairMatches, save_json, sha256
from .view_ablation import ViewSubset


class SparseDataset:
    """Select from the frozen training pool; retain unchanged held-out views."""

    @staticmethod
    def create(source: Path, destination: Path, count=6, sampling="uniform_orbit"):
        original = json.loads((source / "transforms.json").read_text())
        provenance = json.loads((source / "inputs.json").read_text())
        assert sha256(source / "transforms.json") == provenance["transforms_sha256"]
        train = sorted(
            Path(f["file_path"]).name for f in original["frames"] if f["split"] == "train"
        )
        assert len(train) == 36
        selected = ViewSubset.select(train, count, sampling)
        frames = [
            f
            for f in original["frames"]
            if f["split"] != "train" or Path(f["file_path"]).name in selected
        ]
        assert [sum(f["split"] == s for f in frames) for s in ["train", "val", "test"]] == [
            count,
            6,
            6,
        ]
        destination.mkdir(parents=True, exist_ok=True)
        keep = {f["file_path"] for f in frames}
        records = [r for r in provenance["images"] if r["path"] in keep]
        for record in records:
            raw = source / record["path"]
            assert sha256(raw) == record["sha256"]
            assert Image.open(raw).size == (512, 512)
            target = destination / record["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(raw)
        splits = {
            s + "_filenames": [f["file_path"] for f in frames if f["split"] == s]
            for s in ["train", "val", "test"]
        }
        save_json(destination / "transforms.json", {**original, **splits, "frames": frames})
        save_json(
            destination / "inputs.json",
            {
                **provenance,
                "images": records,
                "transforms_sha256": sha256(destination / "transforms.json"),
                "parent_inputs": str(source / "inputs.json"),
                "parent_inputs_sha256": sha256(source / "inputs.json"),
            },
        )
        orbit = destination / "orbit.json"
        if not orbit.exists():
            orbit.symlink_to(source / "orbit.json")
        selection = {
            "training_names": selected,
            "training_pool_names": train,
            "pool_indices": [train.index(n) for n in selected],
            "input_views": count,
            "sampling": sampling,
            "size": [512, 512],
            "validation_views": 6,
            "test_views": 6,
            "crop": False,
            "selection_uses_gt_cameras": False,
        }
        save_json(destination / "selection.json", selection)
        return selection


class CachedPairMatcher(ImageMatcher):
    """Reuse only matches of two unchanged images; never reuse scene poses."""

    def __init__(self, name, config, sources, weights):
        self.name, self.config, self.sources, self.weights = name, config, sources, weights
        self.fresh = None
        self.cached_inliers = None
        self.reused = 0

    def prepare(self, paths):
        self.paths = paths
        self.uid = self.config.get("object_uid", paths[0].parent.parent.name)
        self.old = Path(self.config["pair_cache_root"]) / "results" / self.name / self.uid
        self.reused = 0
        self.cache_hashes = {}
        manifest = (
            Path(self.config["pair_cache_root"]) / "inputs" / self.uid / "input_manifest.json"
        )
        expected = {r["name"]: r["sha256"] for r in json.loads(manifest.read_text())["frames"]}
        for path in paths:
            if sha256(path) != expected[path.name]:
                raise ValueError("Pair cache input differs: " + str(path))
        self.display_paths = {}
        for path in paths:
            for image in sorted((self.old / "model_inputs").glob("*/" + path.name)):
                target = self.evidence_root / image.parent.name / image.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(image, target)
                info = image.with_suffix(".json")
                if info.exists():
                    shutil.copy2(info, target.with_suffix(".json"))
                self.display_paths[path] = target

    def match(self, first, second):
        self.cached_inliers = None
        path = self.old / "matches" / f"{first.stem}__{second.stem}.npz"
        if path.exists():
            with np.load(path) as data:
                result = PairMatches(data["first"], data["second"], data["scores"])
                self.cached_inliers = np.array(data["inliers"], dtype=bool)
            if len(self.cached_inliers) != len(result.scores):
                raise ValueError("Corrupt cached inlier mask")
            self.reused += 1
            self.cache_hashes[str(path)] = sha256(path)
            save_json(
                self.evidence_root.parent / "pair_cache_provenance.json",
                {
                    "source": str(self.old),
                    "reused_pairs": self.reused,
                    "pair_sha256": self.cache_hashes,
                    "allowed": "Pair-local matches and actual per-image model input evidence only",
                    "estimated_cameras_reused": False,
                    "selected_names": [p.name for p in self.paths],
                },
            )
            return result
        if self.fresh is None:
            from .matchers import create_matcher

            self.fresh = create_matcher(self.name, self.sources, self.weights)
        signature = tuple(str(p) for p in self.paths)
        if getattr(self, "fresh_signature", None) != signature:
            self.fresh.evidence_root = self.evidence_root
            self.fresh.prepare(self.paths)
            self.fresh_signature = signature
        result = self.fresh.match(first, second)
        self.display_paths.update(getattr(self.fresh, "display_paths", {}))
        return result


class GroundTruthTriangulator:
    """Common GS initialization from selected RGB matches and known cameras only."""

    def build(self, source: Path, matches: Path, output: Path):
        import cv2

        data = json.loads((source / "transforms.json").read_text())
        frames = sorted(
            (f for f in data["frames"] if f["split"] == "train"), key=lambda f: f["file_path"]
        )
        assert len(frames) == 6
        names = [Path(f["file_path"]).name for f in frames]
        poses = np.stack([np.asarray(f["transform_matrix"], dtype=float) for f in frames])
        poses[:, :3, 1:3] *= -1
        k = np.array(
            [[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]], dtype=float
        )
        w2c = np.linalg.inv(poses)
        projections = k @ w2c[:, :3]
        center = np.median(poses[:, :3, 3], axis=0)
        radius = np.median(np.linalg.norm(poses[:, :3, 3] - center, axis=1))
        points, colors, evidence = [], [], []
        for i, j in itertools.combinations(range(len(names)), 2):
            path = matches / f"{Path(names[i]).stem}__{Path(names[j]).stem}.npz"
            with np.load(path) as pair:
                a, b = pair["first"].astype(float), pair["second"].astype(float)
            if not len(a):
                continue
            xyz_h = cv2.triangulatePoints(projections[i], projections[j], a.T, b.T)
            with np.errstate(divide="ignore", invalid="ignore"):
                xyz = (xyz_h[:3] / xyz_h[3]).T
                valid = np.isfinite(xyz).all(1) & (
                    np.linalg.norm(xyz - center, axis=1) < 3 * radius
                )
                for index, pixels in [(i, a), (j, b)]:
                    camera = xyz @ w2c[index, :3, :3].T + w2c[index, :3, 3]
                    uv = camera @ k.T
                    uv = uv[:, :2] / uv[:, 2:3]
                    valid &= (camera[:, 2] > 0.01) & (np.linalg.norm(uv - pixels, axis=1) <= 2.0)
            good = xyz[valid]
            image = np.asarray(Image.open(source / frames[i]["file_path"]).convert("RGB"))
            xy = np.rint(a[valid]).astype(int).clip(0, 511)
            points.append(good)
            colors.append(image[xy[:, 1], xy[:, 0]])
            evidence.append({"pair": path.name, "sha256": sha256(path), "kept": len(good)})
        xyz, rgb = np.concatenate(points), np.concatenate(colors)
        if len(xyz) < 20:
            raise ValueError("Fewer than20 points triangulated from the selected six images")
        _, unique = np.unique(
            np.rint(xyz / (radius * 0.002)).astype(np.int64), axis=0, return_index=True
        )
        xyz, rgb = xyz[unique], rgb[unique]
        if len(xyz) > 20000:
            keep = np.random.default_rng(42).choice(len(xyz), 20000, replace=False)
            xyz, rgb = xyz[keep], rgb[keep]
        geometry = Geometry(
            names,
            poses,
            np.repeat(k[None], len(names), axis=0),
            xyz,
            rgb,
            {
                "method": "GT_camera_triangulation_of_six_view_LoMa_matches",
                "input_views": 6,
                "gt_cameras_consumed": True,
                "gt_mesh_consumed": False,
                "train_names": names,
                "reprojection_threshold_px": 2.0,
                "pairs": evidence,
                "points": len(xyz),
            },
        )
        geometry.save(output)
        return geometry


class SparseGeometryStudy:
    """Small-view geometry comparisons with exact reuse of prior completed subsets."""

    def __init__(self, config):
        self.config = config

    def run(self, method):
        from .sfm import PairwiseReconstruction
        from .visuals import export_geometry

        root = Path(self.config["output"])
        old = Path(self.config["pair_cache_root"])
        for obj in self.config["objects"]:
            uid = obj["uid"]
            pool = old / "inputs" / uid / "images"
            available = sorted(p.name for p in pool.glob("*.png"))
            assert len(available) == 36
            config = {**self.config, "object_uid": uid}
            matcher = CachedPairMatcher(method, config, Path(config["sources"]), root / "weights")
            reconstruction = PairwiseReconstruction(matcher, method)
            for count in [3, 5, 6]:
                for sampling in ["uniform_orbit", "contiguous_arc"]:
                    key = f"{sampling}_{count:02d}"
                    target = root / "view_ablation" / method / uid / key
                    if (target / "status.json").exists():
                        continue
                    selected = ViewSubset.select(available, count, sampling)
                    prior = old / "view_ablation" / method / uid / key
                    if (prior / "status.json").exists():
                        manifest = json.loads((prior / "input_manifest.json").read_text())
                        if manifest["names"] != selected:
                            raise ValueError(
                                "Prior view selection does not match the current protocol"
                            )
                        # Every subset was independently reconstructed; these are not36-view poses.
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.symlink_to(prior, target_is_directory=True)
                        continue
                    target.mkdir(parents=True, exist_ok=True)
                    images = target / "images"
                    images.mkdir(exist_ok=True)
                    for name in selected:
                        p = images / name
                        if not p.exists():
                            p.symlink_to(pool / name)
                    save_json(
                        target / "input_manifest.json",
                        {
                            "names": selected,
                            "count": count,
                            "sampling": sampling,
                            "pool_indices": [available.index(n) for n in selected],
                            "source_manifest": str(pool.parent / "input_manifest.json"),
                            "GT_cameras_supplied": False,
                            "cache_protocol": "Only unchanged pair-local matches reused; selected-subset SfM solved independently",
                        },
                    )
                    started = time.monotonic()
                    try:
                        geometry = reconstruction.reconstruct(images, target)
                        geometry.save(target)
                        export_geometry(
                            geometry, Path(config["original_source_inputs"]) / uid, target
                        )
                        status = {"state": "COMPLETE", **geometry.metadata}
                    except Exception as error:
                        (target / "error.log").write_text(traceback.format_exc())
                        status = {"state": "FAILED", "error": str(error)}
                    save_json(
                        target / "status.json",
                        {
                            **status,
                            "input_views": count,
                            "seconds": time.monotonic() - started,
                            "checked_at": time.time(),
                            "matching_runtime_included": False,
                        },
                    )
