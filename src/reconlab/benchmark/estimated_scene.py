"""Train the selected Gaussian method on estimated cameras; GT is evaluation-only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import sha256, similarity_alignment
from .gt_rendering import GroundTruthScene


class EstimatedScene(GroundTruthScene):
    def __init__(self, source: Path, geometry):
        self.source, self.geometry = source, geometry
        self.data = json.loads((source / "transforms.json").read_text())
        records = json.loads((source / "inputs.json").read_text())
        assert sha256(source / "transforms.json") == records["transforms_sha256"]
        self.hashes = {r["path"]: r["sha256"] for r in records["images"]}
        by_name = {
            Path(f["file_path"]).name: f for f in self.data["frames"] if f["split"] == "train"
        }
        assert len(geometry.names) >= 3 and set(geometry.names) <= set(by_name)
        train = []
        for name, pose in zip(geometry.names, geometry.c2w):
            gl = pose.copy()
            gl[:3, 1:3] *= -1
            train.append(
                {
                    "file_path": by_name[name]["file_path"],
                    "split": "train",
                    "transform_matrix": gl.tolist(),
                }
            )
        self.frames = {
            "train": train,
            **{s: [f for f in self.data["frames"] if f["split"] == s] for s in ["val", "test"]},
        }
        self.center = np.median(geometry.c2w[:, :3, 3], axis=0)
        self.scale = float(np.median(np.linalg.norm(geometry.c2w[:, :3, 3] - self.center, axis=1)))
        if not np.isfinite(self.scale) or self.scale <= 1e-8:
            raise ValueError("Degenerate estimated camera gauge")
        points = (geometry.points - self.center) / self.scale
        valid = np.isfinite(points).all(1) & (np.linalg.norm(points, axis=1) < 10)
        self.points = points[valid]
        self.colors = geometry.colors[valid].astype(np.float32) / 255
        if len(self.points) < 20:
            raise ValueError("Too few estimated initialization points")
        if len(self.points) > 20000:
            keep = np.random.default_rng(42).choice(len(self.points), 20000, replace=False)
            self.points, self.colors = self.points[keep], self.colors[keep]
        self.train_intrinsics = geometry.intrinsics.astype(np.float32)
        self.k = np.array(
            [
                [self.data["fl_x"], 0, self.data["cx"]],
                [0, self.data["fl_y"], self.data["cy"]],
                [0, 0, 1],
            ],
            np.float32,
        )
        self.camera_source = "Estimated intrinsics/extrinsics from selected training RGB; GT used only for evaluation-coordinate alignment"
        self.initialization_description = (
            "Selected frontend's six-view point cloud; no GT mesh, no36-view geometry"
        )
        self._alignment = None

    def eval_view(self, matrix):
        if self._alignment is None:
            by_name = {
                Path(f["file_path"]).name: f for f in self.data["frames"] if f["split"] == "train"
            }
            gt = np.stack([self.pose(by_name[n]["transform_matrix"]) for n in self.geometry.names])
            self._alignment = similarity_alignment(self.geometry.c2w[:, :3, 3], gt[:, :3, 3])
        scale, rotation, translation = self._alignment
        pose = self.pose(matrix)
        estimated = np.eye(4)
        estimated[:3, :3] = rotation.T @ pose[:3, :3]
        estimated[:3, 3] = (
            (rotation.T @ (pose[:3, 3] - translation) / scale) - self.center
        ) / self.scale
        return np.linalg.inv(estimated).astype(np.float32)
