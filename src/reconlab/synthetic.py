"""Reproducible synthetic multi-view inputs; mesh rendering is not reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ValidationError
from .io import file_hash, read_json, write_json
from .rendering import OrbitCameraPath


@dataclass(frozen=True)
class SyntheticPolicy:
    width: int = 512
    height: int = 512
    views_per_ring: int = 24
    elevations: tuple[float, ...] = (20.0, 40.0)
    radius: float = 2.2
    fov_deg: float = 50.0
    orbit_frames: int = 72
    fps: int = 24
    samples: int = 32
    seed: int = 42

    def cameras(self) -> tuple[list[dict], dict]:
        frames = []
        for ring, elevation in enumerate(self.elevations):
            orbit = OrbitCameraPath().build(
                (0, 0, 0),
                self.radius,
                elevation,
                self.views_per_ring,
                self.fps,
                self.width,
                self.height,
                self.fov_deg,
            )
            for index, camera in enumerate(orbit["camera_path"]):
                number = len(frames)
                frames.append(
                    {
                        "file_path": f"images/view_{number:03d}.png",
                        "transform_matrix": np.array(camera["camera_to_world"])
                        .reshape(4, 4)
                        .tolist(),
                        "split": "test" if index % 8 == 0 else "val" if index % 8 == 4 else "train",
                        "ring": ring,
                        "azimuth_deg": index * 360 / self.views_per_ring,
                        "elevation_deg": elevation,
                    }
                )
        novel = OrbitCameraPath().build(
            (0, 0, 0),
            self.radius,
            30.0,
            self.orbit_frames,
            self.fps,
            self.width,
            self.height,
            self.fov_deg,
        )
        return frames, novel

    def transforms(self) -> dict:
        frames, _ = self.cameras()
        focal = self.height / (2 * np.tan(np.deg2rad(self.fov_deg) / 2))
        return {
            "camera_model": "OPENCV",
            "w": self.width,
            "h": self.height,
            "fl_x": float(focal),
            "fl_y": float(focal),
            "cx": self.width / 2,
            "cy": self.height / 2,
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "frames": frames,
            **{
                f"{split}_filenames": [f["file_path"] for f in frames if f["split"] == split]
                for split in ("train", "val", "test")
            },
        }


class PosedImageDataset:
    """Validated RGB images and fixed pinhole poses, independent of any mesh loader."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.data = read_json(self.root / "transforms.json")
        self.provenance = read_json(self.root / "inputs.json")
        self.width, self.height = self.data["w"], self.data["h"]
        self.frames = self.data["frames"]
        names = [frame["file_path"] for frame in self.frames]
        if len(names) != len(set(names)):
            raise ValidationError("Duplicate input images")
        splits = [set(self.data[f"{s}_filenames"]) for s in ("train", "val", "test")]
        if any(not split for split in splits) or set.union(*splits) != set(names):
            raise ValidationError("Every image must belong to a nonempty explicit split")
        if any(splits[i] & splits[j] for i in range(3) for j in range(i)):
            raise ValidationError("Train/val/test overlap")
        if any(self.data.get(k, 0) != 0 for k in ("k1", "k2", "k3", "k4", "p1", "p2")):
            raise ValidationError("Known-camera adapter requires undistorted pinhole images")
        if self.provenance["transforms_sha256"] != file_hash(self.root / "transforms.json"):
            raise ValidationError("Cameras or splits changed after input rendering")

    def load_split(self, split: str) -> tuple[np.ndarray, np.ndarray]:
        names = set(self.data[f"{split}_filenames"])
        images, cameras = [], []
        hashes = {f["path"]: f["sha256"] for f in self.provenance["images"]}
        for frame in self.frames:
            name = frame["file_path"]
            if name not in names:
                continue
            path = self.root / name
            if file_hash(path) != hashes[name]:
                raise ValidationError(f"Input changed: {path}")
            with Image.open(path) as image:
                if image.mode != "RGB" or image.size != (self.width, self.height):
                    raise ValidationError(f"Unexpected RGB dimensions: {path}")
                images.append(np.asarray(image).copy())
            c2w = np.asarray(frame["transform_matrix"], dtype=np.float32)
            if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
                raise ValidationError("Invalid camera matrix")
            if not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5):
                raise ValidationError("Camera rotation is not orthogonal")
            cameras.append(c2w)
        return np.stack(images), np.stack(cameras)

    @property
    def intrinsics(self) -> np.ndarray:
        d = self.data
        return np.array(
            [[d["fl_x"], 0, d["cx"]], [0, d["fl_y"], d["cy"]], [0, 0, 1]], dtype=np.float32
        )


def write_input_evidence(root: Path, policy: SyntheticPolicy, source: dict) -> None:
    transforms = read_json(root / "transforms.json")
    images = []
    for frame in transforms["frames"]:
        path = root / frame["file_path"]
        with Image.open(path) as image:
            if image.mode != "RGB" or image.size != (policy.width, policy.height):
                raise ValidationError("Renderer violated approved image policy")
        images.append(
            {"path": frame["file_path"], "split": frame["split"], "sha256": file_hash(path)}
        )
    write_json(
        root / "inputs.json",
        {
            "kind": "actual_model_inputs",
            "policy": asdict(policy),
            "source": source,
            "transforms_sha256": file_hash(root / "transforms.json"),
            "images": images,
            "processing": "Full-frame RGBA composited over white; no resize, crop or frame duplication",
            "camera_convention": "OpenGL camera-to-world; renderer converts to OpenCV once",
        },
    )
