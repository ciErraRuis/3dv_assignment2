"""Camera trajectories and GIF export. A GIF is a display encoding, not a metric input."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ValidationError
from .io import file_hash, write_json
from .types import RenderArtifact


def validate_camera_path(data: dict) -> None:
    for key in ("render_width", "render_height"):
        if type(data.get(key)) is not int or data[key] < 1:
            raise ValidationError(f"{key} must be a positive integer")
    if (
        not isinstance(data.get("seconds"), (float, int))
        or not math.isfinite(data["seconds"])
        or data["seconds"] <= 0
    ):
        raise ValidationError("Camera-path duration must be finite and positive")
    if not data.get("camera_path") or data.get("camera_type") != "perspective":
        raise ValidationError("Expected a nonempty perspective camera path")
    if data.get("crop") is not None:
        raise ValidationError("This renderer does not apply implicit scene cropping")
    for frame in data["camera_path"]:
        matrix = np.asarray(frame.get("camera_to_world"), dtype=float)
        if matrix.size != 16 or not np.isfinite(matrix).all():
            raise ValidationError("Each camera needs a finite 4x4 camera_to_world matrix")
        matrix = matrix.reshape(4, 4)
        if (
            not np.allclose(matrix[3], [0, 0, 0, 1])
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-5)
        ):
            raise ValidationError("Camera matrix must be a rigid right-handed transform")
        if not 0 < frame.get("fov", 0) < 180:
            raise ValidationError("Perspective field of view must be in (0, 180)")


class OrbitCameraPath:
    """OpenGL camera convention; explicit center/radius in the backend's world frame."""

    def build(
        self,
        center: tuple[float, float, float],
        radius: float,
        elevation_deg: float,
        frames: int,
        fps: float,
        width: int,
        height: int,
        fov_deg: float = 50,
    ) -> dict:
        if (
            radius <= 0
            or not math.isfinite(radius)
            or type(frames) is not int
            or frames < 2
            or fps <= 0
        ):
            raise ValidationError("Orbit requires positive radius/fps and at least two frames")
        if abs(elevation_deg) >= 89:
            raise ValidationError("Orbit elevation must be within (-89, 89) degrees")
        target = np.asarray(center, dtype=float)
        if target.shape != (3,) or not np.isfinite(target).all():
            raise ValidationError("Orbit center must have three finite coordinates")
        elevation = math.radians(elevation_deg)
        path = []
        for angle in np.linspace(0, 2 * math.pi, frames, endpoint=False):
            position = target + radius * np.array(
                [
                    math.cos(angle) * math.cos(elevation),
                    math.sin(angle) * math.cos(elevation),
                    math.sin(elevation),
                ]
            )
            backward = (position - target) / radius
            right = np.cross([0, 0, 1], backward)
            right /= np.linalg.norm(right)
            up = np.cross(backward, right)
            matrix = np.eye(4)
            matrix[:3, :3] = np.column_stack((right, up, backward))
            matrix[:3, 3] = position
            path.append(
                {
                    "camera_to_world": matrix.flatten().tolist(),
                    "fov": fov_deg,
                    "aspect": width / height,
                }
            )
        result = {
            "camera_type": "perspective",
            "render_width": width,
            "render_height": height,
            "seconds": frames / fps,
            "camera_path": path,
        }
        validate_camera_path(result)
        return result


class GifExporter:
    def export(self, render: RenderArtifact, output: Path) -> Path:
        if not render.frames or not math.isfinite(render.fps) or not 0 < render.fps <= 100:
            raise ValidationError("GIF requires frames and 0 < fps <= 100")
        if output.exists():
            raise ValidationError(f"Refusing to overwrite GIF: {output}")
        images = []
        for path in render.frames:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        if len({image.size for image in images}) != 1:
            raise ValidationError("GIF frames must have identical dimensions; no automatic resize")
        # GIF durations are integer centiseconds. Alternate durations to bound cumulative error.
        ticks = [
            round((i + 1) * 100 / render.fps) - round(i * 100 / render.fps)
            for i in range(len(images))
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(
            output,
            save_all=True,
            append_images=images[1:],
            duration=[10 * n for n in ticks],
            loop=0,
            optimize=False,
            disposal=2,
        )
        write_json(
            output.with_suffix(".gif.json"),
            {
                "source_frames": [{"path": p, "sha256": file_hash(p)} for p in render.frames],
                "requested_fps": render.fps,
                "source_frame_count": len(images),
                "source_duration_seconds": len(images) / render.fps,
                "gif_duration_seconds": sum(ticks) / 100,
                "encoding": "GIF palette quantization; centisecond timing; identical frames may be coalesced by encoder. PNG sequence is authoritative.",
                "render_metadata": render.metadata,
            },
        )
        return output
