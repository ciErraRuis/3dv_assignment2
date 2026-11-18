"""HLoc imports the exact exported correspondences into COLMAP, then verifies them."""

from __future__ import annotations

import inspect
from pathlib import Path

from ..data import verify_inputs
from ..errors import DependencyError, ValidationError
from ..interfaces import PoseEstimator
from ..io import file_hash, write_json
from ..types import FrontendArtifacts, PreparedObject, SfMArtifact


class HlocPoseEstimator(PoseEstimator):
    def __init__(
        self,
        camera_model: str = "SIMPLE_RADIAL",
        shared_camera: bool = False,
        mapper_options: dict | None = None,
    ) -> None:
        self.camera_model, self.shared_camera = camera_model, shared_camera
        self.mapper_options = mapper_options or {}

    def estimate(
        self, obj: PreparedObject, frontend: FrontendArtifacts, output: Path
    ) -> SfMArtifact:
        try:
            import pycolmap
            from hloc import reconstruction
        except ImportError as exc:
            raise DependencyError(
                "SfM requires a separately installed compatible HLoc + pycolmap environment"
            ) from exc
        verify_inputs(obj)
        if output.exists():
            raise ValidationError(f"Refusing to overwrite an existing SfM database: {output}")
        if self.shared_camera and len({(v.width, v.height) for v in obj.views}) != 1:
            raise ValidationError("Shared camera requires identical image dimensions")
        model = reconstruction.main(
            output,
            obj.root / "images",
            frontend.pairs,
            frontend.features,
            frontend.matches,
            camera_mode=pycolmap.CameraMode.SINGLE
            if self.shared_camera
            else pycolmap.CameraMode.AUTO,
            image_list=[v.image.name for v in obj.views],
            image_options={"camera_model": self.camera_model},
            mapper_options=self.mapper_options,
            skip_geometric_verification=False,
        )
        if model is None or model.num_reg_images() < 2:
            raise ValidationError("SfM did not produce a usable reconstruction")
        names = tuple(sorted(model.images[image_id].name for image_id in model.reg_image_ids()))
        # Conservative: unfamiliar camera types must not silently enter an undistorter.
        distorted = any(
            camera.model_name not in {"PINHOLE", "SIMPLE_PINHOLE"}
            for camera in model.cameras.values()
        )
        statistics = {
            "registered": len(names),
            "input_views": len(obj.views),
            "registration_rate": len(names) / len(obj.views),
            "points3d": model.num_points3D(),
            "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
            "geometry_scope": "all_selected_views_transductive",
            "hloc_source_sha256": file_hash(Path(inspect.getfile(reconstruction))),
            "pycolmap_version": pycolmap.__version__,
            "files_sha256": {p.name: file_hash(p) for p in output.glob("*.bin")},
        }
        artifact = SfMArtifact(output, names, statistics, distorted)
        write_json(output / "artifact.json", artifact)
        return artifact
