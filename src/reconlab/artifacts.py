"""Validated loading allows reuse of exact inputs and poses across reconstruction methods."""

from pathlib import Path

from .data import verify_inputs
from .errors import ValidationError
from .io import file_hash, read_json
from .types import PreparedObject, PreparedView, SfMArtifact


class ArtifactStore:
    def load_prepared(self, root: Path) -> PreparedObject:
        root = root.resolve()
        provenance = root / "inputs.json"
        raw = read_json(provenance)
        if raw.get("policy") != "identity-rgb-v1":
            raise ValidationError(
                "This loader only supports identity-rgb-v1; extend it for other input policies"
            )
        views = tuple(
            PreparedView(
                row["view_id"],
                Path(row["model_input"]),
                row["split"],
                row["width"],
                row["height"],
                row["pixel_sha256"],
                Path(row["mask_input"]) if row["mask_input"] else None,
                row.get("azimuth_deg"),
            )
            for row in raw["views"]
        )
        obj = PreparedObject(raw["object_id"], root, views, provenance)
        verify_inputs(obj)
        return obj

    def load_sfm(self, path: Path) -> SfMArtifact:
        raw = read_json(path)
        artifact = SfMArtifact(
            Path(raw["root"]),
            tuple(raw["registered_names"]),
            raw["statistics"],
            raw["requires_undistortion"],
        )
        hashes = artifact.statistics.get("files_sha256", {})
        if not {"images.bin", "cameras.bin", "points3D.bin"}.issubset(hashes):
            raise ValidationError("SfM artifact needs recorded hashes for images/cameras/points3D")
        for name, expected in hashes.items():
            if Path(name).name != name or file_hash(artifact.root / name) != expected:
                raise ValidationError(f"SfM artifact changed: {name}")
        return artifact
