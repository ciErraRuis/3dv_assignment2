"""Manifest loading and lossless RGB staging. No implicit resize, crop, or EXIF rotation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import ValidationError
from .interfaces import DatasetLoader, Preprocessor
from .io import exact_keys, file_hash, read_json, safe_id, write_json
from .types import ObjectViews, PreparedObject, PreparedView, View


class JsonDatasetLoader(DatasetLoader):
    """Paths resolve against the manifest directory, never the caller's CWD."""

    def load(self, manifest: Path) -> tuple[ObjectViews, ...]:
        data = read_json(manifest)
        exact_keys(data, {"schema_version", "objects"}, {"schema_version", "objects"}, "manifest")
        if (
            data["schema_version"] != 1
            or not isinstance(data["objects"], list)
            or not data["objects"]
        ):
            raise ValidationError("Expected schema_version=1 and a nonempty objects list")
        objects, object_ids = [], set()
        for item in data["objects"]:
            exact_keys(item, {"id", "views"}, {"id", "views"}, "object")
            object_id = safe_id(item["id"])
            if object_id in object_ids:
                raise ValidationError(f"Duplicate object id {object_id}")
            object_ids.add(object_id)
            views, ids, paths = [], set(), set()
            if not isinstance(item["views"], list) or not item["views"]:
                raise ValidationError(f"Object {object_id} has no views")
            for raw in item["views"]:
                exact_keys(
                    raw,
                    {"id", "image", "split", "mask", "azimuth_deg"},
                    {"id", "image", "split"},
                    "view",
                )
                view_id = safe_id(raw["id"])
                image = self._path(manifest, raw["image"])
                mask = self._path(manifest, raw["mask"]) if raw.get("mask") else None
                if raw["split"] not in {"train", "val", "test"}:
                    raise ValidationError("Every view needs an explicit train/val/test split")
                if view_id in ids or image in paths:
                    raise ValidationError(f"Duplicate view id/path in {object_id}: {view_id}")
                azimuth = raw.get("azimuth_deg")
                if azimuth is not None and (
                    type(azimuth) not in (float, int) or not math.isfinite(azimuth)
                ):
                    raise ValidationError("azimuth_deg must be a finite number")
                ids.add(view_id)
                paths.add(image)
                views.append(View(view_id, image, raw["split"], mask, azimuth))
            if not any(v.split == "train" for v in views):
                raise ValidationError(f"Object {object_id} has no training views")
            objects.append(ObjectViews(object_id, tuple(views)))
        return tuple(objects)

    @staticmethod
    def _path(manifest: Path, raw: str) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ValidationError("Image/mask path must be a nonempty string")
        path = (manifest.parent / raw).resolve()
        if not path.is_file():
            raise ValidationError(f"Missing data file: {path}")
        return path


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.getexif().get(274, 1) != 1:
            raise ValidationError(f"EXIF rotation requires a reviewed preprocessing rule: {path}")
        if image.mode not in {"RGB", "L"}:
            raise ValidationError(
                f"Mode {image.mode} requires an explicit color/alpha policy: {path}"
            )
        return np.array(image.convert("RGB"), dtype=np.uint8)


def read_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as image:
        if image.size != size or image.mode not in {"1", "L"}:
            raise ValidationError(
                f"Mask must be single-channel and match image size {size}: {path}"
            )
        array = np.array(image.convert("L"))
    if not np.isin(array, [0, 255]).all():
        raise ValidationError(f"Mask must contain only 0/255; no implicit threshold: {path}")
    if not array.any():
        raise ValidationError(f"Empty foreground mask: {path}")
    return array


def pixel_hash(array: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


class IdentityPreprocessor(Preprocessor):
    """Stage decoded RGB pixels as PNG; preserve dimensions, order, and split."""

    def prepare(self, obj: ObjectViews, output: Path) -> PreparedObject:
        output.mkdir(parents=True, exist_ok=False)
        (output / "images").mkdir()
        (output / "masks").mkdir()
        records, prepared, seen = [], [], {}
        for index, view in enumerate(obj.views):
            source_hash = file_hash(view.image)
            rgb = read_rgb(view.image)
            if file_hash(view.image) != source_hash:
                raise ValidationError(f"Source changed during staging: {view.image}")
            height, width = rgb.shape[:2]
            digest = pixel_hash(rgb)
            if digest in seen:
                raise ValidationError(
                    f"Duplicate decoded content {view.id}/{seen[digest]}; resolve leakage/redundancy explicitly"
                )
            seen[digest] = view.id
            target = output / "images" / f"{view.id}.png"
            Image.fromarray(rgb).save(target)
            mask_target, mask_source_hash = None, None
            if view.mask:
                mask_source_hash = file_hash(view.mask)
                mask = read_mask(view.mask, (width, height))
                mask_target = output / "masks" / f"{view.id}.png"
                Image.fromarray(mask).save(mask_target)
                if file_hash(view.mask) != mask_source_hash:
                    raise ValidationError(f"Mask changed during staging: {view.mask}")
            prepared.append(
                PreparedView(
                    view.id,
                    target,
                    view.split,
                    width,
                    height,
                    digest,
                    mask_target,
                    view.azimuth_deg,
                )
            )
            records.append(
                {
                    "view_id": view.id,
                    "source": view.image,
                    "source_sha256": source_hash,
                    "model_input": target,
                    "file_sha256": file_hash(target),
                    "pixel_sha256": digest,
                    "source_order_index": index,
                    "split": view.split,
                    "width": width,
                    "height": height,
                    "azimuth_deg": view.azimuth_deg,
                    "mask_source": view.mask,
                    "mask_source_sha256": mask_source_hash,
                    "mask_input": mask_target,
                    "mask_sha256": file_hash(mask_target) if mask_target else None,
                }
            )
        split_files = {}
        for split in ("train", "val", "test"):
            split_path = output / f"{split}_list.txt"
            split_path.write_text(
                "".join(v.image.name + "\n" for v in prepared if v.split == split)
            )
            split_files[split] = {
                "path": str(split_path.resolve()),
                "sha256": file_hash(split_path),
            }
        provenance = output / "inputs.json"
        write_json(
            provenance,
            {
                "object_id": obj.id,
                "split_files": split_files,
                "policy": "identity-rgb-v1",
                "resize": None,
                "crop": None,
                "fps": None,
                "duration": None,
                "media_type": "unordered_still_views",
                "encoding": "PNG of decoded RGB; no spatial transform",
                "views": records,
            },
        )
        return PreparedObject(obj.id, output, tuple(prepared), provenance)


def verify_inputs(obj: PreparedObject) -> None:
    """Recheck bytes immediately before handing inputs to a model/backend."""
    evidence = read_json(obj.provenance)
    for split in evidence.get("split_files", {}).values():
        if file_hash(Path(split["path"])) != split["sha256"]:
            raise ValidationError(f"Split file changed: {split['path']}")
    for row in evidence["views"]:
        if file_hash(Path(row["model_input"])) != row["file_sha256"]:
            raise ValidationError(f"Staged image changed: {row['model_input']}")
        if row["mask_input"] and file_hash(Path(row["mask_input"])) != row["mask_sha256"]:
            raise ValidationError(f"Staged mask changed: {row['mask_input']}")
