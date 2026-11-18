"""Reference-image metrics with explicit correspondence, split, and mask semantics."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from .data import read_mask, read_rgb
from .errors import ValidationError
from .interfaces import Metric
from .io import file_hash, read_json, write_json


class PSNRMetric(Metric):
    def compute(self, prediction, reference, mask=None) -> dict:
        a, b = np.asarray(prediction), np.asarray(reference)
        if a.shape != b.shape or a.ndim != 3 or a.shape[2] != 3:
            raise ValidationError("PSNR requires equally sized HxWx3 RGB arrays")
        if a.dtype != np.uint8 or b.dtype != np.uint8:
            raise ValidationError("PSNR contract is uint8 RGB in [0,255]")
        error = ((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != a.shape[:2] or mask.dtype != bool or not mask.any():
                raise ValidationError("Metric mask must be a nonempty HxW boolean array")
            error = error[mask]
        mse = float(error.mean())
        return {
            "mse": mse,
            "psnr_db": None if mse == 0 else -10 * math.log10(mse),
            "perfect_match": mse == 0,
            "pixels": int(error.size // 3),
        }


class ImageEvaluator:
    """A pair manifest prevents accidental comparisons based only on directory sorting."""

    def __init__(self, metric: Metric | None = None) -> None:
        self.metric = metric or PSNRMetric()

    def evaluate(self, manifest: Path, output: Path) -> dict:
        data = read_json(manifest)
        if data.get("schema_version") != 1 or not data.get("pairs"):
            raise ValidationError("Metric manifest needs schema_version=1 and explicit pairs")
        rows, ids = [], set()
        for pair in data["pairs"]:
            if pair["id"] in ids or pair.get("split") != "test":
                raise ValidationError("Evaluation IDs must be unique and split must be test")
            ids.add(pair["id"])
            prediction, reference = [
                (manifest.parent / pair[key]).resolve() for key in ("prediction", "reference")
            ]
            a, b = read_rgb(prediction), read_rgb(reference)
            row = {
                "id": pair["id"],
                "prediction": str(prediction),
                "reference": str(reference),
                "prediction_sha256": file_hash(prediction),
                "reference_sha256": file_hash(reference),
                "full": self.metric.compute(a, b),
            }
            if pair.get("mask"):
                mask_path = (manifest.parent / pair["mask"]).resolve()
                mask = read_mask(mask_path, (b.shape[1], b.shape[0])) > 0
                row["foreground"] = self.metric.compute(a, b, mask)
                row["mask_sha256"] = file_hash(mask_path)
            rows.append(row)
        result = {
            "status": "EVALUATED",
            "metric": type(self.metric).__name__,
            "pairs": rows,
            "coverage": len(rows),
            "note": "Image appearance metrics, not geometry accuracy. Zero MSE has infinite PSNR represented by null + perfect_match.",
        }
        write_json(output, result)
        return result
