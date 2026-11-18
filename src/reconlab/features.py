"""Feature extraction backends. Learned models are optional and loaded once."""

from __future__ import annotations

import numpy as np

from .data import read_mask, read_rgb
from .errors import DependencyError, ValidationError
from .interfaces import FeatureExtractor
from .types import FeatureSet, PreparedView


def require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise DependencyError("Install the vision extra for OpenCV SIFT/RANSAC") from exc
    return cv2


class SIFTExtractor(FeatureExtractor):
    def __init__(self, max_keypoints: int = 4096, foreground_only: bool = False) -> None:
        if type(max_keypoints) is not int or max_keypoints < 1:
            raise ValidationError("max_keypoints must be positive")
        self.cv2 = require_cv2()
        self.extractor = self.cv2.SIFT_create(nfeatures=max_keypoints)
        self.foreground_only = foreground_only

    def extract(self, view: PreparedView) -> FeatureSet:
        rgb = read_rgb(view.image)
        gray = self.cv2.cvtColor(rgb, self.cv2.COLOR_RGB2GRAY)
        mask = None
        if self.foreground_only:
            if view.mask is None:
                raise ValidationError(f"Foreground extraction requires mask: {view.id}")
            mask = read_mask(view.mask, (view.width, view.height))
        keypoints, descriptors = self.extractor.detectAndCompute(gray, mask)
        points = np.array([p.pt for p in keypoints], dtype=np.float32).reshape(-1, 2)
        scores = np.array([p.response for p in keypoints], dtype=np.float32)
        if descriptors is None:
            descriptors = np.empty((0, 128), dtype=np.float32)
        return FeatureSet(points, descriptors, scores, (view.width, view.height))


class SuperPointExtractor(FeatureExtractor):
    """Requires pre-cached upstream weights; never downloads as a side effect."""

    def __init__(
        self, max_keypoints: int = 4096, foreground_only: bool = False, device: str = "cpu"
    ) -> None:
        try:
            import torch
            from lightglue import SuperPoint
        except ImportError as exc:
            raise DependencyError(
                "SuperPoint needs the separately installed, pinned LightGlue environment"
            ) from exc
        from pathlib import Path

        weight = Path(torch.hub.get_dir()) / "checkpoints" / "superpoint_v1.pth"
        if not weight.is_file():
            raise DependencyError(f"Pre-cache the reviewed SuperPoint weights: {weight}")
        if max_keypoints < 1 or device not in {"cpu", "cuda"}:
            raise ValidationError("Invalid SuperPoint keypoint count/device")
        if device == "cuda":
            from .runtime import inspect_gpu

            inspect_gpu()
        self.torch, self.device = torch, device
        self.model = SuperPoint(max_num_keypoints=max_keypoints).eval().to(device)
        self.foreground_only = foreground_only

    def extract(self, view: PreparedView) -> FeatureSet:
        rgb = read_rgb(view.image)
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255).to(self.device)
        with self.torch.inference_mode():
            result = self.model.extract(tensor, resize=None)
        points = result["keypoints"][0].cpu().numpy()
        descriptors = result["descriptors"][0].cpu().numpy()
        scores = result["keypoint_scores"][0].cpu().numpy()
        if self.foreground_only:
            if view.mask is None:
                raise ValidationError(f"Missing foreground mask: {view.id}")
            mask = read_mask(view.mask, (view.width, view.height))
            xy = np.rint(points).astype(int)
            valid = (
                (xy[:, 0] >= 0)
                & (xy[:, 0] < view.width)
                & (xy[:, 1] >= 0)
                & (xy[:, 1] < view.height)
            )
            keep = np.zeros(len(points), dtype=bool)
            keep[valid] = mask[xy[valid, 1], xy[valid, 0]] > 0
            points, descriptors, scores = points[keep], descriptors[keep], scores[keep]
        return FeatureSet(points, descriptors, scores, (view.width, view.height))
