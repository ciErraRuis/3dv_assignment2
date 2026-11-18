"""Descriptor matching and geometric verification are separate operations."""

from __future__ import annotations

import numpy as np

from .errors import DependencyError, ValidationError
from .features import require_cv2
from .interfaces import FeatureMatcher, GeometricVerifier
from .types import FeatureSet, MatchSet, VerifiedMatches


def empty_matches() -> MatchSet:
    return MatchSet(np.empty((0, 2), dtype=np.int32), np.empty(0, dtype=np.float32))


class RatioMatcher(FeatureMatcher):
    def __init__(self, ratio: float = 0.8, mutual: bool = True) -> None:
        if not 0 < ratio < 1:
            raise ValidationError("ratio must be in (0, 1)")
        self.ratio, self.mutual, self.cv2 = ratio, mutual, require_cv2()

    def match(self, first: FeatureSet, second: FeatureSet) -> MatchSet:
        if len(first.keypoints) < 2 or len(second.keypoints) < 2:
            return empty_matches()
        matcher = self.cv2.BFMatcher(self.cv2.NORM_L2)
        a, b = first.descriptors.astype(np.float32), second.descriptors.astype(np.float32)
        forward = matcher.knnMatch(a, b, k=2)
        reverse = {m.queryIdx: m.trainIdx for m in matcher.match(b, a)} if self.mutual else {}
        candidates = []
        for pair in forward:
            if len(pair) != 2:
                continue
            m, n = pair
            if m.distance < self.ratio * n.distance and (
                not self.mutual or reverse.get(m.trainIdx) == m.queryIdx
            ):
                candidates.append(
                    (1.0 - m.distance / max(n.distance, 1e-12), m.queryIdx, m.trainIdx)
                )
        # HLoc matches0 requires a one-to-one mapping, including non-mutual mode.
        used, pairs, scores = set(), [], []
        for score, i, j in sorted(candidates, reverse=True):
            if j not in used:
                used.add(j)
                pairs.append((i, j))
                scores.append(score)
        return MatchSet(
            np.asarray(pairs, dtype=np.int32).reshape(-1, 2), np.asarray(scores, dtype=np.float32)
        )


class LightGlueMatcher(FeatureMatcher):
    def __init__(self, features: str = "superpoint", device: str = "cpu") -> None:
        try:
            import torch
            from lightglue import LightGlue
        except ImportError as exc:
            raise DependencyError("Install a pinned LightGlue environment separately") from exc
        from pathlib import Path

        if features != "superpoint" or device not in {"cpu", "cuda"}:
            raise ValidationError(
                "This adapter supports SuperPoint descriptors only; SIFT-LightGlue additionally needs scale/orientation metadata"
            )
        # Upstream version v0.1_arxiv renames '.' to '-' for cached filenames.
        weight = Path(torch.hub.get_dir()) / "checkpoints" / f"{features}_lightglue_v0-1_arxiv.pth"
        if not weight.is_file():
            raise DependencyError(f"Pre-cache the reviewed LightGlue weights: {weight}")
        if device == "cuda":
            from .runtime import inspect_gpu

            inspect_gpu()
        self.torch, self.device = torch, device
        self.model = LightGlue(features=features).eval().to(device)

    def match(self, first: FeatureSet, second: FeatureSet) -> MatchSet:
        if not len(first.keypoints) or not len(second.keypoints):
            return empty_matches()

        def pack(features):
            return {
                key: self.torch.as_tensor(value, dtype=self.torch.float32, device=self.device)[None]
                for key, value in {
                    "keypoints": features.keypoints,
                    "descriptors": features.descriptors,
                    "image_size": features.image_size,
                }.items()
            }

        with self.torch.inference_mode():
            result = self.model({"image0": pack(first), "image1": pack(second)})
        return MatchSet(
            result["matches"][0].cpu().numpy().astype(np.int32), result["scores"][0].cpu().numpy()
        )


class FundamentalVerifier(GeometricVerifier):
    def __init__(
        self, threshold_px: float = 1.0, confidence: float = 0.999, seed: int = 42
    ) -> None:
        if threshold_px <= 0 or not 0 < confidence < 1:
            raise ValidationError("Invalid RANSAC threshold/confidence")
        self.threshold, self.confidence, self.seed = threshold_px, confidence, seed
        self.cv2 = require_cv2()

    def verify(self, a: FeatureSet, b: FeatureSet, matches: MatchSet) -> VerifiedMatches:
        inliers = np.zeros(len(matches.indices), dtype=bool)
        if len(matches.indices) < 8:
            return VerifiedMatches(matches, inliers, None)
        xy_a, xy_b = a.keypoints[matches.indices[:, 0]], b.keypoints[matches.indices[:, 1]]
        self.cv2.setRNGSeed(self.seed)
        matrix, mask = self.cv2.findFundamentalMat(
            xy_a, xy_b, self.cv2.FM_RANSAC, self.threshold, self.confidence
        )
        if (
            matrix is None
            or matrix.shape != (3, 3)
            or mask is None
            or not np.isfinite(matrix).all()
        ):
            return VerifiedMatches(matches, inliers, None)
        return VerifiedMatches(matches, mask.ravel().astype(bool), matrix)
