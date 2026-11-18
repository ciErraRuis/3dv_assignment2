"""Backend-neutral contracts. Pixel coordinates use pixel centers, (x, y)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class View:
    id: str
    image: Path
    split: str
    mask: Path | None = None
    azimuth_deg: float | None = None


@dataclass(frozen=True)
class ObjectViews:
    id: str
    views: tuple[View, ...]


@dataclass(frozen=True)
class PreparedView:
    id: str
    image: Path
    split: str
    width: int
    height: int
    pixel_sha256: str
    mask: Path | None = None
    azimuth_deg: float | None = None


@dataclass(frozen=True)
class PreparedObject:
    id: str
    root: Path
    views: tuple[PreparedView, ...]
    provenance: Path


@dataclass(frozen=True)
class FeatureSet:
    keypoints: np.ndarray  # N x 2
    descriptors: np.ndarray  # N x D
    scores: np.ndarray  # N
    image_size: tuple[int, int]  # width, height


@dataclass(frozen=True)
class MatchSet:
    indices: np.ndarray  # M x 2, unique source and target indices
    scores: np.ndarray  # M, matcher confidence (not geometric correctness)


@dataclass(frozen=True)
class VerifiedMatches:
    matches: MatchSet
    inliers: np.ndarray  # M boolean
    fundamental: np.ndarray | None


@dataclass(frozen=True)
class FrontendArtifacts:
    features: Path
    matches: Path
    pairs: Path
    summary: Path


@dataclass(frozen=True)
class SfMArtifact:
    root: Path
    registered_names: tuple[str, ...]
    statistics: dict[str, Any]
    # Distortion is checked before any backend may silently undistort images.
    requires_undistortion: bool


@dataclass(frozen=True)
class ModelArtifact:
    backend: str
    config: Path
    checkpoint: Path
    checkpoint_sha256: str
    step: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderArtifact:
    frames: tuple[Path, ...]
    fps: float
    metadata: dict[str, Any] = field(default_factory=dict)
