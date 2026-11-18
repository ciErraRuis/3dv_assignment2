"""Small shared contracts; matching, reconstruction and evaluation stay independent."""

from __future__ import annotations

import hashlib
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False))
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class PairMatches:
    first: np.ndarray
    second: np.ndarray
    scores: np.ndarray

    def __post_init__(self):
        self.first = np.asarray(self.first, dtype=np.float32).reshape(-1, 2)
        self.second = np.asarray(self.second, dtype=np.float32).reshape(-1, 2)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        if not (len(self.first) == len(self.second) == len(self.scores)):
            raise ValueError("Inconsistent correspondence arrays")
        if not all(np.isfinite(a).all() for a in (self.first, self.second, self.scores)):
            raise ValueError("Non-finite correspondence")


class ImageMatcher(ABC):
    """The input contract contains RGB paths only, never GT camera metadata."""

    def prepare(self, paths: list[Path]) -> None:
        pass

    @abstractmethod
    def match(self, first: Path, second: Path) -> PairMatches: ...


@dataclass
class Geometry:
    names: list[str]
    c2w: np.ndarray  # OpenCV: +Z forward, +Y down
    intrinsics: np.ndarray  # one 3x3 matrix per registered view, original pixel coordinates
    points: np.ndarray
    colors: np.ndarray
    metadata: dict

    def save(self, output: Path):
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output / "geometry.npz",
            names=np.asarray(self.names),
            c2w=self.c2w,
            intrinsics=self.intrinsics,
            points=self.points,
            colors=self.colors,
        )
        save_json(output / "geometry.json", self.metadata)

    @classmethod
    def load(cls, output: Path):
        with np.load(output / "geometry.npz", allow_pickle=False) as data:
            return cls(
                data["names"].tolist(),
                data["c2w"],
                data["intrinsics"],
                data["points"],
                data["colors"],
                json.loads((output / "geometry.json").read_text()),
            )


class ReconstructionMethod(ABC):
    @abstractmethod
    def reconstruct(self, images: Path, output: Path) -> Geometry: ...


def verify(matches: PairMatches, threshold: float = 1.5) -> np.ndarray:
    import cv2

    if len(matches.scores) < 8:
        return np.zeros(len(matches.scores), dtype=bool)
    cv2.setRNGSeed(42)
    try:
        matrix, mask = cv2.findFundamentalMat(
            matches.first, matches.second, cv2.USAC_MAGSAC, threshold, 0.999, 10000
        )
    except cv2.error as error:
        # OpenCV's USAC solver throws for certain degenerate minimal samples.
        # Reject the pair as geometrically unverified; do not abort the object.
        if "!model.empty()" not in str(error):
            raise
        return np.zeros(len(matches.scores), dtype=bool)
    if matrix is None or mask is None or matrix.shape != (3, 3):
        return np.zeros(len(matches.scores), dtype=bool)
    return mask.ravel().astype(bool)


def draw_matches(first: Path, second: Path, matches: PairMatches, mask: np.ndarray, output: Path):
    """Visualize actual input pixels and actual verified matches, deterministically."""
    a, b = Image.open(first).convert("RGB"), Image.open(second).convert("RGB")
    canvas = Image.new("RGB", (a.width + b.width, max(a.height, b.height)), "white")
    canvas.paste(a)
    canvas.paste(b, (a.width, 0))
    draw = ImageDraw.Draw(canvas)
    indices = np.flatnonzero(mask)
    if len(indices) > 100:
        indices = indices[np.linspace(0, len(indices) - 1, 100).astype(int)]
    for i in indices:
        x, y = matches.first[i]
        u, v = matches.second[i]
        u += a.width
        draw.line((float(x), float(y), float(u), float(v)), fill=(44, 190, 165), width=1)
        for px, py in [(x, y), (u, v)]:
            draw.ellipse((px - 2, py - 2, px + 2, py + 2), outline=(44, 190, 165))
    draw.rectangle((0, 0, 430, 22), fill="white")
    draw.text(
        (5, 5), f"{first.name} / {second.name}: {int(mask.sum())}/{len(mask)} inliers", fill="black"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


class TrackGraph:
    """Consolidate pair-specific detections into shared per-image pixel identities.

    Dense matches are snapped to a 1 px lattice in the ORIGINAL 512 px image.
    This makes track construction explicit; no pair-local IDs enter the mapper.
    The snapping policy is shared across all frontends and recorded in reports.
    """

    def __init__(self, names: list[str], size: int = 512):
        self.names, self.size = names, size
        self.points = {name: [] for name in names}
        self.indices = {name: {} for name in names}
        self.pairs: dict[tuple[str, str], np.ndarray] = {}

    def add(self, a: str, b: str, matches: PairMatches, mask: np.ndarray):
        rows, used_a, used_b = [], set(), set()
        order = np.argsort(-matches.scores, kind="stable")
        for i in order:
            if not mask[i]:
                continue
            pa, pb = np.rint(matches.first[i]).astype(int), np.rint(matches.second[i]).astype(int)
            if min(*pa, *pb) < 0 or max(*pa, *pb) >= self.size:
                continue
            ids = []
            for name, point in ((a, pa), (b, pb)):
                key = tuple(point)
                if key not in self.indices[name]:
                    self.indices[name][key] = len(self.points[name])
                    self.points[name].append(point.astype(float))
                ids.append(self.indices[name][key])
            if ids[0] in used_a or ids[1] in used_b:
                continue
            used_a.add(ids[0])
            used_b.add(ids[1])
            rows.append(ids)
        self.pairs[(a, b)] = np.asarray(rows, dtype=np.uint32).reshape(-1, 2)


def similarity_alignment(source: np.ndarray, target: np.ndarray):
    """Umeyama Sim(3), fitted only on reconstructed TRAIN camera centers."""
    if source.shape != target.shape or len(source) < 3:
        raise ValueError("At least three corresponding camera centers required")
    x, y = source - source.mean(0), target - target.mean(0)
    u, singular, vh = np.linalg.svd(y.T @ x / len(x))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vh)
    rotation = u @ correction @ vh
    scale = (singular * np.diag(correction)).sum() / (x * x).sum(1).mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate camera alignment")
    translation = target.mean(0) - scale * rotation @ source.mean(0)
    return float(scale), rotation, translation
