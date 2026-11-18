"""Replace one stage by implementing its abstract class; no pipeline edits required."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .synthetic import PosedImageDataset

from .types import (
    FeatureSet,
    FrontendArtifacts,
    MatchSet,
    ModelArtifact,
    ObjectViews,
    PreparedObject,
    PreparedView,
    RenderArtifact,
    SfMArtifact,
    VerifiedMatches,
    View,
)


class DatasetLoader(ABC):
    @abstractmethod
    def load(self, manifest: Path) -> tuple[ObjectViews, ...]: ...


class ViewSelector(ABC):
    @abstractmethod
    def select(self, views: Sequence[View]) -> tuple[View, ...]: ...


class Preprocessor(ABC):
    @abstractmethod
    def prepare(self, obj: ObjectViews, output: Path) -> PreparedObject: ...


class FeatureExtractor(ABC):
    @abstractmethod
    def extract(self, view: PreparedView) -> FeatureSet: ...


class FeatureMatcher(ABC):
    @abstractmethod
    def match(self, first: FeatureSet, second: FeatureSet) -> MatchSet: ...


class GeometricVerifier(ABC):
    @abstractmethod
    def verify(self, a: FeatureSet, b: FeatureSet, matches: MatchSet) -> VerifiedMatches: ...


class PoseEstimator(ABC):
    @abstractmethod
    def estimate(
        self, obj: PreparedObject, frontend: FrontendArtifacts, output: Path
    ) -> SfMArtifact: ...


class Reconstructor(ABC):
    @abstractmethod
    def fit(self, obj: PreparedObject, sfm: SfMArtifact, output: Path) -> ModelArtifact: ...


class KnownPoseReconstructor(ABC):
    """Reconstruction from supplied cameras, without claiming to have run SfM."""

    @abstractmethod
    def fit(self, dataset: "PosedImageDataset", output: Path) -> ModelArtifact: ...


class Renderer(ABC):
    @abstractmethod
    def render(self, model: ModelArtifact, camera_path: Path, output: Path) -> RenderArtifact: ...


class Metric(ABC):
    @abstractmethod
    def compute(self, prediction, reference, mask=None) -> dict: ...
