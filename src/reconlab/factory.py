"""Built-in registrations; GPU libraries are imported only by adapter methods."""

from .backends.gsplat_known import GaussianRenderer, KnownCameraGaussianReconstructor
from .backends.hloc import HlocPoseEstimator
from .backends.nerfstudio import NerfactoReconstructor, NerfstudioRenderer, SplatfactoReconstructor
from .data import IdentityPreprocessor, JsonDatasetLoader
from .evaluation import PSNRMetric
from .features import SIFTExtractor, SuperPointExtractor
from .interfaces import (
    DatasetLoader,
    FeatureExtractor,
    FeatureMatcher,
    GeometricVerifier,
    KnownPoseReconstructor,
    Metric,
    PoseEstimator,
    Preprocessor,
    Reconstructor,
    Renderer,
    ViewSelector,
)
from .matching import FundamentalVerifier, LightGlueMatcher, RatioMatcher
from .registry import ComponentRegistry
from .selection import AllViews, FractionSelector


def default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    for interface, name, cls in (
        (DatasetLoader, "json", JsonDatasetLoader),
        (Preprocessor, "identity", IdentityPreprocessor),
        (ViewSelector, "all", AllViews),
        (ViewSelector, "fraction", FractionSelector),
        (FeatureExtractor, "sift", SIFTExtractor),
        (FeatureExtractor, "superpoint", SuperPointExtractor),
        (FeatureMatcher, "ratio", RatioMatcher),
        (FeatureMatcher, "lightglue", LightGlueMatcher),
        (GeometricVerifier, "fundamental", FundamentalVerifier),
        (PoseEstimator, "hloc", HlocPoseEstimator),
        (Reconstructor, "splatfacto", SplatfactoReconstructor),
        (Reconstructor, "nerfacto", NerfactoReconstructor),
        (KnownPoseReconstructor, "gsplat_known", KnownCameraGaussianReconstructor),
        (Renderer, "gsplat_known", GaussianRenderer),
        (Renderer, "nerfstudio", NerfstudioRenderer),
        (Metric, "psnr", PSNRMetric),
    ):
        registry.register(interface, name, cls)
    return registry
