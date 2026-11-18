import numpy as np

from reconlab.data import IdentityPreprocessor, JsonDatasetLoader
from reconlab.features import SIFTExtractor
from reconlab.matching import FundamentalVerifier, RatioMatcher
from reconlab.types import FeatureSet, MatchSet


def test_real_sift_recovers_known_translation(dataset, tmp_path):
    prepared = IdentityPreprocessor().prepare(
        JsonDatasetLoader().load(dataset)[0], tmp_path / "prepared"
    )
    extractor = SIFTExtractor(max_keypoints=500)
    a, b = [extractor.extract(view) for view in prepared.views[:2]]
    matches = RatioMatcher().match(a, b)
    assert len(matches.indices) > 10
    differences = b.keypoints[matches.indices[:, 1]] - a.keypoints[matches.indices[:, 0]]
    np.testing.assert_allclose(np.median(differences, axis=0), [5, 0], atol=0.25)
    verified = FundamentalVerifier().verify(a, b, matches)
    assert verified.inliers.sum() > 8


def test_ransac_rejects_constructed_outliers():
    rng = np.random.default_rng(7)
    xyz = rng.uniform([-1, -1, 3], [1, 1, 7], (100, 3))

    def project(points):
        return (points[:, :2] / points[:, 2:] * 200 + [160, 120]).astype(np.float32)

    a, b = project(xyz), project(xyz + [0.4, 0.1, 0.0])
    b[-20:] = rng.uniform([0, 0], [320, 240], (20, 2))
    features = [
        FeatureSet(p, np.zeros((100, 128), dtype=np.float32), np.ones(100), (320, 240))
        for p in (a, b)
    ]
    matches = MatchSet(np.column_stack((np.arange(100), np.arange(100))), np.ones(100))
    verified = FundamentalVerifier(threshold_px=0.5).verify(*features, matches)
    assert verified.inliers[:80].sum() >= 75
    assert verified.inliers[-20:].sum() <= 5


def test_empty_features():
    feature = FeatureSet(np.empty((0, 2)), np.empty((0, 128)), np.empty(0), (320, 240))
    matches = RatioMatcher().match(feature, feature)
    assert matches.indices.shape == (0, 2)
    assert FundamentalVerifier().verify(feature, feature, matches).fundamental is None
