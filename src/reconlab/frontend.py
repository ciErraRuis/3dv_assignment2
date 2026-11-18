"""Extract once per view, stream pair results, and export HLoc-compatible artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .data import read_rgb, verify_inputs
from .errors import DependencyError, ValidationError
from .interfaces import FeatureExtractor, FeatureMatcher, GeometricVerifier
from .io import write_json
from .selection import PairGenerator
from .types import FeatureSet, FrontendArtifacts, PreparedObject


def validate_features(features: FeatureSet) -> None:
    n = len(features.keypoints)
    if (
        features.keypoints.shape != (n, 2)
        or features.descriptors.ndim != 2
        or features.descriptors.shape[0] != n
        or features.scores.shape != (n,)
    ):
        raise ValidationError("Feature arrays must have shapes Nx2, NxD, N")
    if not all(
        np.isfinite(a).all() for a in (features.keypoints, features.descriptors, features.scores)
    ):
        raise ValidationError("Features contain NaN/Inf")


class MatchingFrontend:
    def __init__(
        self,
        extractor: FeatureExtractor,
        matcher: FeatureMatcher,
        verifier: GeometricVerifier,
        pairs: PairGenerator,
        max_visualizations: int = 6,
    ) -> None:
        self.extractor, self.matcher, self.verifier = extractor, matcher, verifier
        self.pairs, self.max_visualizations = pairs, max_visualizations

    def run(self, obj: PreparedObject, output: Path) -> FrontendArtifacts:
        try:
            import h5py
        except ImportError as exc:
            raise DependencyError("Install h5py (vision extra) for feature/match export") from exc
        verify_inputs(obj)
        output.mkdir(parents=True, exist_ok=False)
        features_path, matches_path, pairs_path = (
            output / name for name in ("features.h5", "matches.h5", "pairs.txt")
        )
        features = []
        with h5py.File(features_path, "w") as store:
            for view in obj.views:
                feature = self.extractor.extract(view)
                validate_features(feature)
                features.append(feature)
                group = store.create_group(view.image.name)
                group.create_dataset("keypoints", data=feature.keypoints)
                group.create_dataset("descriptors", data=feature.descriptors.T)
                group.create_dataset("scores", data=feature.scores)
                group.create_dataset("image_size", data=feature.image_size)
        statistics = []
        with h5py.File(matches_path, "w") as store, pairs_path.open("w") as pairs_file:
            for pair_index, (i, j) in enumerate(self.pairs.generate(len(obj.views))):
                first, second = obj.views[i], obj.views[j]
                matches = self.matcher.match(features[i], features[j])
                indices = matches.indices
                if (
                    indices.ndim != 2
                    or indices.shape[1] != 2
                    or matches.scores.shape != (len(indices),)
                ):
                    raise ValidationError("Match indices/scores must be Mx2 and M")
                if (
                    not np.issubdtype(indices.dtype, np.integer)
                    or not np.isfinite(matches.scores).all()
                ):
                    raise ValidationError("Match indices must be integers and scores finite")
                if len(indices) and (
                    indices.min() < 0
                    or indices[:, 0].max() >= len(features[i].keypoints)
                    or indices[:, 1].max() >= len(features[j].keypoints)
                ):
                    raise ValidationError("Match index outside feature array")
                if any(len(np.unique(indices[:, col])) != len(indices) for col in (0, 1)):
                    raise ValidationError("Matches must be one-to-one")
                verified = self.verifier.verify(features[i], features[j], matches)
                group = store.create_group(f"{first.image.name}/{second.image.name}")
                matches0 = np.full(len(features[i].keypoints), -1, dtype=np.int32)
                scores0 = np.zeros(len(matches0), dtype=np.float32)
                matches0[indices[:, 0]] = indices[:, 1]
                scores0[indices[:, 0]] = matches.scores
                group.create_dataset("matches0", data=matches0)
                group.create_dataset("matching_scores0", data=scores0)
                # Diagnostic F-RANSAC is separate from COLMAP's calibrated verification.
                group.create_dataset("diagnostic_pairs", data=indices)
                group.create_dataset("diagnostic_inliers", data=verified.inliers)
                if verified.fundamental is not None:
                    group.create_dataset("diagnostic_fundamental", data=verified.fundamental)
                pairs_file.write(f"{first.image.name} {second.image.name}\n")
                count, inlier_count = len(indices), int(verified.inliers.sum())
                record = {
                    "first": first.id,
                    "second": second.id,
                    "matches": count,
                    "diagnostic_f_inliers": inlier_count,
                    "diagnostic_f_inlier_ratio": inlier_count / count if count else None,
                }
                if pair_index < self.max_visualizations:
                    image_path = output / f"pair_{pair_index:04d}.png"
                    self._visualize(
                        first.image, second.image, features[i], features[j], verified, image_path
                    )
                    record["visualization"] = str(image_path)
                statistics.append(record)
        summary = output / "summary.json"
        write_json(
            summary,
            {
                "object_id": obj.id,
                "keypoints": {v.id: len(f.keypoints) for v, f in zip(obj.views, features)},
                "pairs": statistics,
                "note": "Green=diagnostic F-RANSAC inlier, red=outlier; not ground truth. COLMAP verifies independently.",
            },
        )
        return FrontendArtifacts(features_path, matches_path, pairs_path, summary)

    @staticmethod
    def _visualize(first, second, a, b, verified, output):
        left, right = Image.fromarray(read_rgb(first)), Image.fromarray(read_rgb(second))
        canvas = Image.new(
            "RGB", (left.width + right.width, max(left.height, right.height)), "white"
        )
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        draw = ImageDraw.Draw(canvas)
        # Highest-confidence subset for readable plots; complete pairs remain in HDF5.
        subset = np.argsort(-verified.matches.scores, kind="stable")[:150]
        for k in subset:
            i, j = verified.matches.indices[k]
            x, y = a.keypoints[i]
            u, v = b.keypoints[j]
            color = "#18bba6" if verified.inliers[k] else "#e34a52"
            draw.line((float(x), float(y), float(u + left.width), float(v)), fill=color, width=1)
        canvas.save(output)
