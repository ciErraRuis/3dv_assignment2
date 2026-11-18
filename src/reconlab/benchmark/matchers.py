"""Adapters for official pretrained matchers. Models and per-image features are reused."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .core import ImageMatcher, PairMatches, save_json, sha256


def record_pixels(matcher, source, tensor, stage):
    """Archive the exact pre-normalization array used at the model call boundary."""
    if not hasattr(matcher, "evidence_root"):
        return
    import hashlib

    if hasattr(tensor, "detach"):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    raw = np.ascontiguousarray(array)
    shown = array
    if shown.ndim == 4:
        shown = shown[0]
    if shown.ndim == 3 and shown.shape[0] in (1, 3):
        shown = shown.transpose(1, 2, 0)
    if shown.ndim == 3 and shown.shape[-1] == 1:
        shown = shown[..., 0]
    if shown.dtype != np.uint8:
        shown = (shown.clip(0, 1) * 255).round().astype(np.uint8)
    folder = matcher.evidence_root / stage
    folder.mkdir(parents=True, exist_ok=True)
    Image.fromarray(shown).save(folder / source.name)
    save_json(
        folder / (source.stem + ".json"),
        {
            "source_sha256": sha256(source),
            "shape": list(raw.shape),
            "dtype": str(raw.dtype),
            "tensor_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
            "display": "PNG rounded to 8-bit; float tensor hash retained",
            "stage": stage,
        },
    )
    if not hasattr(matcher, "display_paths"):
        matcher.display_paths = {}
    matcher.display_paths[source] = folder / source.name


def rgb_tensor(path, device="cuda"):
    import torch

    array = np.asarray(Image.open(path).convert("RGB")).copy()
    return torch.from_numpy(array).permute(2, 0, 1).float().div(255).to(device)


class OpenCvMatcher(ImageMatcher):
    def __init__(self, feature="sift"):
        import cv2

        self.cv2 = cv2
        self.extractor = {
            "sift": lambda: cv2.SIFT_create(nfeatures=4096),
            "surf": lambda: cv2.xfeatures2d.SURF_create(hessianThreshold=400),
            "orb": lambda: cv2.ORB_create(nfeatures=4096),
            "akaze": lambda: cv2.AKAZE_create(threshold=0.001),
        }[feature]()
        self.norm = cv2.NORM_HAMMING if feature in {"orb", "akaze"} else cv2.NORM_L2
        self.cache = {}

    def prepare(self, paths):
        self.cache.clear()
        for path in paths:
            gray = self.cv2.imread(str(path), self.cv2.IMREAD_GRAYSCALE)
            record_pixels(self, path, gray, "grayscale_512")
            keypoints, descriptors = self.extractor.detectAndCompute(gray, None)
            if len(keypoints) > 4096:
                order = np.argsort([-point.response for point in keypoints], kind="stable")[:4096]
                keypoints = [keypoints[i] for i in order]
                descriptors = descriptors[order]
            self.cache[path] = (
                np.asarray([x.pt for x in keypoints], dtype=np.float32).reshape(-1, 2),
                descriptors,
            )

    def match(self, first, second):
        a, da = self.cache[first]
        b, db = self.cache[second]
        if da is None or db is None or min(len(da), len(db)) < 2:
            return PairMatches([], [], [])
        matcher = self.cv2.BFMatcher(self.norm)
        reverse = {x.queryIdx: x.trainIdx for x in matcher.match(db, da)}
        pairs, scores = [], []
        for candidates in matcher.knnMatch(da, db, k=2):
            if len(candidates) != 2:
                continue
            m, n = candidates
            if m.distance < 0.8 * n.distance and reverse.get(m.trainIdx) == m.queryIdx:
                pairs.append((m.queryIdx, m.trainIdx))
                scores.append(1 - m.distance / max(n.distance, 1e-12))
        idx = np.asarray(pairs, dtype=int).reshape(-1, 2)
        return PairMatches(a[idx[:, 0]], b[idx[:, 1]], scores)


class LightGlueMatcher(ImageMatcher):
    def __init__(self, source: Path, feature="superpoint"):
        import torch

        sys.path.insert(0, str(source))
        from lightglue import ALIKED, DISK, LightGlue, SuperPoint

        self.torch = torch
        detector = {"superpoint": SuperPoint, "disk": DISK, "aliked": ALIKED}[feature]
        self.detector = detector(max_num_keypoints=4096).eval().cuda()
        self.matcher = LightGlue(features=feature).eval().cuda()
        self.cache = {}

    def prepare(self, paths):
        self.cache.clear()
        with self.torch.inference_mode():
            for path in paths:
                tensor = rgb_tensor(path)
                record_pixels(self, path, tensor, "rgb_512")
                self.cache[path] = self.detector.extract(tensor, resize=None)

    def match(self, first, second):
        with self.torch.inference_mode():
            a, b = self.cache[first], self.cache[second]
            result = self.matcher({"image0": a, "image1": b})
            pairs = result["matches"][0]
            return PairMatches(
                a["keypoints"][0, pairs[:, 0]].cpu().numpy(),
                b["keypoints"][0, pairs[:, 1]].cpu().numpy(),
                result["scores"][0].cpu().numpy(),
            )


class EfficientLoftr(ImageMatcher):
    """Official Transformers port; bypass automatic resizing for 512-square inputs."""

    def __init__(self, weights: Path):
        import torch
        from transformers import EfficientLoFTRForKeypointMatching

        self.torch = torch
        self.model = (
            EfficientLoFTRForKeypointMatching.from_pretrained(str(weights), local_files_only=True)
            .eval()
            .cuda()
        )
        self.cache = {}

    def prepare(self, paths):
        self.cache.clear()
        for path in paths:
            # Same grayscale conversion as the original OpenCV example.
            import cv2

            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray.shape != (512, 512):
                raise ValueError("Unexpected input size")
            self.cache[path] = self.torch.from_numpy(gray.copy()).float().div(255)[None].cuda()
            record_pixels(self, path, self.cache[path], "grayscale_512")

    def match(self, first, second):
        with self.torch.inference_mode():
            inputs = self.torch.stack([self.cache[first], self.cache[second]])[None]
            result = self.model(pixel_values=inputs)
            # EfficientLoFTR returns already paired, refined coordinates in the
            # SAME slot on the two sides; `matches` stores coarse-grid indices.
            a, b = result.keypoints[0, 0], result.keypoints[0, 1]
            score = result.matching_scores[0, 0]
            keep = score > 0.2
            return PairMatches(
                (a[keep] * 512).cpu().numpy(),
                (b[keep] * 512).cpu().numpy(),
                score[keep].cpu().numpy(),
            )


class LomaMatcher(ImageMatcher):
    def __init__(self, source: Path):
        import torch

        sys.path.insert(0, str(source / "src"))
        from loma import LoMa, LoMaB

        self.torch = torch
        self.model = LoMa(LoMaB(num_keypoints=4096))
        self.cache = {}
        self._capture_loader(self.model._detector, "load_image", "detector_1024")
        self._capture_loader(self.model._descriptor, "read_image", "descriptor_784")

    def _capture_loader(self, module, name, stage):
        original = getattr(module, name)

        def captured(path, *args, **kwargs):
            tensor = original(path, *args, **kwargs)
            record_pixels(self, Path(path), tensor, stage)
            return tensor

        setattr(module, name, captured)

    def prepare(self, paths):
        self.cache.clear()
        for path in paths:
            # Official path API: detector and descriptor preserve square FOV;
            # detector uses 1024, descriptor's own documented spatial scale.
            self.cache[path] = self.model.detect_and_describe(str(path), 4096)

    def match(self, first, second):
        from loma.loma import filter_matches, to_pixel_coords

        ka, da, ha, wa = self.cache[first]
        kb, db, hb, wb = self.cache[second]
        with self.torch.inference_mode():
            result = self.model(ka, kb, da, db)
            m, _, scores, _ = filter_matches(result["scores"], self.model.cfg.filter_threshold)
            keep = m[0] >= 0
            return PairMatches(
                to_pixel_coords(ka[0, keep], ha, wa).cpu().numpy(),
                to_pixel_coords(kb[0, m[0, keep]], hb, wb).cpu().numpy(),
                scores[0, keep].cpu().numpy(),
            )


class RomaV2Matcher(ImageMatcher):
    def __init__(self, source: Path):
        sys.path.insert(0, str(source / "src"))
        from romav2 import RoMaV2

        self.model = RoMaV2(RoMaV2.Cfg(setting="precise", compile=False))
        # RoMaV2.forward explicitly rejects reduced float32 matmul precision.
        # Set this after model/dependency imports, which may change torch defaults.
        import torch

        torch.set_float32_matmul_precision("highest")
        self.captured = set()
        self.model.register_forward_pre_hook(self._capture, with_kwargs=True)

    def prepare(self, paths):
        self.captured.clear()

    def _capture(self, module, args, kwargs):
        for index, path in enumerate(self.current_pair):
            if path in self.captured:
                continue
            record_pixels(self, path, args[index], "coarse_800")
            key = "img_A_hr" if index == 0 else "img_B_hr"
            record_pixels(self, path, kwargs[key], "fine_1280")
            self.captured.add(path)

    def match(self, first, second):
        self.current_pair = (first, second)
        predictions = self.model.match(first, second)
        matches, scores, _, _ = self.model.sample(predictions, 5000)
        a, b = self.model.to_pixel_coordinates(matches, 512, 512, 512, 512)
        return PairMatches(a.cpu().numpy(), b.cpu().numpy(), scores.cpu().numpy())


def create_matcher(name: str, sources: Path, weights: Path) -> ImageMatcher:
    constructors = {
        "sift": lambda: OpenCvMatcher(),
        "surf": lambda: OpenCvMatcher("surf"),
        "orb": lambda: OpenCvMatcher("orb"),
        "akaze": lambda: OpenCvMatcher("akaze"),
        "superpoint_lightglue": lambda: LightGlueMatcher(sources / "lightglue"),
        "disk_lightglue": lambda: LightGlueMatcher(sources / "lightglue", "disk"),
        "aliked_lightglue": lambda: LightGlueMatcher(sources / "lightglue", "aliked"),
        "efficientloftr": lambda: EfficientLoftr(weights / "eloftr"),
        "loma": lambda: LomaMatcher(sources / "loma"),
        "romav2": lambda: RomaV2Matcher(sources / "romav2"),
    }
    return constructors[name]()

# Compatibility with the initial single-feature adapter names.
SiftMatcher = OpenCvMatcher
SuperPointLightGlue = LightGlueMatcher
