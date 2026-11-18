"""Native GLUEMAP entry and export; no replacement of its global solver."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import sys
import time
from pathlib import Path

import numpy as np

from .core import PairMatches, draw_matches, save_json, verify
from .matchers import record_pixels
from .sfm import from_colmap


class InputEvidence:
    """Intercept upstream preprocessing outputs without changing any pixels."""

    def __init__(self, output):
        import gluemap.utils.load_fn as loaders

        self.evidence_root = output / "model_inputs"
        self.known = {}
        self.seen = set()
        load_paths = loaders.load_and_preprocess_images
        load_inner = loaders.load_and_preprocess_images_inner
        load_high = loaders.load_and_preprocess_images_1024

        def paths(names, *args, **kwargs):
            result = load_paths(names, *args, **kwargs)
            for name, original, tensor in zip(names, result[1], result[0]):
                self.known[self.key(original)] = Path(name)
                self.record(Path(name), tensor)
            return result

        def inner(originals, *args, **kwargs):
            result = load_inner(originals, *args, **kwargs)
            self.capture(originals, result[0])
            return result

        def high(originals, *args, **kwargs):
            result = load_high(originals, *args, **kwargs)
            self.capture(originals, result[0])
            return result

        loaders.load_and_preprocess_images = paths
        loaders.load_and_preprocess_images_inner = inner
        loaders.load_and_preprocess_images_1024 = high

    @staticmethod
    def key(tensor):
        return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()

    def capture(self, originals, tensors):
        for original, tensor in zip(originals, tensors):
            path = self.known.get(self.key(original))
            if path is not None:
                self.record(path, tensor)

    def record(self, path, tensor):
        stage = f"rgb_{tensor.shape[-2]}x{tensor.shape[-1]}"
        if (path, stage) not in self.seen:
            record_pixels(self, path, tensor, stage)
            self.seen.add((path, stage))


def export_tracks(model, images, output):
    """Actual image observations of native refined tracks, not projected GT points."""
    observations = {}
    for image in model.images.values():
        if image.has_pose:
            observations[Path(image.name).name] = {
                point.point3D_id: np.asarray(point.xy) - 0.5
                for point in image.points2D
                if point.has_point3D()
            }
    paths = sorted(images.glob("*.png"))
    (output / "matches").mkdir(exist_ok=True)
    rows = []
    for index, (a, b) in enumerate(itertools.combinations(paths, 2)):
        first, second = observations.get(a.name, {}), observations.get(b.name, {})
        common = sorted(first.keys() & second.keys())
        matches = PairMatches(
            [first[k] for k in common], [second[k] for k in common], np.ones(len(common))
        )
        mask = verify(matches)
        np.savez_compressed(
            output / "matches" / f"{a.stem}__{b.stem}.npz",
            first=matches.first,
            second=matches.second,
            scores=matches.scores,
            inliers=mask,
        )
        rows.append(
            {"first": a.name, "second": b.name, "matches": len(common), "inliers": int(mask.sum())}
        )
        if index in [0, 1, 4, 8, 17, 34]:
            draw_matches(a, b, matches, mask, output / "visuals" / f"pair_{index:03d}.png")
    save_json(
        output / "matching.json",
        {
            "method": "gluemap",
            "pairs": rows,
            "correspondence_source": "actual native refined track observations (Pi3/VGGSfM tracks snapped to SIFT); no GT or synthetic reprojections",
            "verification": "posthoc diagnostic; native global solver uses its own filtering",
        },
    )


def main():
    parser = argparse.ArgumentParser()
    for key in ["source", "weights", "images", "output"]:
        parser.add_argument("--" + key, type=Path, required=True)
    options = parser.parse_args()
    sys.path.insert(0, str(options.source))
    sys.path.insert(0, str(options.source.parent / "lightglue"))
    import pycolmap
    import torch
    import yaml

    # Even a cached hub repo without an explicit ref may query GitHub for its
    # default branch. Use the frozen local DINOv2 source; weights stay identical.
    hub_load = torch.hub.load

    def offline_hub(repo, model, *args, **kwargs):
        if isinstance(repo, str) and repo.split(":")[0] == "facebookresearch/dinov2":
            repo = str(options.source.parent / "dinov2")
            kwargs["source"] = "local"
        return hub_load(repo, model, *args, **kwargs)

    torch.hub.load = offline_hub
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    InputEvidence(options.output)
    from gluemap.controllers.gluemap_impl import run_inference_pipeline
    from gluemap.controllers.image_retrieval import run_preprocessing_pipeline
    from gluemap.datasets.twoview import BaseTwoViewDataset
    from gluemap.utils.cli import get_args_parser, parse_args_with_config
    from gluemap.utils.gpu import init_distributed

    config = yaml.safe_load((options.source / "configs/base.yaml").read_text())
    native = options.output / "native"
    native.mkdir(exist_ok=True)
    config.update(
        images_path=str(options.images),
        write_path=str(native),
        path_feedforward=str(options.weights / "pi3/model.safetensors"),
        path_retrieval=str(options.weights / "dino_salad.ckpt"),
        path_tracker=str(options.weights / "vggsfm_tracker/vggsfm_v2_tracker.pt"),
        path_dg=str(options.weights / "doppelgangers/checkpoint-dg+visym.pth"),
        use_gt_intrinsics=False,
        gt_path=None,
        gt_intrinsics_path=None,
        nodist=True,
        num_workers=4,
        temp_path=str(native / "tmp"),
    )
    config_file = options.output / "native_config.yaml"
    config_file.write_text(yaml.safe_dump(config))
    sys.argv = ["gluemap-demo", "--config", str(config_file)]
    args = parse_args_with_config(get_args_parser())
    rank, world_size, device, dtype = init_distributed(args)
    args.curr_processed = args.curr_path = args.write_path
    start = time.monotonic()
    run_preprocessing_pipeline(args, world_size, rank)
    dataset = BaseTwoViewDataset(args)
    run_inference_pipeline(args, dataset, world_size, rank, device, dtype)
    model = pycolmap.Reconstruction(str(native / "gluemap_aba"))
    geometry = from_colmap(model, "gluemap_native_global")
    geometry.names = [Path(name).name for name in geometry.names]
    input_count = len(list(options.images.glob("*.png")))
    geometry.metadata.update(
        seconds=time.monotonic() - start,
        input_views=input_count,
        registration_rate=len(geometry.names) / input_count,
    )
    geometry.save(options.output)
    export_tracks(model, options.images, options.output)


if __name__ == "__main__":
    main()
