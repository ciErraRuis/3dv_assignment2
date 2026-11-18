"""Isolated optional worker. Pin the exact checkpoint, render each camera once."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..errors import ValidationError
from ..io import file_hash, read_json, write_json
from ..types import RenderArtifact
from .nerfstudio import load_model


def main() -> None:
    import numpy as np
    import torch
    from nerfstudio.cameras.camera_paths import get_path_from_json
    from nerfstudio.utils.eval_utils import eval_setup
    from PIL import Image

    from ..rendering import validate_camera_path

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--camera-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = load_model(args.model)
    raw = read_json(args.camera_path)
    validate_camera_path(raw)

    def pin(config):
        if config.get_checkpoint_dir().resolve() != model.checkpoint.parent.resolve():
            raise ValidationError("Config points to another checkpoint directory")
        config.load_step = model.step
        return config

    _, pipeline, loaded_path, loaded_step = eval_setup(
        model.config, test_mode="inference", update_config_callback=pin
    )
    if loaded_path.resolve() != model.checkpoint.resolve() or loaded_step != model.step:
        raise ValidationError("Backend loaded a checkpoint different from the requested one")
    cameras = get_path_from_json(raw).to(pipeline.device)
    paths = []
    with torch.inference_mode():
        for index in range(len(cameras)):
            rgb = pipeline.model.get_outputs_for_camera(cameras[index : index + 1])["rgb"]
            pixels = (rgb.clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)
            path = args.output / f"frame_{index:05d}.png"
            Image.fromarray(pixels).save(path)
            paths.append(path)
    artifact = RenderArtifact(
        tuple(paths),
        len(paths) / raw["seconds"],
        {
            "kind": "model_novel_view_render",
            "model": str(args.model),
            "checkpoint_sha256": model.checkpoint_sha256,
            "camera_path_sha256": file_hash(args.camera_path),
            "width": raw["render_width"],
            "height": raw["render_height"],
        },
    )
    write_json(args.output / "render.json", artifact)


if __name__ == "__main__":
    main()
