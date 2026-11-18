"""Nerfstudio 1.1.5 adapters; training/renderer integrations require GPU validation."""

from __future__ import annotations

import sys
from pathlib import Path

from ..data import verify_inputs
from ..errors import ValidationError
from ..interfaces import Reconstructor, Renderer
from ..io import file_hash, read_json, write_json
from ..runtime import CommandRunner, inspect_gpu, require_version
from ..types import ModelArtifact, PreparedObject, RenderArtifact, SfMArtifact


class NerfstudioReconstructor(Reconstructor):
    method = ""

    def __init__(
        self,
        iterations: int = 30000,
        save_every: int = 2000,
        foreground_loss: bool = False,
        seed: int = 42,
    ) -> None:
        if (
            type(iterations) is not int
            or iterations < 1
            or type(save_every) is not int
            or save_every < 1
        ):
            raise ValidationError("iterations/save_every must be positive integers")
        self.iterations, self.save_every = iterations, save_every
        self.foreground_loss, self.seed = foreground_loss, seed

    def build_command(self, obj: PreparedObject, sfm: SfMArtifact, output: Path) -> list[str]:
        """Pure command construction; does not install software, train, or modify inputs."""
        if sfm.requires_undistortion:
            raise ValidationError(
                "This identity-input adapter accepts reviewed pinhole inputs only. Distorted inputs need an approved undistortion stage with new provenance; do not relabel the camera model to bypass this check."
            )
        expected = {v.image.name for v in obj.views}
        if set(sfm.registered_names) != expected:
            raise ValidationError(
                "Some views failed SfM registration; no silent removal of train/val/test views"
            )
        if any(not any(v.split == split for v in obj.views) for split in ("train", "val", "test")):
            raise ValidationError("Training requires separate, nonempty train/val/test lists")
        if self.foreground_loss and any(v.mask is None for v in obj.views):
            raise ValidationError("Foreground loss requires masks for every view")
        command = [
            str(Path(sys.executable).with_name("ns-train")),
            self.method,
            "--data",
            str(obj.root),
            "--output-dir",
            str(output),
            "--experiment-name",
            obj.id,
            "--timestamp",
            "run",
            "--max-num-iterations",
            str(self.iterations),
            "--steps-per-save",
            str(self.save_every),
            "--save-only-latest-checkpoint",
            "False",
            "--vis",
            "tensorboard",
            "--machine.seed",
            str(self.seed),
            "--pipeline.model.camera-optimizer.mode",
            "off",
        ]
        if self.method == "splatfacto":
            command += ["--pipeline.model.num-downscales", "0"]
        command += [
            "colmap",
            "--data",
            str(obj.root),
            "--colmap-path",
            str(sfm.root),
            "--downscale-factor",
            "1",
            "--orientation-method",
            "none",
            "--center-method",
            "none",
            "--auto-scale-poses",
            "False",
            "--assume-colmap-world-coordinate-convention",
            "False",
        ]
        if self.foreground_loss:
            command += ["--masks-path", "masks"]
        return command

    def fit(self, obj: PreparedObject, sfm: SfMArtifact, output: Path) -> ModelArtifact:
        command = self.build_command(obj, sfm, output)
        require_version("nerfstudio", "1.1.5")
        gpu = inspect_gpu()
        verify_inputs(obj)
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "gpu.json", gpu)
        # Prepared split files override Nerfstudio's default interval split.
        # They are immutable and may be shared by several reconstruction methods.
        result = CommandRunner().run(command, obj.root, output / "train.log")
        config = output / obj.id / self.method / "run" / "config.yml"
        step = self.iterations - 1  # Nerfstudio's last saved trainer step is zero-based.
        checkpoint = config.parent / "nerfstudio_models" / f"step-{step:09d}.ckpt"
        if not config.is_file() or not checkpoint.is_file():
            raise ValidationError(
                "Training returned without the exact final config/checkpoint; inspect train.log"
            )
        artifact = ModelArtifact(
            self.method,
            config,
            checkpoint,
            file_hash(checkpoint),
            step,
            {
                "nerfstudio": "1.1.5",
                "config_sha256": file_hash(config),
                "training_seconds": result["elapsed_seconds"],
                "geometry_scope": "all_selected_views_transductive",
                "foreground_loss": self.foreground_loss,
                "input_provenance": str(obj.provenance),
            },
        )
        write_json(output / "model.json", artifact)
        return artifact


class SplatfactoReconstructor(NerfstudioReconstructor):
    method = "splatfacto"


class NerfactoReconstructor(NerfstudioReconstructor):
    method = "nerfacto"


def load_model(path: Path) -> ModelArtifact:
    raw = read_json(path)
    model = ModelArtifact(
        raw["backend"],
        Path(raw["config"]),
        Path(raw["checkpoint"]),
        raw["checkpoint_sha256"],
        raw["step"],
        raw.get("metadata", {}),
    )
    if (
        model.backend not in {"splatfacto", "nerfacto"}
        or type(model.step) is not int
        or model.step < 0
    ):
        raise ValidationError("Invalid Nerfstudio model artifact")
    if file_hash(model.checkpoint) != model.checkpoint_sha256:
        raise ValidationError("Checkpoint content no longer matches model.json")
    expected_config_hash = model.metadata.get("config_sha256")
    if not expected_config_hash or file_hash(model.config) != expected_config_hash:
        raise ValidationError("Config content no longer matches model.json")
    return model


class NerfstudioRenderer(Renderer):
    def render(self, model: ModelArtifact, camera_path: Path, output: Path) -> RenderArtifact:
        require_version("nerfstudio", "1.1.5")
        gpu = inspect_gpu()
        if file_hash(model.checkpoint) != model.checkpoint_sha256:
            raise ValidationError("Checkpoint changed before rendering")
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "model.json", model)
        write_json(output / "gpu.json", gpu)
        command = [
            sys.executable,
            "-m",
            "reconlab.backends.ns_render_worker",
            "--model",
            str(output / "model.json"),
            "--camera-path",
            str(camera_path),
            "--output",
            str(output),
        ]
        CommandRunner().run(command, output, output / "render.log")
        raw = read_json(output / "render.json")
        return RenderArtifact(tuple(Path(p) for p in raw["frames"]), raw["fps"], raw["metadata"])
