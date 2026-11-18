"""Small known-camera 3DGS backend using gsplat 1.4.0's rasterizer/strategy.

This is an independent pilot recipe, not Nerfstudio Splatfacto or the exact
gsplat example configuration. It consumes RGB and poses, never mesh geometry.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..errors import ValidationError
from ..interfaces import KnownPoseReconstructor, Renderer
from ..io import file_hash, read_json, utc_now, write_json
from ..rendering import GifExporter, validate_camera_path
from ..runtime import inspect_gpu, require_version
from ..synthetic import PosedImageDataset
from ..types import ModelArtifact, RenderArtifact


@dataclass(frozen=True)
class GaussianConfig:
    steps: int = 3000
    seed: int = 42
    initial_points: int = 20000
    initial_extent: float = 0.65
    initial_scale: float = 0.015
    initial_opacity: float = 0.1
    sh_degree: int = 2
    sh_interval: int = 500
    ssim_weight: float = 0.2
    refine_start: int = 200
    refine_stop: int = 2500
    refine_every: int = 100
    reset_every: int = 1000
    log_every: int = 100

    def __post_init__(self):
        for key in (
            "steps",
            "initial_points",
            "sh_interval",
            "refine_every",
            "reset_every",
            "log_every",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValidationError(f"{key} must be a positive integer")
        if not 0 < self.initial_opacity < 1 or not 0 <= self.ssim_weight <= 1:
            raise ValidationError("Invalid opacity or SSIM weight")
        if self.sh_degree not in (0, 1, 2, 3) or min(self.initial_extent, self.initial_scale) <= 0:
            raise ValidationError("Invalid Gaussian extent/scale or SH degree")


def opengl_to_view(cameras: np.ndarray) -> np.ndarray:
    """OpenGL c2w (-Z forward, +Y up) -> OpenCV world-to-camera."""
    cv = cameras.copy()
    cv[..., :3, 1:3] *= -1
    return np.linalg.inv(cv).astype(np.float32)


class GaussianScene:
    """Trainable Gaussian parameters with a shared training/inference rasterizer."""

    def __init__(self, config: GaussianConfig, state: dict | None = None):
        import torch

        self.config = config
        n = config.initial_points
        if state is None:
            state = {
                "means": (torch.rand(n, 3) * 2 - 1) * config.initial_extent,
                "scales": torch.full((n, 3), math.log(config.initial_scale)),
                "quats": torch.randn(n, 4),
                "opacities": torch.full(
                    (n,), math.log(config.initial_opacity / (1 - config.initial_opacity))
                ),
                "sh0": (torch.rand(n, 1, 3) - 0.5) / 0.28209479177387814,
                "shN": torch.zeros(n, (config.sh_degree + 1) ** 2 - 1, 3),
            }
        self.params = torch.nn.ParameterDict(
            {key: torch.nn.Parameter(value.to("cuda")) for key, value in state.items()}
        )

    def render(self, view, intrinsics, width: int, height: int, degree: int):
        import torch
        from gsplat import rasterization

        p = self.params
        return rasterization(
            means=p["means"],
            quats=p["quats"],
            scales=p["scales"].exp(),
            opacities=p["opacities"].sigmoid(),
            colors=torch.cat((p["sh0"], p["shN"]), 1),
            viewmats=view,
            Ks=intrinsics,
            width=width,
            height=height,
            sh_degree=degree,
            packed=False,
            backgrounds=torch.ones((len(view), 3), device="cuda"),
            near_plane=0.01,
            far_plane=100.0,
            rasterize_mode="classic",
        )


class KnownCameraGaussianReconstructor(KnownPoseReconstructor):
    """Fit a scene from a validated PosedImageDataset; no source mesh dependency."""

    def __init__(self, config: GaussianConfig | dict = GaussianConfig()):
        self.config = GaussianConfig(**config) if isinstance(config, dict) else config

    def fit(self, dataset: PosedImageDataset, output: Path) -> ModelArtifact:
        import torch
        from gsplat.strategy import DefaultStrategy
        from pytorch_msssim import ssim

        require_version("gsplat", "1.4.0")
        gpu = inspect_gpu()
        if not torch.cuda.is_available():
            raise ValidationError("PyTorch cannot use CUDA")
        output.mkdir(parents=True, exist_ok=False)
        cfg = self.config
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.cuda.reset_peak_memory_stats()
        images, cameras = dataset.load_split("train")
        # A 36-view RGB cache is ~108 MiB: avoid per-step AFS reads and host copies.
        targets = torch.from_numpy(images).to("cuda", dtype=torch.float32) / 255
        views = torch.from_numpy(opengl_to_view(cameras)).to("cuda")
        intrinsics = torch.from_numpy(dataset.intrinsics).to("cuda")[None]
        scene = GaussianScene(cfg)
        learning_rates = {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 1.25e-4,
        }
        optimizers = {
            key: torch.optim.Adam([value], lr=learning_rates[key], eps=1e-15)
            for key, value in scene.params.items()
        }
        strategy = DefaultStrategy(
            refine_start_iter=cfg.refine_start,
            refine_stop_iter=cfg.refine_stop,
            refine_every=cfg.refine_every,
            reset_every=cfg.reset_every,
        )
        strategy.check_sanity(scene.params, optimizers)
        state = strategy.initialize_state(scene_scale=1.0)
        config_path = output / "config.json"
        write_json(
            config_path,
            {
                "recipe": asdict(cfg),
                "dataset": str(dataset.root),
                "input_sha256": file_hash(dataset.root / "inputs.json"),
                "learning_rates": learning_rates,
                "gsplat": "1.4.0",
                "torch": torch.__version__,
                "gpu": gpu,
                "batch_size": 1,
                "background": "white",
                "initialization": "random; no pretrained checkpoint or source geometry",
            },
        )
        start = time.monotonic()
        rng = np.random.default_rng(cfg.seed)
        order = np.arange(len(images))
        with (output / "progress.jsonl").open("x") as log:
            for step in range(cfg.steps):
                if step % len(order) == 0:
                    rng.shuffle(order)
                index = int(order[step % len(order)])
                degree = min(step // cfg.sh_interval, cfg.sh_degree)
                prediction, _, info = scene.render(
                    views[index : index + 1], intrinsics, dataset.width, dataset.height, degree
                )
                strategy.step_pre_backward(scene.params, optimizers, state, step, info)
                target = targets[index : index + 1]
                l1 = (prediction - target).abs().mean()
                similarity = ssim(
                    prediction.permute(0, 3, 1, 2),
                    target.permute(0, 3, 1, 2),
                    data_range=1,
                    size_average=True,
                )
                loss = (1 - cfg.ssim_weight) * l1 + cfg.ssim_weight * (1 - similarity)
                loss.backward()
                strategy.step_post_backward(
                    scene.params, optimizers, state, step, info, packed=False
                )
                optimizers["means"].param_groups[0]["lr"] = learning_rates["means"] * 0.1 ** (
                    step / cfg.steps
                )
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if (step + 1) % cfg.log_every == 0 or step == 0:
                    scalar = float(loss.detach())
                    if not math.isfinite(scalar) or not all(
                        bool(torch.isfinite(p).all()) for p in scene.params.values()
                    ):
                        raise ValidationError(f"Non-finite loss/parameters at step {step + 1}")
                    record = {
                        "checked_at": utc_now(),
                        "optimizer_step": step + 1,
                        "loss": scalar,
                        "gaussians": len(scene.params["means"]),
                        "elapsed_seconds": time.monotonic() - start,
                        "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
        checkpoint = output / f"step_{cfg.steps:06d}.pt"
        temporary = checkpoint.with_suffix(".tmp")
        torch.save(
            {
                "optimizer_step": cfg.steps,
                "recipe": asdict(cfg),
                "state": {k: v.detach().cpu() for k, v in scene.params.items()},
            },
            temporary,
        )
        temporary.replace(checkpoint)
        artifact = ModelArtifact(
            "gsplat_known",
            config_path,
            checkpoint,
            file_hash(checkpoint),
            cfg.steps,
            {
                "config_sha256": file_hash(config_path),
                "input_sha256": file_hash(dataset.root / "inputs.json"),
                "initialization": "random",
                "geometry_scope": "known_cameras_no_mesh",
                "training_seconds": time.monotonic() - start,
            },
        )
        write_json(output / "model.json", artifact)
        return artifact


class GaussianRenderer(Renderer):
    def __init__(self):
        self._key = None
        self._scene = None

    def load(self, model: ModelArtifact) -> GaussianScene:
        import torch

        require_version("gsplat", "1.4.0")
        if (
            file_hash(model.checkpoint) != model.checkpoint_sha256
            or file_hash(model.config) != model.metadata["config_sha256"]
        ):
            raise ValidationError("Checkpoint/config changed")
        if self._key != model.checkpoint_sha256:
            state = torch.load(model.checkpoint, map_location="cpu", weights_only=True)
            if state["optimizer_step"] != model.step:
                raise ValidationError("Checkpoint step mismatch")
            self._scene = GaussianScene(GaussianConfig(**state["recipe"]), state["state"])
            self._key = model.checkpoint_sha256
        return self._scene

    def predict(
        self,
        model: ModelArtifact,
        cameras: np.ndarray,
        intrinsics: np.ndarray,
        width: int,
        height: int,
    ):
        import torch

        scene = self.load(model)
        views = torch.from_numpy(opengl_to_view(cameras)).to("cuda")
        ks = torch.from_numpy(intrinsics).to("cuda")[None]
        with torch.inference_mode():
            for view in views:
                rgb, _, _ = scene.render(view[None], ks, width, height, scene.config.sh_degree)
                yield (rgb[0].clamp(0, 1).cpu().numpy() * 255).round().astype(np.uint8)

    def render(self, model: ModelArtifact, camera_path: Path, output: Path) -> RenderArtifact:
        data = read_json(camera_path)
        validate_camera_path(data)
        output.mkdir(parents=True, exist_ok=False)
        width, height = data["render_width"], data["render_height"]
        fovs = {frame["fov"] for frame in data["camera_path"]}
        if len(fovs) != 1:
            raise ValidationError("This adapter needs a constant-FOV trajectory")
        focal = height / (2 * math.tan(math.radians(next(iter(fovs))) / 2))
        ks = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float32)
        cameras = np.array(
            [f["camera_to_world"] for f in data["camera_path"]], dtype=np.float32
        ).reshape(-1, 4, 4)
        paths = []
        for i, pixels in enumerate(self.predict(model, cameras, ks, width, height)):
            path = output / f"frame_{i:05d}.png"
            Image.fromarray(pixels).save(path)
            paths.append(path)
        result = RenderArtifact(
            tuple(paths),
            len(paths) / data["seconds"],
            {
                "kind": "learned_3dgs_reconstruction",
                "checkpoint": str(model.checkpoint),
                "checkpoint_sha256": model.checkpoint_sha256,
                "camera_path_sha256": file_hash(camera_path),
            },
        )
        write_json(output / "render.json", result)
        return result

    def evaluate(self, model: ModelArtifact, dataset: PosedImageDataset, output: Path) -> dict:
        output.mkdir(parents=True, exist_ok=False)
        result = {}
        for split in ("val", "test"):
            targets, cameras = dataset.load_split(split)
            rows = []
            for i, prediction in enumerate(
                self.predict(model, cameras, dataset.intrinsics, dataset.width, dataset.height)
            ):
                Image.fromarray(prediction).save(output / f"{split}_{i:02d}_prediction.png")
                Image.fromarray(targets[i]).save(output / f"{split}_{i:02d}_input.png")
                error = ((prediction.astype(np.float64) - targets[i]) / 255) ** 2
                mse = float(error.mean())
                rows.append({"index": i, "psnr_db": -10 * math.log10(max(mse, 1e-12))})
            result[split] = {
                "views": rows,
                "mean_psnr_db": float(np.mean([r["psnr_db"] for r in rows])),
            }
        result.update(
            checkpoint_sha256=model.checkpoint_sha256,
            input_sha256=file_hash(dataset.root / "inputs.json"),
            metric="Full-frame RGB PSNR from saved 8-bit PNGs; white background can inflate PSNR; final fixed step, no test-set tuning",
        )
        write_json(output / "metrics.json", result)
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    args = parser.parse_args()
    cfg = GaussianConfig(**read_json(args.config))
    for name in args.objects:
        dataset = PosedImageDataset(args.inputs / name)
        root = args.output / name
        artifact = KnownCameraGaussianReconstructor(cfg).fit(dataset, root / "training")
        renderer = GaussianRenderer()
        renderer.evaluate(artifact, dataset, root / "evaluation")
        render = renderer.render(artifact, dataset.root / "orbit.json", root / "orbit")
        GifExporter().export(render, root / "reconstruction.gif")
        write_json(
            root / "COMPLETE.json",
            {
                "checkpoint_sha256": artifact.checkpoint_sha256,
                "optimizer_steps": artifact.step,
                "gif": str(root / "reconstruction.gif"),
                "checked_at": utc_now(),
            },
        )


if __name__ == "__main__":
    main()
