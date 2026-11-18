"""GT-camera Gaussian comparison; explicitly separate from image-only SfM tests."""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from .core import Geometry, save_json, sha256, similarity_alignment
from .rendering import GaussianRenderer


class GroundTruthScene:
    def __init__(self, source: Path, estimated: Geometry):
        self.source = source
        self.data = json.loads((source / "transforms.json").read_text())
        records = json.loads((source / "inputs.json").read_text())
        if sha256(source / "transforms.json") != records["transforms_sha256"]:
            raise ValueError("Camera provenance changed")
        self.hashes = {row["path"]: row["sha256"] for row in records["images"]}
        self.frames = {
            split: [f for f in self.data["frames"] if f["split"] == split]
            for split in ["train", "val", "test"]
        }
        if not (3 <= len(self.frames["train"]) <= 36) or [
            len(self.frames[s]) for s in ["val", "test"]
        ] != [6, 6]:
            raise ValueError("Unexpected frozen split")
        train_by_name = {Path(f["file_path"]).name: f for f in self.frames["train"]}
        gt_registered = np.stack(
            [self.pose(train_by_name[n]["transform_matrix"]) for n in estimated.names]
        )
        scale, rotation, translation = similarity_alignment(
            estimated.c2w[:, :3, 3], gt_registered[:, :3, 3]
        )
        points = scale * estimated.points @ rotation.T + translation
        poses = np.stack([self.pose(f["transform_matrix"]) for f in self.frames["train"]])
        self.center = np.median(poses[:, :3, 3], axis=0)
        self.scale = float(np.median(np.linalg.norm(poses[:, :3, 3] - self.center, axis=1)))
        self.points = (points - self.center) / self.scale
        self.colors = estimated.colors.astype(np.float32) / 255
        valid = np.isfinite(self.points).all(1) & (np.linalg.norm(self.points, axis=1) < 10)
        self.points, self.colors = self.points[valid], self.colors[valid]
        if len(self.points) < 20:
            raise ValueError("Invalid common SfM initialization")
        if len(self.points) > 20000:
            keep = np.random.default_rng(42).choice(len(self.points), 20000, replace=False)
            self.points, self.colors = self.points[keep], self.colors[keep]
        self.k = np.array(
            [
                [self.data["fl_x"], 0, self.data["cx"]],
                [0, self.data["fl_y"], self.data["cy"]],
                [0, 0, 1],
            ],
            np.float32,
        )

    @staticmethod
    def pose(matrix):
        p = np.array(matrix, dtype=float).reshape(4, 4)
        p[:3, 1:3] *= -1
        return p

    def view(self, matrix):
        p = self.pose(matrix)
        p[:3, 3] = (p[:3, 3] - self.center) / self.scale
        return np.linalg.inv(p).astype(np.float32)

    def pixels(self, frame):
        path = self.source / frame["file_path"]
        if sha256(path) != self.hashes[frame["file_path"]]:
            raise ValueError("Changed source image")
        im = Image.open(path).convert("RGB")
        if im.size != (512, 512):
            raise ValueError("Unexpected resolution")
        return np.asarray(im).copy()


class GaussianComparison:
    def __init__(self, method, steps=30000, scale_schedule=False):
        self.method, self.steps = method, steps
        self.schedule_scale = steps / 30000 if scale_schedule else 1.0
        self.renderer = GaussianRenderer(method, self.schedule_scale)

    def initialize(self, scene):
        import torch
        from scipy.spatial import cKDTree

        rng = np.random.default_rng(42)
        n = len(scene.points)
        distance, _ = cKDTree(scene.points).query(scene.points, k=4)
        size = np.sqrt(np.mean(distance[:, 1:] ** 2, axis=1)).clip(1e-4, 0.1)
        arrays = {
            "means": scene.points.astype(np.float32),
            "scales": np.repeat(np.log(size)[:, None], 3, axis=1).astype(np.float32),
            "quats": rng.normal(size=(n, 4)).astype(np.float32),
            "opacities": np.full(n, math.log(0.1 / 0.9), np.float32),
            "sh0": ((scene.colors[:, None] - 0.5) / 0.28209479177387814).astype(np.float32),
            "shN": np.zeros((n, 15, 3), np.float32),
        }
        return torch.nn.ParameterDict(
            {
                key: torch.nn.Parameter(torch.from_numpy(value).cuda())
                for key, value in arrays.items()
            }
        )

    def evaluate(self, scene, params, split, folder=None):
        import torch
        from pytorch_msssim import ssim

        rows = []
        kt = torch.from_numpy(scene.k[None]).cuda()
        for frame in scene.frames[split]:
            with torch.inference_mode():
                view_fn = getattr(scene, "eval_view", scene.view)
                view = torch.from_numpy(view_fn(frame["transform_matrix"])[None]).cuda()
                pred = self.renderer.render(params, view, kt).rgb[0].clamp(0, 1)
                target = torch.from_numpy(scene.pixels(frame)).float().cuda() / 255
                err = (pred - target).square()
                foreground = (target < 0.96).any(-1)
                row = {
                    "name": Path(frame["file_path"]).name,
                    "psnr": float(-10 * torch.log10(err.mean().clamp_min(1e-12))),
                    "foreground_psnr": float(
                        -10 * torch.log10(err[foreground].mean().clamp_min(1e-12))
                    ),
                    "ssim": float(
                        ssim(
                            pred.permute(2, 0, 1)[None], target.permute(2, 0, 1)[None], data_range=1
                        )
                    ),
                }
                rows.append(row)
                if folder:
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray((pred.cpu().numpy() * 255).round().astype(np.uint8)).save(
                        folder / row["name"]
                    )
        return {
            "views": rows,
            **{
                key: float(np.mean([row[key] for row in rows]))
                for key in ["psnr", "foreground_psnr", "ssim"]
            },
        }

    def fit(self, scene, output):
        import torch
        from gsplat.strategy import MCMCStrategy
        from pytorch_msssim import ssim

        output.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        params = self.initialize(scene)
        rates = {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 1.25e-4,
        }
        optimizers = {
            key: torch.optim.Adam([value], lr=rates[key], eps=1e-15)
            for key, value in params.items()
        }
        strategy = self.renderer.strategy()
        strategy.check_sanity(params, optimizers)
        mcmc = isinstance(strategy, MCMCStrategy)
        state = strategy.initialize_state() if mcmc else strategy.initialize_state(scene_scale=1.0)
        frames = scene.frames["train"]
        views = torch.from_numpy(
            np.stack([scene.view(f["transform_matrix"]) for f in frames])
        ).cuda()
        targets = torch.from_numpy(np.stack([scene.pixels(f) for f in frames])).float().cuda() / 255
        train_k = getattr(scene, "train_intrinsics", scene.k[None])
        kt = torch.from_numpy(np.asarray(train_k, dtype=np.float32)).cuda()
        best = -math.inf
        best_step = None
        validations = []
        order = np.arange(len(frames))
        started = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        with (output / "loss.jsonl").open("w") as log:
            for step in range(self.steps):
                if step % len(order) == 0:
                    rng.shuffle(order)
                i = int(order[step % len(order)])
                render = self.renderer.render(
                    params,
                    views[i : i + 1],
                    kt[i : i + 1] if len(kt) > 1 else kt,
                    min(step // max(1, int(1000 * self.schedule_scale)), 3),
                )
                strategy.step_pre_backward(params, optimizers, state, step, render.info)
                loss = 0.8 * (render.rgb - targets[i : i + 1]).abs().mean() + 0.2 * (
                    1
                    - ssim(
                        render.rgb.permute(0, 3, 1, 2),
                        targets[i : i + 1].permute(0, 3, 1, 2),
                        data_range=1,
                    )
                )
                loss = loss + self.renderer.regularization(params, render, step)
                loss.backward()
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                lr = rates["means"] * 0.01 ** (step / max(1, self.steps - 1))
                optimizers["means"].param_groups[0]["lr"] = lr
                if mcmc:
                    strategy.step_post_backward(params, optimizers, state, step, render.info, lr=lr)
                else:
                    strategy.step_post_backward(
                        params, optimizers, state, step, render.info, packed=False
                    )
                if step % 100 == 0 or step + 1 == self.steps:
                    value = float(loss.detach())
                    if not math.isfinite(value):
                        raise RuntimeError("Non-finite loss")
                    row = {
                        "step": step + 1,
                        "loss": value,
                        "points": len(params["means"]),
                        "seconds": time.monotonic() - started,
                    }
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    save_json(output / "progress.json", row)
                if step + 1 in {
                    max(1, int(7000 * self.schedule_scale)),
                    max(1, int(15000 * self.schedule_scale)),
                    self.steps,
                }:
                    result = self.evaluate(scene, params, "val")
                    validations.append({"step": step + 1, **result})
                    score = result["foreground_psnr"]
                    if score > best:
                        best, best_step = score, step + 1
                        torch.save(
                            {
                                "state": {k: v.detach().cpu() for k, v in params.items()},
                                "step": best_step,
                                "method": self.method,
                                "center": scene.center,
                                "scale": scene.scale,
                            },
                            output / "best.pt",
                        )
                    save_json(
                        output / "validation.json",
                        {
                            "checkpoints": validations,
                            "selection": "maximum mean validation foreground PSNR",
                            "best_step": best_step,
                        },
                    )
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in params.items()},
                "step": self.steps,
                "method": self.method,
                "center": scene.center,
                "scale": scene.scale,
            },
            output / "final.pt",
        )
        save_json(
            output / "training.json",
            {
                "method": self.method,
                "steps": self.steps,
                "seconds": time.monotonic() - started,
                "best_step": best_step,
                "points": len(params["means"]),
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "camera_source": getattr(
                    scene, "camera_source", "GT intrinsics/extrinsics; fixed, no pose optimization"
                ),
                "initialization": getattr(
                    scene,
                    "initialization_description",
                    "same LoMa-SfM point cloud for all methods, aligned using registered train cameras only; no GT mesh",
                ),
                "schedule_scale": self.schedule_scale,
                "training_views": len(frames),
                "best_checkpoint_sha256": sha256(output / "best.pt"),
                "final_checkpoint_sha256": sha256(output / "final.pt"),
            },
        )
        best_state = torch.load(output / "best.pt", map_location="cuda", weights_only=False)[
            "state"
        ]
        del params, optimizers
        for split in ["val", "test"]:
            save_json(
                output / f"{split}_metrics.json",
                self.evaluate(scene, best_state, split, output / split),
            )
        self.orbit(scene, best_state, output)

    def orbit(self, scene, params, output):
        import torch

        orbit = json.loads((scene.source / "orbit.json").read_text())
        images = []
        folder = output / "orbit"
        folder.mkdir(exist_ok=True)
        kt = torch.from_numpy(scene.k[None]).cuda()
        for index, frame in enumerate(orbit["camera_path"]):
            with torch.inference_mode():
                view_fn = getattr(scene, "eval_view", scene.view)
                view = torch.from_numpy(view_fn(frame["camera_to_world"])[None]).cuda()
                rgb = self.renderer.render(params, view, kt).rgb[0].clamp(0, 1).cpu().numpy()
            im = Image.fromarray((rgb * 255).round().astype(np.uint8))
            im.save(folder / f"frame_{index:05d}.png")
            images.append(im)
        durations = [int(round((i + 1) * 100 / 24) - round(i * 100 / 24)) * 10 for i in range(72)]
        images[0].save(
            output / "reconstruction.gif",
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0,
        )
        save_json(
            output / "render.json",
            {"frames": 72, "fps": 24, "seconds": 3, "size": [512, 512], "checkpoint": "best.pt"},
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = Path(config["output"])
    import torch

    torch.set_num_threads(4)
    # A shared compiler/dependency failure is a setup failure, not forty model failures.
    from gsplat.cuda._backend import _C

    if _C is None:
        raise RuntimeError("gsplat CUDA extension did not load")
    for obj in config["objects"]:
        seed_path = Path(config["matching_output"]) / "results/loma" / obj["uid"]
        seed = Geometry.load(seed_path)
        scene = GroundTruthScene(Path(config["source_inputs"]) / obj["uid"], seed)
        save_json(
            root / "initialization" / (obj["uid"] + ".json"),
            {
                "source_geometry": str(seed_path / "geometry.npz"),
                "source_geometry_sha256": sha256(seed_path / "geometry.npz"),
                "camera_json_sha256": sha256(scene.source / "transforms.json"),
                "points": len(scene.points),
                "camera_source": "GT",
                "seed": 42,
                "normalization_center": scene.center.tolist(),
                "normalization_scale": scene.scale,
            },
        )
        for method in config["render_methods"]:
            output = root / "results" / method / obj["uid"]
            output.mkdir(parents=True, exist_ok=True)
            if (output / "status.json").exists() and json.loads(
                (output / "status.json").read_text()
            )["state"] == "COMPLETE":
                continue
            save_json(output / "status.json", {"state": "RUNNING", "checked_at": time.time()})
            try:
                GaussianComparison(method, config["steps"]).fit(scene, output)
                save_json(output / "status.json", {"state": "COMPLETE", "checked_at": time.time()})
            except Exception as e:
                (output / "error.log").write_text(traceback.format_exc())
                save_json(
                    output / "status.json",
                    {"state": "FAILED", "error": str(e), "checked_at": time.time()},
                )
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
