"""Common Gaussian optimizer consuming estimated geometry, with no GT training poses."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
from PIL import Image

from reconlab.benchmark.core import Geometry, save_json, sha256, similarity_alignment


class GaussianBenchmark:
    def __init__(self, steps=15000, on_checkpoint=None):
        self.steps = steps
        self.on_checkpoint = on_checkpoint

    def fit(self, geometry: Geometry, images: Path, output: Path):
        import torch
        from gsplat import rasterization
        from gsplat.strategy import DefaultStrategy
        from pytorch_msssim import ssim
        from scipy.spatial import cKDTree

        if len(geometry.names) < 3 or len(geometry.points) < 20:
            raise ValueError("Insufficient reconstructed geometry for Gaussian training")
        output.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        # Similarity normalization uses estimated cameras only.
        center = np.median(geometry.c2w[:, :3, 3], axis=0)
        scene_scale = float(np.median(np.linalg.norm(geometry.c2w[:, :3, 3] - center, axis=1)))
        if not np.isfinite(scene_scale) or scene_scale <= 1e-8:
            raise ValueError("Degenerate camera scale")
        points = (geometry.points - center) / scene_scale
        colors = geometry.colors.astype(np.float32) / 255
        finite = np.isfinite(points).all(1) & (np.linalg.norm(points, axis=1) < 10)
        points, colors = points[finite], colors[finite]
        if len(points) > 20000:
            choose = rng.choice(len(points), 20000, replace=False)
            points, colors = points[choose], colors[choose]
        if len(points) < 20:
            raise ValueError("No usable initialization points")
        distance, _ = cKDTree(points).query(points, k=4)
        scales = np.sqrt(np.mean(distance[:, 1:] ** 2, axis=1)).clip(1e-4, 0.1)
        n = len(points)
        tensors = {
            "means": points.astype(np.float32),
            "scales": np.repeat(np.log(scales)[:, None], 3, axis=1).astype(np.float32),
            "quats": rng.normal(size=(n, 4)).astype(np.float32),
            "opacities": np.full(n, math.log(0.1 / 0.9), np.float32),
            "sh0": ((colors[:, None] - 0.5) / 0.28209479177387814).astype(np.float32),
            "shN": np.zeros((n, 8, 3), np.float32),
        }
        params = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(torch.from_numpy(v).cuda()) for k, v in tensors.items()}
        )
        rates = {
            "means": 1.6e-4,
            "scales": 5e-3,
            "quats": 1e-3,
            "opacities": 5e-2,
            "sh0": 2.5e-3,
            "shN": 1.25e-4,
        }
        optimizers = {k: torch.optim.Adam([v], lr=rates[k], eps=1e-15) for k, v in params.items()}
        c2w = geometry.c2w.copy()
        c2w[:, :3, 3] = (c2w[:, :3, 3] - center) / scene_scale
        views = torch.from_numpy(np.linalg.inv(c2w).astype(np.float32)).cuda()
        ks = torch.from_numpy(geometry.intrinsics.astype(np.float32)).cuda()
        targets = (
            torch.from_numpy(
                np.stack(
                    [
                        np.asarray(Image.open(images / name).convert("RGB"))
                        for name in geometry.names
                    ]
                )
            )
            .float()
            .cuda()
            / 255
        )
        strategy = DefaultStrategy(
            refine_start_iter=200, refine_stop_iter=2500, refine_every=100, reset_every=3000
        )
        strategy.check_sanity(params, optimizers)
        state = strategy.initialize_state(scene_scale=1.0)
        bg = torch.ones((1, 3), device="cuda")
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        order = np.arange(len(views))
        with (output / "loss.jsonl").open("w") as log:
            for step in range(self.steps):
                if step % len(order) == 0:
                    rng.shuffle(order)
                i = int(order[step % len(order)])
                prediction, _, info = rasterization(
                    means=params["means"],
                    quats=params["quats"],
                    scales=params["scales"].exp(),
                    opacities=params["opacities"].sigmoid(),
                    colors=torch.cat((params["sh0"], params["shN"]), 1),
                    viewmats=views[i : i + 1],
                    Ks=ks[i : i + 1],
                    width=512,
                    height=512,
                    sh_degree=min(step // 500, 2),
                    backgrounds=bg,
                    packed=False,
                    near_plane=0.01,
                    far_plane=100,
                )
                strategy.step_pre_backward(params, optimizers, state, step, info)
                target = targets[i : i + 1]
                loss = 0.8 * (prediction - target).abs().mean() + 0.2 * (
                    1
                    - ssim(prediction.permute(0, 3, 1, 2), target.permute(0, 3, 1, 2), data_range=1)
                )
                loss.backward()
                for optimizer in optimizers.values():
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                # Official gsplat example updates strategy after optimizer.step.
                strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
                optimizers["means"].param_groups[0]["lr"] = rates["means"] * 0.01 ** (
                    step / max(1, self.steps - 1)
                )
                if step % 100 == 0 or step == self.steps - 1:
                    value = float(loss.detach())
                    if not math.isfinite(value):
                        raise RuntimeError("Non-finite Gaussian loss")
                    record = {
                        "step": step + 1,
                        "loss": value,
                        "points": len(params["means"]),
                        "seconds": time.monotonic() - start,
                    }
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if self.on_checkpoint is not None and (step + 1) % 3000 == 0:
                    self.on_checkpoint(step + 1, params, optimizers, state, rng, order,
                                       center, scene_scale, n, time.monotonic() - start)
        checkpoint = output / "gaussians.pt"
        torch.save(
            {
                "state": {k: v.detach().cpu() for k, v in params.items()},
                "center": center,
                "scene_scale": scene_scale,
                "steps": self.steps,
            },
            checkpoint,
        )
        save_json(
            output / "training.json",
            {
                "steps": self.steps,
                "initial_points": n,
                "final_points": len(params["means"]),
                "seconds": time.monotonic() - start,
                "checkpoint_sha256": sha256(checkpoint),
                "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
                "initialization": "estimated point cloud, at most 20000 points, KNN scales",
                "training_camera_source": "estimated; no GT",
                "registered_views": len(views),
                "tf32": False,
                "opacity_reset_every": 3000,
                "seed": 42,
            },
        )
        return params, center, scene_scale

    def evaluate(self, geometry: Geometry, source: Path, output: Path, params, center, scene_scale):
        """Only this evaluation stage opens synthetic camera metadata."""
        import torch
        from gsplat import rasterization
        from pytorch_msssim import ssim

        data = json.loads((source / "transforms.json").read_text())
        frames = {Path(f["file_path"]).name: f for f in data["frames"]}
        gt_train = np.stack([np.asarray(frames[n]["transform_matrix"]) for n in geometry.names])
        gt_train[:, :3, 1:3] *= -1
        scale, rotation, translation = similarity_alignment(
            geometry.c2w[:, :3, 3], gt_train[:, :3, 3]
        )
        aligned_centers = scale * geometry.c2w[:, :3, 3] @ rotation.T + translation
        rotated = np.einsum("ij,njk->nik", rotation, geometry.c2w[:, :3, :3])
        delta = np.einsum("nij,nkj->nik", rotated, gt_train[:, :3, :3])
        angles = np.rad2deg(np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        save_json(
            output / "pose_metrics.json",
            {
                "alignment": "Sim3 fitted on train centers only",
                "rotation_median_deg": float(np.median(angles)),
                "rotation_mean_deg": float(angles.mean()),
                "center_rmse": float(np.sqrt(np.mean((aligned_centers - gt_train[:, :3, 3]) ** 2))),
                "scale": scale,
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            },
        )
        k = np.array(
            [[data["fl_x"], 0, data["cx"]], [0, data["fl_y"], data["cy"]], [0, 0, 1]], np.float32
        )
        kt = torch.from_numpy(k[None]).cuda()

        def render(gl_pose):
            gt = np.array(gl_pose, dtype=np.float64).reshape(4, 4)
            gt[:3, 1:3] *= -1
            estimated = np.eye(4)
            estimated[:3, :3] = rotation.T @ gt[:3, :3]
            estimated[:3, 3] = (
                (rotation.T @ (gt[:3, 3] - translation) / scale) - center
            ) / scene_scale
            view = torch.from_numpy(np.linalg.inv(estimated).astype(np.float32)[None]).cuda()
            with torch.inference_mode():
                image, _, _ = rasterization(
                    means=params["means"],
                    quats=params["quats"],
                    scales=params["scales"].exp(),
                    opacities=params["opacities"].sigmoid(),
                    colors=torch.cat((params["sh0"], params["shN"]), 1),
                    viewmats=view,
                    Ks=kt,
                    width=512,
                    height=512,
                    sh_degree=2,
                    backgrounds=torch.ones((1, 3), device="cuda"),
                    packed=False,
                    near_plane=0.01,
                    far_plane=100,
                )
            return image[0].clamp(0, 1)

        metrics = []
        for f in data["frames"]:
            if f["split"] == "train":
                continue
            pred = render(f["transform_matrix"])
            gt = (
                torch.from_numpy(
                    np.asarray(Image.open(source / f["file_path"]).convert("RGB")).copy()
                )
                .float()
                .cuda()
                / 255
            )
            error = (pred - gt).square()
            fg = (gt < 0.96).any(-1)
            row = {
                "name": Path(f["file_path"]).name,
                "split": f["split"],
                "psnr": float(-10 * torch.log10(error.mean().clamp_min(1e-12))),
                "foreground_psnr": float(-10 * torch.log10(error[fg].mean().clamp_min(1e-12)))
                if fg.any()
                else None,
                "ssim": float(
                    ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], data_range=1)
                ),
            }
            metrics.append(row)
            path = output / "heldout" / row["name"]
            path.parent.mkdir(exist_ok=True)
            Image.fromarray((pred.cpu().numpy() * 255).round().astype(np.uint8)).save(path)
        save_json(
            output / "render_metrics.json",
            {"views": metrics, "mask_source": "reference RGB < .96 in any channel"},
        )
        orbit = json.loads((source / "orbit.json").read_text())
        rendered = []
        (output / "orbit").mkdir(exist_ok=True)
        for index, frame in enumerate(orbit["camera_path"]):
            rgb = (render(frame["camera_to_world"]).cpu().numpy() * 255).round().astype(np.uint8)
            im = Image.fromarray(rgb)
            im.save(output / "orbit" / f"frame_{index:05d}.png")
            rendered.append(im)
        # GIF stores centiseconds. 40/40/50 ms repeats sum to exactly 3 seconds.
        durations = [int(round((i + 1) * 100 / 24) - round(i * 100 / 24)) * 10 for i in range(72)]
        rendered[0].save(
            output / "reconstruction.gif",
            save_all=True,
            append_images=rendered[1:],
            duration=durations,
            loop=0,
            disposal=2,
        )
        save_json(
            output / "render.json",
            {
                "frames": 72,
                "fps": 24,
                "seconds": 3,
                "size": [512, 512],
                "camera_path": "GT orbit transformed only by train-camera Sim3 for evaluation",
                "gif_timing": "centisecond quantization; original 72 PNG frames retained",
            },
        )
