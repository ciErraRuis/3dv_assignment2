"""Adapter and export hooks for the official uncalibrated MASt3R-SLAM solver."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .core import Geometry, PairMatches, ReconstructionMethod, draw_matches, save_json, verify


@dataclass
class HeadlessWindowMessage:
    is_terminated: bool = False
    is_paused: bool = False
    next: bool = False
    C_conf_threshold: float = 1.5


class Mast3rSlam(ReconstructionMethod):
    def __init__(self, config):
        self.config = config

    def reconstruct(self, images: Path, output: Path) -> Geometry:
        source = Path(self.config["sources"]) / "mast3r_slam"
        weights = Path(self.config["output"]) / "weights"
        env = os.environ.copy()
        env.update(
            RECONLAB_SLAM_OUTPUT=str(output),
            RECONLAB_SLAM_IMAGES=str(images),
            RECONLAB_SLAM_MODEL=str(weights / "mast3r"),
            RECONLAB_SLAM_RETRIEVER=str(
                weights
                / "slam_retrieval"
                / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"
            ),
        )
        env["PYTHONPATH"] = ":".join(
            [
                str(source),
                str(source / "thirdparty/mast3r"),
                str(source / "thirdparty/mast3r/dust3r"),
                env.get("PYTHONPATH", ""),
            ]
        )
        command = [
            self.config["slam_python"],
            str(source / "main.py"),
            "--dataset",
            str(images),
            "--config",
            str(source / "config/base.yaml"),
            "--save-as",
            str(output / "native"),
            "--no-viz",
        ]
        with (output / "native.log").open("w") as log:
            process = subprocess.Popen(
                command,
                env=env,
                cwd=source,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                returncode = process.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        return Geometry.load(output)


def export_edges(graph):
    """Backend hook: serialize the accepted native correspondences, without GT."""
    output = Path(os.environ["RECONLAB_SLAM_OUTPUT"])
    paths = sorted(Path(os.environ["RECONLAB_SLAM_IMAGES"]).glob("*.png"))
    rows = []
    folder = output / "matches"
    folder.mkdir(exist_ok=True)
    for edge, (i, j) in enumerate(zip(graph.ii.tolist(), graph.jj.tolist())):
        first, second = graph.frames[i], graph.frames[j]
        a, b = paths[int(first.frame_id)], paths[int(second.frame_id)]
        h, w = map(int, second.img_shape.flatten().tolist())
        indices = graph.idx_ii2jj[edge].detach().cpu().numpy().reshape(-1)
        confidence = graph.Q_ii2jj[edge].detach().cpu().numpy().reshape(-1)
        valid = graph.valid_match_j[edge].detach().cpu().numpy().reshape(-1)
        valid &= confidence > graph.cfg["Q_conf"]
        keep = np.flatnonzero(valid)
        matches = PairMatches(
            np.column_stack([indices[keep] % w, indices[keep] // w]),
            np.column_stack([keep % w, keep // w]),
            confidence[keep],
        )
        mask = verify(matches)
        name = f"{edge:05d}_{a.stem}_{b.stem}.npz"
        np.savez_compressed(
            folder / name,
            first=matches.first,
            second=matches.second,
            scores=matches.scores,
            inliers=mask,
        )
        if edge < 6:
            draw_matches(a, b, matches, mask, output / "visuals" / f"pair_{edge:03d}.png")
        rows.append(
            {
                "first": a.name,
                "second": b.name,
                "matches": len(keep),
                "inliers": int(mask.sum()),
                "file": name,
            }
        )
    save_json(
        output / "matching.json",
        {
            "method": "mast3r_slam",
            "pairs": rows,
            "graph": "accepted native keyframe tracking/retrieval graph",
            "verification": "F-MAGSAC diagnostic only; native optimizer used its own confidences",
            "camera_metadata_supplied": False,
        },
    )


def export_keyframes(dataset, keyframes):
    """Main-process hook after backend join; export its final geometry and inputs."""
    import torch
    from dust3r.post_process import estimate_focal_knowing_depth
    from mast3r_slam.lietorch_utils import as_SE3

    output = Path(os.environ["RECONLAB_SLAM_OUTPUT"])
    names, poses, intrinsics, clouds, colors, evidence = [], [], [], [], [], []
    for i in range(len(keyframes)):
        frame = keyframes[i]
        name = Path(dataset.rgb_files[int(frame.frame_id)]).name
        h, w = map(int, frame.img_shape.flatten().tolist())
        if (h, w) != (512, 512):
            raise ValueError("SLAM input violated full-frame 512 policy")
        xyz = frame.X_canon.reshape(1, h, w, 3)
        pp = torch.tensor([w / 2, h / 2], device=xyz.device)
        focal = float(estimate_focal_knowing_depth(xyz, pp, focal_mode="weiszfeld").item())
        if not np.isfinite(focal) or focal <= 0:
            raise ValueError("Native point-map focal estimate is invalid")
        intrinsics.append([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]])
        poses.append(as_SE3(frame.T_WC).matrix().detach().cpu().numpy().reshape(4, 4))
        names.append(name)
        rgb = (frame.uimg.cpu().numpy() * 255).round().astype(np.uint8)
        points = frame.T_WC.act(frame.X_canon).detach().cpu().numpy().reshape(-1, 3)
        confidence = frame.get_average_conf().detach().cpu().numpy().reshape(-1)
        valid = (confidence > 1.5) & np.isfinite(points).all(1) & (rgb.reshape(-1, 3) < 245).any(1)
        clouds.append(points[valid])
        colors.append(rgb.reshape(-1, 3)[valid])
        folder = output / "model_inputs/normalized_rgb_512"
        folder.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(folder / name)
        tensor = frame.img.detach().cpu().contiguous().numpy()
        evidence.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "normalized_tensor_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
                "display": "actual normalized input inverted to RGB8",
            }
        )
    if len(names) < 3:
        raise RuntimeError(
            f"Only {len(names)} native keyframes; insufficient for common evaluation"
        )
    points, rgb = np.concatenate(clouds), np.concatenate(colors)
    n = len(points)
    if n > 100000:
        keep = np.random.default_rng(42).choice(n, 100000, replace=False)
        points, rgb = points[keep], rgb[keep]
    Geometry(
        names,
        np.asarray(poses),
        np.asarray(intrinsics),
        points,
        rgb,
        {
            "method": "MASt3R-SLAM uncalibrated",
            "registered": len(names),
            "input_views": len(dataset),
            "registration_semantics": "native keyframes only, not all tracked input frames",
            "intrinsics": "estimated from native point maps using DUSt3R Weiszfeld estimator; no GT K",
            "initialization_points": len(points),
            "valid_foreground_points_before_cap": n,
            "native_full_cloud": "native/images.ply",
            "point_filter": "confidence>1.5, RGB<245 in any channel; seeded cap100000",
            "source_order": "ascending train view index; static images, no physical video fps",
        },
    ).save(output)
    save_json(
        output / "model_inputs/manifest.json",
        {"frames": evidence, "crop": False, "size": [512, 512]},
    )
