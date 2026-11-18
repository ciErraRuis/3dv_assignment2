"""Centered upright trajectory used by the selected RoMa + gsplat pipeline."""

import numpy as np


class CenteredOrbit:
    def __init__(self, model):
        color = model.sh[:, 0] * 0.28209479177387814 + 0.5
        mask = (
            (model.opacity[:, 0] > 0.1)
            & (color.min(1) < 0.9)
            & np.isfinite(model.xyz).all(1)
        )
        p = model.xyz[mask]
        if len(p) < 20:
            raise ValueError(
                "Insufficient foreground Gaussians for centered presentation"
            )
        lo, hi = np.quantile(p, [0.02, 0.98], axis=0)
        core = p[np.all((p >= lo) & (p <= hi), 1)]
        eig, axes = np.linalg.eigh(np.cov(core.T))
        avg_up = -model.c2w[:, :3, 1].mean(0)
        consistency = float(np.linalg.norm(avg_up))
        if consistency < 0.1:
            avg_up = -model.c2w[0, :3, 1].copy()
        avg_up /= np.linalg.norm(avg_up)
        alignment = abs(float(np.dot(axes[:, -1], avg_up)))
        use_major = eig[-1] / max(eig[-2], 1e-12) >= 3 and alignment >= 0.85
        self.up = (
            axes[:, -1] * np.sign(np.dot(axes[:, -1], avg_up)) if use_major else avg_up
        )
        up_policy = (
            "camera-consistent elongated PCA major axis"
            if use_major
            else "mean optimized camera up (first camera fallback if degenerate)"
        )
        basis = axes
        coords = core @ basis
        self.center = ((coords.min(0) + coords.max(0)) * 0.5) @ basis.T
        front = model.c2w[0, :3, 3] - self.center
        front -= np.dot(front, self.up) * self.up
        self.front = front / np.linalg.norm(front)
        self.side = np.cross(self.up, self.front)
        self.side /= np.linalg.norm(self.side)
        # Fit all selected foreground centers plus 3-sigma splat extent.
        bound = np.max(
            np.linalg.norm(p - self.center, axis=1) + 3 * model.scales[mask].max(1)
        )
        self.radius = float(1.12 * bound / np.sin(np.arctan(256 / model.k[0, 0])))
        self.poses = []
        for t in np.arange(72) * 2 * np.pi / 72:
            elev = np.deg2rad(12)
            eye = self.center + self.radius * (
                np.cos(elev) * (np.cos(t) * self.front + np.sin(t) * self.side)
                + np.sin(elev) * self.up
            )
            forward = self.center - eye
            forward /= np.linalg.norm(forward)
            right = np.cross(forward, self.up)
            right /= np.linalg.norm(right)
            down = np.cross(forward, right)
            pose = np.eye(4)
            pose[:3, :3] = np.stack([right, down, forward], 1)
            pose[:3, 3] = eye
            assert np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
            assert np.linalg.det(pose[:3, :3]) > 0.999
            target = np.linalg.inv(pose) @ np.r_[self.center, 1]
            assert np.linalg.norm(target[:2]) < 1e-6 and target[2] > 0
            assert np.dot(-pose[:3, 1], self.up) > 0
            self.poses.append(pose)
        self.metadata = {
            "purpose": "presentation_only",
            "target": self.center.tolist(),
            "up": self.up.tolist(),
            "up_policy": up_policy,
            "camera_up_consistency": consistency,
            "major_axis_camera_alignment": alignment,
            "radius": self.radius,
            "elevation_deg": 12,
            "frames": 72,
            "fps": 24,
            "seconds": 3,
            "size": [512, 512],
            "geometry_modified": False,
            "gt_poses_used": False,
            "camera_to_world_opencv": [x.tolist() for x in self.poses],
            "framing": "All finite opacity>0.1 and nonwhite DC-color centers plus 3-sigma extent; all Gaussians still rendered",
        }
