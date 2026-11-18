"""Posthoc scientific plots of estimated camera layouts against reference poses."""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .core import Geometry, save_json, similarity_alignment


class CameraDiagnostics:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"])

    def build(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        output = self.root / "analysis"
        output.mkdir(exist_ok=True)
        rows = []
        for obj in self.config["objects"]:
            source = Path(self.config["source_inputs"]) / obj["uid"]
            frames = {
                Path(f["file_path"]).name: np.asarray(f["transform_matrix"], float)
                for f in json.loads((source / "transforms.json").read_text())["frames"]
            }
            methods = ["sift", "superpoint_lightglue", "loma", "gluemap"]
            fig = plt.figure(figsize=(12, 9))
            for i, method in enumerate(methods):
                ax = fig.add_subplot(2, 2, i + 1, projection="3d")
                folder = self.root / "results" / method / obj["uid"]
                if not (folder / "geometry.npz").exists():
                    ax.set_title(method + ": no reconstructed geometry")
                    ax.set_axis_off()
                    continue
                geometry = Geometry.load(folder)
                if len(geometry.names) < 3:
                    ax.set_title(f"{method}: only {len(geometry.names)} cameras; no Sim(3) fit")
                    ax.set_axis_off()
                    continue
                gt = np.stack([frames[name].copy() for name in geometry.names])
                gt[:, :3, 1:3] *= -1
                scale, rotation, translation = similarity_alignment(
                    geometry.c2w[:, :3, 3], gt[:, :3, 3]
                )
                centers = scale * geometry.c2w[:, :3, 3] @ rotation.T + translation
                estimated_r = np.einsum("ij,njk->nik", rotation, geometry.c2w[:, :3, :3])
                delta = np.einsum("nij,nkj->nik", estimated_r, gt[:, :3, :3])
                angle = float(
                    np.median(
                        np.rad2deg(
                            np.arccos(np.clip((np.trace(delta, axis1=1, axis2=2) - 1) / 2, -1, 1))
                        )
                    )
                )
                p = gt[:, :3, 3]
                ax.plot(
                    p[:, 0],
                    p[:, 1],
                    p[:, 2],
                    "--",
                    color="#777777",
                    label="Reference (posthoc only)",
                )
                ax.scatter(
                    centers[:, 0],
                    centers[:, 1],
                    centers[:, 2],
                    c="#2071b5",
                    s=16,
                    label="Estimated cameras",
                )
                z = estimated_r[:, :, 2]
                ax.quiver(
                    centers[:, 0],
                    centers[:, 1],
                    centers[:, 2],
                    z[:, 0],
                    z[:, 1],
                    z[:, 2],
                    length=0.18,
                    color="#2071b5",
                    alpha=0.65,
                )
                mid = np.mean(p, axis=0)
                radius = max(float(np.linalg.norm(p - mid, axis=1).max()), 0.1) * 1.15
                ax.set(
                    xlim=(mid[0] - radius, mid[0] + radius),
                    ylim=(mid[1] - radius, mid[1] + radius),
                    zlim=(mid[2] - radius, mid[2] + radius),
                    xlabel="X",
                    ylabel="Y",
                    zlabel="Z",
                )
                ax.set_box_aspect([1, 1, 1])
                ax.set_title(f"{method}: {len(p)}/36 cameras, median rotation {angle:.2f} deg")
                ax.legend(fontsize=7, loc="upper right")
                rows.append(
                    {
                        "uid": obj["uid"],
                        "method": method,
                        "registered": len(p),
                        "rotation_median_deg": angle,
                    }
                )
            fig.suptitle(
                f"{obj['name']} | train-center Sim(3) alignment; camera directions shown",
                fontsize=12,
            )
            fig.tight_layout()
            fig.savefig(output / (obj["uid"] + "_cameras.png"), dpi=150)
            plt.close(fig)
        save_json(
            output / "camera_diagnostics.json",
            {
                "rows": rows,
                "interpretation": "Posthoc reference alignment; plot is not input to reconstruction. Same axis limits per object. Axes describe normalized source-render coordinates.",
            },
        )


class ViewCountDiagnostics:
    """Summarize complete method grids; never treat pending runs as failures."""

    def __init__(self, config):
        self.config = config
        self.root = Path(config["output"])

    def build(self):
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt

        with (self.root / "view_ablation.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        expected = {obj["uid"] for obj in self.config["objects"]}
        methods = []
        for method in self.config["view_ablation_methods"]:
            group = [r for r in rows if r["method"] == method]
            keys = {(r["uid"], r["subset"]) for r in group}
            if len(keys) == len(expected) * 9 and all(
                r["state"] in {"COMPLETE", "FAILED"} for r in group
            ):
                methods.append(method)
        if not methods:
            raise ValueError("No complete view-count method grid yet")
        fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)
        records = []
        counts = [3, 5, 10, 20, 36]
        for si, sampling in enumerate(["uniform_orbit", "contiguous_arc"]):
            coverage = np.zeros((len(methods), len(counts)))
            rotations = np.full_like(coverage, np.nan)
            valid = np.zeros_like(coverage, dtype=int)
            for mi, method in enumerate(methods):
                for ni, count in enumerate(counts):
                    subset = f"{sampling if count < 36 else 'uniform_orbit'}_{count:02d}"
                    group = [r for r in rows if r["method"] == method and r["subset"] == subset]
                    assert {r["uid"] for r in group} == expected
                    assert len(group) == len(expected)
                    registered = [
                        int(r["registered"]) if r["state"] == "COMPLETE" else 0 for r in group
                    ]
                    angles = [
                        float(r["rotation_median_deg"])
                        for r in group
                        if r["state"] == "COMPLETE" and r["rotation_median_deg"]
                    ]
                    coverage[mi, ni] = np.mean(registered) / count
                    valid[mi, ni] = len(angles)
                    rotations[mi, ni] = np.median(angles) if angles else np.nan
                    records.append(
                        {
                            "method": method,
                            "sampling": sampling,
                            "count": count,
                            "mean_registered_fraction": float(coverage[mi, ni]),
                            "median_object_rotation_deg": float(rotations[mi, ni])
                            if angles
                            else None,
                            "pose_metric_objects": len(angles),
                            "total_objects": len(expected),
                        }
                    )
            for column, (matrix, cmap, vmax, label) in enumerate(
                [
                    (coverage, "YlGnBu", 1, "Mean registered fraction; failed geometry = 0"),
                    (
                        rotations,
                        "magma_r",
                        180,
                        "Median object rotation error (deg); available results only",
                    ),
                ]
            ):
                ax = axes[si, column]
                cm = plt.get_cmap(cmap).copy()
                cm.set_bad("#dddddd")
                graphic = ax.imshow(matrix, vmin=0, vmax=vmax, cmap=cm, aspect="auto")
                ax.set_xticks(range(len(counts)), [str(n) for n in counts])
                ax.set_yticks(range(len(methods)), methods)
                ax.set_xlabel("Reference images (36-view column shared)")
                ax.set_title(f"{sampling}\n{label}", fontsize=10)
                for mi in range(len(methods)):
                    for ni in range(len(counts)):
                        value = matrix[mi, ni]
                        if column == 0:
                            label = f"{value:.0%}"
                        else:
                            label = (
                                f"{value:.1f}\n(n={valid[mi, ni]})"
                                if np.isfinite(value)
                                else "--\n(n=0)"
                            )
                        rgb = cm(value / vmax)[:3] if np.isfinite(value) else (0.87, 0.87, 0.87)
                        brightness = sum(c * w for c, w in zip(rgb, [0.2126, 0.7152, 0.0722]))
                        color = "black" if brightness > 0.5 else "white"
                        ax.text(ni, mi, label, ha="center", va="center", fontsize=8, color=color)
                fig.colorbar(graphic, ax=ax, shrink=0.75)
        fig.suptitle(
            "View-count diagnostic snapshot | fixed 10 objects, independent SfM per subset\n"
            "Two elevation rings: contiguous 20 crosses rings. Rotation panels have different survivor sets.",
            fontsize=12,
        )
        output = self.root / "analysis"
        output.mkdir(exist_ok=True)
        fig.savefig(output / "view_count_diagnostics.png", dpi=150)
        plt.close(fig)
        save_json(
            output / "view_count_diagnostics.json",
            {
                "checked_at": time.time(),
                "methods": methods,
                "source_record_count": len(rows),
                "rows": records,
                "aggregation": "Equal object weighting. Failed geometry has zero usable registration. Pose errors summarize only available results; n disclosed. Pending methods excluded. 36-view column reused across sampling panels.",
            },
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--kind", choices=["cameras", "views"], default="cameras")
    args = parser.parse_args()
    diagnostic = CameraDiagnostics if args.kind == "cameras" else ViewCountDiagnostics
    diagnostic(json.loads(args.config.read_text())).build()


if __name__ == "__main__":
    main()
