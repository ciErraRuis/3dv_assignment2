"""Read-only aggregation of live results; never infer completion from a directory."""

from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .core import save_json


def read(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"state": "READ_IN_PROGRESS"}


def mean(values):
    finite = [v for v in values if isinstance(v, (int, float))]
    return statistics.mean(finite) if finite else None


class ExperimentReport:
    def __init__(self, config, gt_root):
        self.config, self.root, self.gt_root = config, Path(config["output"]), gt_root

    def collect(self):
        rows, gt_rows, view_rows = [], [], []
        gt_execution = (
            read(Path(self.config["gt_execution_status"]))
            if self.config.get("gt_execution_status")
            else {}
        )
        for obj in self.config["objects"]:
            for method in self.config["methods"]:
                folder = self.root / "results" / method / obj["uid"]
                status = read(folder / "status.json") or {"state": "PENDING"}
                geometry = read(folder / "geometry.json")
                poses = read(folder / "pose_metrics.json") or read(
                    folder / "gaussian/pose_metrics.json"
                )
                metrics = read(folder / "gaussian/render_metrics.json").get("views", [])
                visuals = sorted((folder / "visuals").glob("*.png"))
                models = sorted((folder / "model_inputs").glob("*/*.png"))
                # Show each preprocessing stage (e.g. detector and descriptor),
                # instead of three images from only the first stage.
                models = list({p.parent: p for p in reversed(models)}.values())
                rows.append(
                    {
                        "uid": obj["uid"],
                        "object": obj["name"],
                        "method": method,
                        "state": status["state"],
                        "error": status.get("error", ""),
                        "registered": geometry.get("registered", status.get("registered")),
                        "points": geometry.get("points3d", geometry.get("initialization_points")),
                        "reprojection_px": geometry.get("reprojection_error_px"),
                        "rotation_median_deg": poses.get("rotation_median_deg"),
                        "center_coordinate_rmse": poses.get("center_rmse"),
                        "seconds": status.get("seconds"),
                        "val_foreground_psnr": mean(
                            [v["foreground_psnr"] for v in metrics if v["split"] == "val"]
                        ),
                        "test_foreground_psnr": mean(
                            [v["foreground_psnr"] for v in metrics if v["split"] == "test"]
                        ),
                        "test_psnr": mean([v["psnr"] for v in metrics if v["split"] == "test"]),
                        "test_ssim": mean([v["ssim"] for v in metrics if v["split"] == "test"]),
                        "matching": [str(p.relative_to(self.root)) for p in visuals],
                        "model_inputs": [str(p.relative_to(self.root)) for p in models],
                        "geometry": str((folder / "geometry.html").relative_to(self.root))
                        if (folder / "geometry.html").exists()
                        else None,
                        "gif": str((folder / "gaussian/reconstruction.gif").relative_to(self.root))
                        if (folder / "gaussian/reconstruction.gif").exists()
                        else None,
                        "path": str(folder),
                    }
                )
            for method in ["3dgs", "3dgs_mcmc", "2dgs", "3dgut_mcmc"]:
                folder = self.gt_root / "results" / method / obj["uid"]
                status, train = read(folder / "status.json"), read(folder / "training.json")
                metrics, val = read(folder / "test_metrics.json"), read(folder / "val_metrics.json")
                progress = read(folder / "progress.json")
                if (
                    status.get("state") == "FAILED"
                    and gt_execution.get("state") in {"STARTING", "RUNNING"}
                    and status.get("checked_at", 0) < gt_execution.get("checked_at", 0)
                ):
                    status = {
                        "state": "WAITING_RETRY",
                        "error": "上次为环境启动失败；修正版已启动，尚未轮到本项。",
                    }
                gt_rows.append(
                    {
                        "uid": obj["uid"],
                        "object": obj["name"],
                        "method": method,
                        "state": status.get("state", "PENDING"),
                        "error": status.get("error", ""),
                        "step": progress.get("step"),
                        "seconds": train.get("seconds"),
                        "best_step": train.get("best_step"),
                        "points": train.get("points"),
                        "val_foreground_psnr": val.get("foreground_psnr"),
                        "test_foreground_psnr": metrics.get("foreground_psnr"),
                        "test_psnr": metrics.get("psnr"),
                        "test_ssim": metrics.get("ssim"),
                        "gif": "../"
                        + str((folder / "reconstruction.gif").relative_to(self.gt_root.parent))
                        if (folder / "reconstruction.gif").exists()
                        else None,
                        "path": str(folder),
                    }
                )
        for path in sorted((self.root / "view_ablation").glob("*/*/*/status.json")):
            method, uid, subset = path.parts[-4:-1]
            status, poses = read(path), read(path.parent / "pose_metrics.json")
            view_rows.append(
                {
                    "method": method,
                    "uid": uid,
                    "subset": subset,
                    "state": status.get("state"),
                    "registered": poses.get("registered"),
                    "rotation_median_deg": poses.get("rotation_median_deg"),
                    "input_views": status.get("input_views"),
                    "seconds": status.get("seconds"),
                    "geometry": str((path.parent / "geometry.html").relative_to(self.root))
                    if (path.parent / "geometry.html").exists()
                    else None,
                    "path": str(path.parent),
                }
            )
        return {
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "execution": read(self.root / "execution_status.json"),
            "objects": self.config["objects"],
            "input_views": self.config.get("input_views", 36),
            "reconstruction": rows,
            "gaussian_gt": gt_rows,
            "view_ablation": view_rows,
            "view_target": self.config.get(
                "view_target", len(self.config["view_ablation_methods"]) * 10 * 9
            ),
            "gt_protocol": self.config.get(
                "gt_protocol",
                "同一训练/验证/测试划分（36/6/6），固定GT相机和LoMa-SfM初始化，各30000步，按验证前景PSNR选checkpoint。与上表不是同预算或同相机条件。",
            ),
        }

    @staticmethod
    def table(rows, columns):
        def fmt(value):
            if value is None:
                return "—"
            return f"{value:.3f}" if isinstance(value, float) else html.escape(str(value))

        return (
            "<table><thead><tr>"
            + "".join(f"<th>{html.escape(label)}</th>" for key, label in columns)
            + "</tr></thead><tbody>"
            + "".join(
                "<tr>"
                + "".join(f"<td>{fmt(row.get(key))}</td>" for key, label in columns)
                + "</tr>"
                for row in rows
            )
            + "</tbody></table>"
        )

    def aggregate(self, rows):
        groups = defaultdict(list)
        for row in rows:
            groups[row["method"]].append(row)
        summary = []
        for method, group in groups.items():
            complete = [x for x in group if x["state"] == "COMPLETE"]
            summary.append(
                {
                    "method": method,
                    "complete": len(complete),
                    "statuses": ", ".join(
                        f"{k}:{v}" for k, v in Counter(x["state"] for x in group).items()
                    ),
                    "registered": mean([x.get("registered") for x in complete]),
                    "rotation": mean([x.get("rotation_median_deg") for x in complete]),
                    "val_fg": mean([x.get("val_foreground_psnr") for x in complete]),
                    "test_fg": mean([x.get("test_foreground_psnr") for x in complete]),
                    "seconds": mean([x.get("seconds") for x in complete]),
                }
            )
        return summary

    def build(self):
        data = self.collect()
        save_json(self.root / "comparison.json", data)
        for key in ["reconstruction", "gaussian_gt", "view_ablation"]:
            rows = data[key]
            if rows:
                with (self.root / (key + ".csv")).open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            elif key == "view_ablation":
                # A newly launched study has zero subset results but still needs
                # a valid downloadable table for the pending-state report.
                with (self.root / (key + ".csv")).open("w", newline="") as stream:
                    csv.writer(stream).writerow(["method", "uid", "subset", "state", "registered", "rotation_median_deg", "input_views", "seconds", "geometry", "path"])
        summary, gt_summary = (
            self.aggregate(data["reconstruction"]),
            self.aggregate(data["gaussian_gt"]),
        )
        from .portable_report import PortableReport

        PortableReport(
            self.root, self.gt_root, [Path(p) for p in self.config.get("artifact_roots", [])]
        ).build(data, summary, gt_summary)
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--watch-seconds", type=int, default=0)
    args = parser.parse_args()
    reporter = ExperimentReport(read(args.config), args.gt_root)
    deadline = time.monotonic() + args.watch_seconds
    while True:
        data = reporter.build()
        print(data["checked_at"], len(data["view_ablation"]), flush=True)
        if time.monotonic() >= deadline:
            break
        time.sleep(120)


if __name__ == "__main__":
    main()
